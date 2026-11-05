"""Visualisation bonus utilities: Multi-Scale CNN EigenCAM/Grad-CAM (U5.3), RT-DETR Attention (U5.4), and UMAP Embeddings (U5.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from vdet.constants import CLASS_NAMES
from vdet.data.manifest import load_boxes, load_manifest
from vdet.paths import P
from vdet.temporal.linking import pairwise_iou_xywh
from vdet.viz import save_figure_under_budget


def _compute_feature_eigencam(feat_tensor: torch.Tensor, target_hw: tuple[int, int]) -> np.ndarray:
    """Compute first principal component activation map (EigenCAM) from a `(1, C, H, W)` feature tensor."""
    f = feat_tensor.detach().float()[0]  # (C, H, W)
    c, h, w = f.shape
    flat = f.reshape(c, -1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    # Power iteration / SVD first principal component across channels
    u, _, _ = torch.linalg.svd(flat, full_matrices=False)
    cam = (u[:, 0:1].T @ flat).reshape(h, w).cpu().numpy()
    cam = np.abs(cam)
    cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-6)
    cam_resized = cv2.resize(cam.astype(np.float32), (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(cam_resized, 0.0, 1.0)


def _overlay_heatmap(rgb_img: np.ndarray, cam_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_bgr = cv2.applyColorMap((cam_01 * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = (1.0 - alpha) * rgb_img.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_yolo_multiscale_cam(
    weights_path: str | Path,
    out_path: str | Path,
    image_key: str | None = None,
) -> Path:
    """Hook YOLO11 neck feature maps (`P3/8` small, `P4/16` medium, `P5/32` large) and render EigenCAM overlays (U5.3)."""
    from ultralytics import YOLO

    manifest_df = load_manifest()
    val_m = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
    row = val_m[val_m["image_key"] == image_key].iloc[0] if image_key else val_m.iloc[0]

    bgr = cv2.imread((P.data / str(row["path"])).as_posix(), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    disp_h, disp_w = 270, 480
    rgb_disp = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    model = YOLO(str(weights_path))
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    # Layers 16 (P3/8), 19 (P4/16), 22 (P5/32) in YOLO11 neck feed the Detect head
    for name, layer_idx in [("P3/8 (Small Objects)", 16), ("P4/16 (Medium Objects)", 19), ("P5/32 (Large Objects)", 22)]:
        mod = model.model.model[layer_idx]
        hooks.append(
            mod.register_forward_hook(lambda _m, _inp, out, k=name: captured.setdefault(k, out if isinstance(out, torch.Tensor) else out[0]))
        )

    try:
        _ = model.predict(source=bgr, imgsz=1280, conf=0.25, verbose=False)
    finally:
        for h in hooks:
            h.remove()

    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.8))
    axes[0].imshow(rgb_disp)
    axes[0].set_title(f"Original Frame\n({row['image_key']})", fontsize=8)
    axes[0].axis("off")

    for ax, (lvl_name, feat) in zip(axes[1:], captured.items()):
        cam = _compute_feature_eigencam(feat, (disp_h, disp_w))
        ov = _overlay_heatmap(rgb_disp, cam, alpha=0.48)
        ax.imshow(ov)
        ax.set_title(f"EigenCAM: {lvl_name}", fontsize=8)
        ax.axis("off")

    fig.suptitle("U5.3 Multi-Scale CNN Feature Activation Maps (P3/8 vs P4/16 vs P5/32)", fontsize=9.5, y=1.02)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path, dpi=110)


def generate_rtdetr_attention_map(
    weights_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Visualize RT-DETR encoder-decoder cross-attention and query spatial density on sparse vs dense frames (U5.4)."""
    from ultralytics import RTDETR

    manifest_df = load_manifest()
    boxes_df = load_boxes()
    val_m = manifest_df[manifest_df["split"] == "val"].set_index("image_key")
    gt_counts = boxes_df[boxes_df["split"] == "val"].groupby("image_key").size()
    dense_key = str(gt_counts.idxmax())
    sparse_key = str(gt_counts[gt_counts >= 8].idxmin())

    model = RTDETR(str(weights_path))
    disp_h, disp_w = 260, 460
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.0))

    for row_idx, (label, key) in enumerate([("Dense Traffic Scene", dense_key), ("Sparse Scene", sparse_key)]):
        r = val_m.loc[key]
        bgr = cv2.imread((P.data / str(r["path"])).as_posix(), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_disp = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        captured_enc: list[torch.Tensor] = []
        # Layer 21 is X3 (high-res encoder feature feeding RTDETRDecoder)
        h_hook = model.model.model[21].register_forward_hook(
            lambda _m, _inp, out: captured_enc.append(out if isinstance(out, torch.Tensor) else out[0])
        )
        try:
            res = model.predict(source=bgr, imgsz=1280, conf=0.01, max_det=300, verbose=False)[0]
        finally:
            h_hook.remove()

        cam = _compute_feature_eigencam(captured_enc[0], (disp_h, disp_w)) if captured_enc else np.zeros((disp_h, disp_w), dtype=np.float32)
        ov = _overlay_heatmap(rgb_disp, cam, alpha=0.45)

        axes[row_idx, 0].imshow(ov)
        axes[row_idx, 0].set_title(f"{label} ({key}): Encoder-Decoder Feature Attention", fontsize=8)
        axes[row_idx, 0].axis("off")

        # Plot spatial query sampling centers weighted by query confidence
        query_canvas = rgb_disp.copy()
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            confs = res.boxes.conf.detach().cpu().numpy()
            sx = disp_w / float(rgb.shape[1])
            sy = disp_h / float(rgb.shape[0])
            for box, sc in zip(xyxy[:150], confs[:150]):
                cx = int((box[0] + box[2]) * 0.5 * sx)
                cy = int((box[1] + box[3]) * 0.5 * sy)
                radius = max(2, int(round(2 + 5 * float(sc))))
                color = (255, int(255 * (1.0 - min(1.0, sc))), 30)
                cv2.circle(query_canvas, (cx, cy), radius, color, -1)

        axes[row_idx, 1].imshow(query_canvas)
        axes[row_idx, 1].set_title(f"{label}: Top-150 Decoder Object Query Centers", fontsize=8)
        axes[row_idx, 1].axis("off")

    fig.suptitle("U5.4 RT-DETR Deformable Cross-Attention & Decoder Query Focus", fontsize=9.5, y=1.01)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path, dpi=110)


def generate_umap_embeddings_plot(
    preds_df: pd.DataFrame,
    out_path: str | Path,
    max_samples: int = 2500,
    seed: int = 42,
) -> Path:
    """Extract multi-scale region descriptors + box geometry for val detections and project via UMAP (U5.5)."""
    import umap

    manifest_df = load_manifest()
    boxes_df = load_boxes()
    val_m = manifest_df[manifest_df["split"] == "val"].set_index("image_key")
    gt_val = boxes_df[boxes_df["split"] == "val"]
    gt_groups = {str(k): grp for k, grp in gt_val.groupby("image_key", sort=False)}

    sub_preds = preds_df[preds_df["score"] >= 0.20].copy()
    if len(sub_preds) > max_samples:
        sub_preds = sub_preds.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    features: list[np.ndarray] = []
    classes: list[int] = []
    size_buckets: list[str] = []
    outcomes: list[str] = []

    img_cache: dict[str, np.ndarray] = {}
    for key, grp in sub_preds.groupby("image_key", sort=False):
        if str(key) not in val_m.index:
            continue
        row = val_m.loc[str(key)]
        bgr = cv2.imread((P.data / str(row["path"])).as_posix(), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_img, w_img = bgr.shape[:2]

        g_sub = gt_groups.get(str(key))
        g_boxes = g_sub[["x", "y", "w", "h"]].to_numpy(dtype=np.float64) if (g_sub is not None and not g_sub.empty) else np.zeros((0, 4))
        g_cls = g_sub["cls"].to_numpy(dtype=np.int64) if (g_sub is not None and not g_sub.empty) else np.zeros((0,), dtype=np.int64)

        p_boxes = grp[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
        ious = pairwise_iou_xywh(p_boxes, g_boxes) if len(g_boxes) > 0 else np.zeros((len(p_boxes), 0))

        for i, r in enumerate(grp.itertuples(index=False)):
            x1 = max(0, min(int(r.x), w_img - 2))
            y1 = max(0, min(int(r.y), h_img - 2))
            x2 = max(x1 + 1, min(int(r.x + r.w), w_img))
            y2 = max(y1 + 1, min(int(r.y + r.h), h_img))

            patch = cv2.resize(hsv[y1:y2, x1:x2], (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            geom = np.array(
                [
                    r.x / w_img,
                    r.y / h_img,
                    np.log1p(r.w) / 8.0,
                    np.log1p(r.h) / 8.0,
                    (r.h / max(r.w, 1.0)) * 0.25,
                    float(r.score),
                ],
                dtype=np.float32,
            )
            feat_vec = np.concatenate([patch.reshape(-1), geom])
            features.append(feat_vec)
            classes.append(int(r.cls))

            area = float(r.w * r.h)
            if area < 32**2:
                size_buckets.append("small (<32²)")
            elif area <= 96**2:
                size_buckets.append("medium (32²-96²)")
            else:
                size_buckets.append("large (>96²)")

            is_tp = False
            if ious.shape[1] > 0:
                best_gi = int(np.argmax(ious[i]))
                if ious[i, best_gi] >= 0.50 and int(g_cls[best_gi]) == int(r.cls):
                    is_tp = True
            outcomes.append("TP (Matched)" if is_tp else "FP (False Alarm)")

    X = np.stack(features, axis=0)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.15, random_state=seed)
    emb = reducer.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))

    # Panel 1: By Class
    cls_arr = np.asarray(classes)
    cmap = plt.get_cmap("tab10")
    for c_id, c_name in enumerate(CLASS_NAMES):
        m = cls_arr == c_id
        if np.any(m):
            axes[0].scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.7, color=cmap(c_id), label=c_name)
    axes[0].set_title("Coloured by Object Class", fontsize=8.5)
    axes[0].legend(fontsize=6.5, markerscale=2, loc="best")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel 2: By Size Bucket
    sz_arr = np.asarray(size_buckets)
    for s_lbl, col in [("small (<32²)", "#e63946"), ("medium (32²-96²)", "#457b9d"), ("large (>96²)", "#2a9d8f")]:
        m = sz_arr == s_lbl
        if np.any(m):
            axes[1].scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.7, color=col, label=s_lbl)
    axes[1].set_title("Coloured by COCO Size Bucket", fontsize=8.5)
    axes[1].legend(fontsize=7, markerscale=2, loc="best")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Panel 3: By TP vs FP
    out_arr = np.asarray(outcomes)
    for o_lbl, col in [("TP (Matched)", "#2a9d8f"), ("FP (False Alarm)", "#e76f51")]:
        m = out_arr == o_lbl
        if np.any(m):
            axes[2].scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.7, color=col, label=o_lbl)
    axes[2].set_title("Coloured by Outcome (TP vs FP)", fontsize=8.5)
    axes[2].legend(fontsize=7, markerscale=2, loc="best")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.suptitle(f"U5.5 UMAP Projection of Detection Embeddings (n={len(X)} val detections)", fontsize=9.5, y=1.02)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path, dpi=110)

