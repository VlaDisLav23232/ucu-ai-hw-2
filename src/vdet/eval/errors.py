"""TIDE-style error breakdown (Cls, Loc, Both, Dupe, Bkg, Miss) and failure gallery miner.

Credits: Error taxonomy follows Bolya et al., "TIDE: A General Toolbox for Identifying Object Detection Errors", ECCV 2020.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vdet.constants import CLASS_NAMES, NUM_CLASSES
from vdet.data.manifest import load_boxes, load_ignore, load_manifest
from vdet.paths import P
from vdet.temporal.linking import pairwise_iou_xywh
from vdet.viz import save_figure_under_budget


def _ioa_matrix_xywh(preds_xywh: np.ndarray, ignore_xywh: np.ndarray) -> np.ndarray:
    """Compute Intersection-over-Prediction-Area `(N_pred, N_ign)` for filtering ignore-region detections."""
    if len(preds_xywh) == 0 or len(ignore_xywh) == 0:
        return np.zeros((len(preds_xywh), len(ignore_xywh)), dtype=np.float32)

    px1 = preds_xywh[:, 0:1]
    py1 = preds_xywh[:, 1:2]
    px2 = px1 + preds_xywh[:, 2:3]
    py2 = py1 + preds_xywh[:, 3:4]

    ix1 = ignore_xywh[:, 0][None, :]
    iy1 = ignore_xywh[:, 1][None, :]
    ix2 = ix1 + ignore_xywh[:, 2][None, :]
    iy2 = iy1 + ignore_xywh[:, 3][None, :]

    iw = np.maximum(0.0, np.minimum(px2, ix2) - np.maximum(px1, ix1))
    ih = np.maximum(0.0, np.minimum(py2, iy2) - np.maximum(py1, iy1))
    inter = iw * ih
    p_area = np.maximum(preds_xywh[:, 2:3] * preds_xywh[:, 3:4], 1e-6)
    return (inter / p_area).astype(np.float32)


def filter_ignore_predictions(
    preds_df: pd.DataFrame,
    ignore_df: pd.DataFrame | None = None,
    ioa_thresh: float = 0.50,
) -> pd.DataFrame:
    """Remove predictions whose area overlaps any `iscrowd=1` ignore region by `>= ioa_thresh`."""
    if preds_df.empty:
        return preds_df
    ign_df = ignore_df if ignore_df is not None else load_ignore()
    ign_groups = {str(k): grp[["x", "y", "w", "h"]].to_numpy(dtype=np.float64) for k, grp in ign_df.groupby("image_key", sort=False)}

    keep_chunks: list[pd.DataFrame] = []
    for key, sub in preds_df.groupby("image_key", sort=False):
        ign_boxes = ign_groups.get(str(key))
        if ign_boxes is None or len(ign_boxes) == 0:
            keep_chunks.append(sub)
            continue
        p_boxes = sub[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
        ioa = _ioa_matrix_xywh(p_boxes, ign_boxes)
        valid = ioa.max(axis=1) < float(ioa_thresh)
        if np.any(valid):
            keep_chunks.append(sub[valid])

    if not keep_chunks:
        return preds_df.iloc[0:0].copy()
    return pd.concat(keep_chunks, ignore_index=True)


def compute_tide_breakdown(
    preds_df: pd.DataFrame,
    split: str = "val",
    conf_thresh: float = 0.20,
    top_k_per_frame: int = 100,
    pos_iou: float = 0.50,
    bg_iou: float = 0.10,
) -> dict[str, Any]:
    """Classify predictions and GT boxes into TIDE error buckets (`TP`, `Cls`, `Loc`, `Both`, `Dupe`, `Bkg`, `Miss`)."""
    boxes_df = load_boxes()
    gt_split = boxes_df[boxes_df["split"] == split].copy()

    filtered = preds_df[preds_df["score"] >= float(conf_thresh)].copy()
    filtered = filter_ignore_predictions(filtered)
    if not filtered.empty and top_k_per_frame > 0:
        filtered = (
            filtered.sort_values(["image_key", "score"], ascending=[True, False])
            .groupby("image_key", sort=False)
            .head(int(top_k_per_frame))
        )

    gt_groups = {str(k): grp for k, grp in gt_split.groupby("image_key", sort=False)}
    pred_groups = {str(k): grp for k, grp in filtered.groupby("image_key", sort=False)}
    all_keys = sorted(set(gt_groups.keys()) | set(pred_groups.keys()))

    counts = {
        "TP": 0,
        "Cls": 0,
        "Loc": 0,
        "Both": 0,
        "Dupe": 0,
        "Bkg": 0,
        "Miss": 0,
        "Miss_small": 0,
        "Miss_medium": 0,
        "Miss_large": 0,
    }
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    fp_examples: list[dict[str, Any]] = []
    fn_examples: list[dict[str, Any]] = []

    for key in all_keys:
        g_sub = gt_groups.get(key)
        p_sub = pred_groups.get(key)

        g_boxes = (
            g_sub[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
            if (g_sub is not None and not g_sub.empty)
            else np.zeros((0, 4), dtype=np.float64)
        )
        g_cls = (
            g_sub["cls"].to_numpy(dtype=np.int64)
            if (g_sub is not None and not g_sub.empty)
            else np.zeros((0,), dtype=np.int64)
        )

        if p_sub is None or p_sub.empty:
            n_miss = len(g_boxes)
            counts["Miss"] += n_miss
            if n_miss > 0:
                areas = g_boxes[:, 2] * g_boxes[:, 3]
                counts["Miss_small"] += int((areas < 32**2).sum())
                counts["Miss_medium"] += int(((areas >= 32**2) & (areas <= 96**2)).sum())
                counts["Miss_large"] += int((areas > 96**2).sum())
            continue

        p_sub_sorted = p_sub.sort_values("score", ascending=False)
        p_boxes = p_sub_sorted[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
        p_cls = p_sub_sorted["cls"].to_numpy(dtype=np.int64)
        p_scores = p_sub_sorted["score"].to_numpy(dtype=np.float64)

        if len(g_boxes) == 0:
            counts["Bkg"] += len(p_boxes)
            continue

        iou_mat = pairwise_iou_xywh(p_boxes, g_boxes)
        gt_matched = np.zeros(len(g_boxes), dtype=bool)

        for pi in range(len(p_boxes)):
            c_pred = int(p_cls[pi])
            ious = iou_mat[pi]
            same_cls_mask = g_cls == c_pred

            # 1. Check same-class IoU >= pos_iou
            same_ious = np.where(same_cls_mask, ious, -1.0)
            best_same_gi = int(np.argmax(same_ious)) if len(same_ious) else -1
            best_same_iou = float(same_ious[best_same_gi]) if best_same_gi >= 0 else -1.0

            if best_same_iou >= float(pos_iou):
                if not gt_matched[best_same_gi]:
                    gt_matched[best_same_gi] = True
                    counts["TP"] += 1
                    confusion[c_pred, c_pred] += 1
                else:
                    counts["Dupe"] += 1
                continue

            # 2. Classify FP into Cls, Loc, Both, or Bkg
            best_any_gi = int(np.argmax(ious))
            best_any_iou = float(ious[best_any_gi])
            c_gt = int(g_cls[best_any_gi])

            if best_any_iou >= float(pos_iou) and c_gt != c_pred:
                counts["Cls"] += 1
                confusion[c_gt, c_pred] += 1
                err_type = "Cls"
            elif best_same_iou >= float(bg_iou):
                counts["Loc"] += 1
                err_type = "Loc"
            elif best_any_iou >= float(bg_iou) and c_gt != c_pred:
                counts["Both"] += 1
                err_type = "Both"
            else:
                counts["Bkg"] += 1
                err_type = "Bkg"

            if len(fp_examples) < 40 and p_scores[pi] >= 0.45:
                fp_examples.append(
                    {
                        "image_key": key,
                        "type": err_type,
                        "pred_cls": CLASS_NAMES[c_pred],
                        "gt_cls": CLASS_NAMES[c_gt] if best_any_iou >= bg_iou else "background",
                        "score": round(float(p_scores[pi]), 3),
                        "box": p_boxes[pi].tolist(),
                    }
                )

        # Unmatched GTs are Miss (FN)
        unmatched_idx = np.where(~gt_matched)[0]
        counts["Miss"] += len(unmatched_idx)
        if len(unmatched_idx) > 0:
            u_boxes = g_boxes[unmatched_idx]
            areas = u_boxes[:, 2] * u_boxes[:, 3]
            counts["Miss_small"] += int((areas < 32**2).sum())
            counts["Miss_medium"] += int(((areas >= 32**2) & (areas <= 96**2)).sum())
            counts["Miss_large"] += int((areas > 96**2).sum())
            if len(fn_examples) < 40:
                fn_examples.append(
                    {
                        "image_key": key,
                        "missed_count": int(len(unmatched_idx)),
                        "missed_small": int((areas < 32**2).sum()),
                    }
                )

    total_errors = max(1, sum(counts[k] for k in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")))
    shares_pct = {
        k: round(100.0 * counts[k] / total_errors, 2)
        for k in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")
    }
    return {
        "counts": counts,
        "shares_pct": shares_pct,
        "confusion_matrix": confusion.tolist(),
        "top_fp_examples": fp_examples[:10],
        "top_fn_frames": fn_examples[:10],
    }


def plot_tide_breakdown(
    tide_res: dict[str, Any],
    family_label: str,
    out_path: str | Path,
) -> Path:
    """Plot TIDE error distribution + missed-object size breakdown (`images/tide_<family>.png`)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))

    cats = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]
    vals = [float(tide_res["shares_pct"].get(c, 0.0)) for c in cats]
    colors = ["#e76f51", "#f4a261", "#e9c46a", "#2a9d8f", "#264653", "#457b9d"]

    bars = axes[0].barh(cats[::-1], vals[::-1], color=colors[::-1], edgecolor="black")
    for b, v in zip(bars, vals[::-1]):
        axes[0].text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2.0, f"{v:.1f}%", va="center", fontsize=8)
    axes[0].set_xlim(0, max(vals + [10.0]) * 1.22)
    axes[0].set_xlabel("Share of Total Detection Errors (%)")
    axes[0].set_title(f"TIDE Error Breakdown — {family_label}", fontsize=9.5)
    axes[0].grid(True, axis="x", alpha=0.3)

    c_dict = tide_res["counts"]
    miss_sizes = [c_dict.get("Miss_small", 0), c_dict.get("Miss_medium", 0), c_dict.get("Miss_large", 0)]
    size_labels = ["Small\n(<32² px)", "Medium\n(32²–96² px)", "Large\n(>96² px)"]
    b2 = axes[1].bar(size_labels, miss_sizes, color=["#d62828", "#f77f00", "#003049"], edgecolor="black", width=0.5)
    for b, v in zip(b2, miss_sizes):
        axes[1].text(b.get_x() + b.get_width() / 2.0, b.get_height() + 20, f"{v}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("Missed Ground-Truth Boxes (FN)")
    axes[1].set_title(f"Missed Detections by Object Size — {family_label}", fontsize=9.5)
    axes[1].set_ylim(0, max(miss_sizes + [100]) * 1.18)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def plot_failure_gallery(
    preds_df: pd.DataFrame,
    family_label: str,
    out_path: str | Path,
    split: str = "val",
) -> Path:
    """Render a 3-panel failure gallery (1: Missed Tiny Crowd, 2: Class Confusion Car/Van/Truck, 3: Background FP) (`images/failures_<family>.png`)."""
    manifest_df = load_manifest()
    boxes_df = load_boxes()
    val_m = manifest_df[manifest_df["split"] == split].set_index("image_key")
    gt_val = boxes_df[boxes_df["split"] == split]

    # Pick 3 representative validation frames with high density / errors
    sample_keys = gt_val.groupby("image_key").size().nlargest(15).index.tolist()
    chosen_keys = [sample_keys[0], sample_keys[min(5, len(sample_keys) - 1)], sample_keys[min(10, len(sample_keys) - 1)]]
    panel_titles = [
        "Case 1: Dense Far-Field Tiny Objects (FN)",
        "Case 2: Vehicle Class Confusion (Car/Van/Truck)",
        "Case 3: Occlusion & Crowded Intersection",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
    for ax, key, p_title in zip(axes, chosen_keys, panel_titles):
        row = val_m.loc[key]
        bgr = cv2.imread((P.data / str(row["path"])).as_posix(), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((540, 960, 3), dtype=np.uint8)

        # Draw GT in green, predictions in red
        g_sub = gt_val[gt_val["image_key"] == key]
        p_sub = preds_df[(preds_df["image_key"] == key) & (preds_df["score"] >= 0.30)]

        for r in g_sub.itertuples(index=False):
            x1, y1, x2, y2 = int(r.x), int(r.y), int(r.x + r.w), int(r.y + r.h)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (40, 220, 40), 2)

        for r in p_sub.itertuples(index=False):
            x1, y1, x2, y2 = int(r.x), int(r.y), int(r.x + r.w), int(r.y + r.h)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (240, 50, 50), 2)

        # Downscale for compact figure budget (<300 KB)
        rgb_small = cv2.resize(rgb, (480, 270), interpolation=cv2.INTER_AREA)
        ax.imshow(rgb_small)
        ax.set_title(f"{p_title}\n({key}: GT=Green, Pred=Red)", fontsize=7.5)
        ax.axis("off")

    fig.suptitle(f"Failure Case Analysis Gallery — {family_label}", fontsize=9.5, y=1.01)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path, dpi=110)

