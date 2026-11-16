"""TIDE-style error breakdown (Cls, Loc, Both, Dupe, Bkg, Miss) and failure gallery miner.

Credits: Error taxonomy follows Bolya et al., "TIDE: A General Toolbox for Identifying Object Detection Errors", ECCV 2020.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.patches as patches
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


def _frame_error_types(
    p_sub: pd.DataFrame, g_sub: pd.DataFrame, pos_iou: float = 0.50, bg_iou: float = 0.10
) -> tuple[list[str], list[int], np.ndarray]:
    """Per-frame TIDE labels: (type of each prediction, closest GT class or -1, matched mask of GT).

    Same rules as `compute_tide_breakdown`; predictions must be sorted by score (descending).
    """
    g_boxes = g_sub[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
    g_cls = g_sub["cls"].to_numpy(dtype=np.int64)
    gt_matched = np.zeros(len(g_boxes), dtype=bool)
    if p_sub.empty:
        return [], [], gt_matched
    p_boxes = p_sub[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
    p_cls = p_sub["cls"].to_numpy(dtype=np.int64)
    if len(g_boxes) == 0:
        return ["Bkg"] * len(p_boxes), [-1] * len(p_boxes), gt_matched

    iou_mat = pairwise_iou_xywh(p_boxes, g_boxes)
    types: list[str] = []
    near_cls: list[int] = []
    for pi in range(len(p_boxes)):
        ious = iou_mat[pi]
        same_ious = np.where(g_cls == p_cls[pi], ious, -1.0)
        gi = int(np.argmax(same_ious))
        best_any = int(np.argmax(ious))
        near_cls.append(int(g_cls[best_any]) if ious[best_any] >= bg_iou else -1)
        if same_ious[gi] >= pos_iou:
            types.append("Dupe" if gt_matched[gi] else "TP")
            gt_matched[gi] = True
        elif ious[best_any] >= pos_iou and g_cls[best_any] != p_cls[pi]:
            types.append("Cls")
        elif same_ious[gi] >= bg_iou:
            types.append("Loc")
        elif ious[best_any] >= bg_iou:
            types.append("Both")
        else:
            types.append("Bkg")
    return types, near_cls, gt_matched


# colours (RGB) of the failure-gallery overlay
_GALLERY_COLORS = {
    "TP": (60, 200, 60),
    "Miss": (255, 215, 0),
    "Cls": (230, 40, 230),
    "Loc": (255, 140, 0),
    "Both": (255, 140, 0),
    "Dupe": (150, 150, 150),
    "Bkg": (235, 30, 30),
}


def _best_crop(centers: np.ndarray, w: int, h: int, frac: float = 0.34) -> tuple[int, int, int, int]:
    """Crop window (x, y, cw, ch) of size frac*frame that contains the most of the given box centres."""
    cw, ch = int(w * frac), int(h * frac)
    best, best_xy = -1, (0, 0)
    for x0 in np.linspace(0, w - cw, 13).astype(int):
        for y0 in np.linspace(0, h - ch, 13).astype(int):
            n = 0
            if len(centers):
                n = int(((centers[:, 0] >= x0) & (centers[:, 0] < x0 + cw)
                         & (centers[:, 1] >= y0) & (centers[:, 1] < y0 + ch)).sum())
            if n > best:
                best, best_xy = n, (int(x0), int(y0))
    return best_xy[0], best_xy[1], cw, ch


def plot_failure_gallery(
    preds_df: pd.DataFrame,
    family_label: str,
    out_path: str | Path,
    split: str = "val",
    conf_thresh: float = 0.30,
) -> Path:
    """Three failure cases mined from the predictions (`images/failures_<family>.jpg`).

    Every val frame is labelled with the TIDE rules; the gallery shows the frame with the most
    missed small objects, the frame with the most wrong-class detections and the frame with the most
    background false positives (each from a different clip), as full frame + zoom on the worst region.
    """
    from vdet.viz import save_photo_figure

    manifest_df = load_manifest()
    gt_val = load_boxes()
    gt_val = gt_val[gt_val["split"] == split]
    val_m = manifest_df[manifest_df["split"] == split].set_index("image_key")

    preds = preds_df[preds_df["score"] >= conf_thresh]
    preds = filter_ignore_predictions(preds[preds["image_key"].isin(val_m.index)])
    preds = preds.sort_values(["image_key", "score"], ascending=[True, False])
    pred_groups = {str(k): g for k, g in preds.groupby("image_key", sort=False)}
    gt_groups = {str(k): g for k, g in gt_val.groupby("image_key", sort=False)}

    empty_p = preds.iloc[0:0]
    rows = []
    for key, g_sub in gt_groups.items():
        p_sub = pred_groups.get(key, empty_p)
        types, _, matched = _frame_error_types(p_sub, g_sub)
        small = (g_sub["w"].to_numpy() * g_sub["h"].to_numpy()) < 32**2
        rows.append({"image_key": key, "clip": val_m.loc[key, "clip"], "n_gt": len(g_sub),
                     "miss": int((~matched).sum()), "miss_small": int((~matched & small).sum()),
                     "Cls": types.count("Cls"), "Bkg": types.count("Bkg")})
    stats = pd.DataFrame(rows)

    cases = [("miss_small", "Missed small objects", "Miss"),
             ("Cls", "Wrong class", "Cls"),
             ("Bkg", "Background false positives", "Bkg")]
    used_clips: set[str] = set()
    fig, axes = plt.subplots(len(cases), 2, figsize=(14, 4.3 * len(cases)),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.18})
    for r, (col, label, focus) in enumerate(cases):
        cand = stats[~stats["clip"].isin(used_clips)].sort_values(col, ascending=False)
        st = cand.iloc[0]
        used_clips.add(st["clip"])
        key = st["image_key"]
        g_sub = gt_groups[key]
        p_sub = pred_groups.get(key, empty_p)
        types, near_cls, matched = _frame_error_types(p_sub, g_sub)

        bgr = cv2.imread((P.data / str(val_m.loc[key, "path"])).as_posix(), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        th = max(2, int(round(w / 900)))

        def draw(box, kind):
            x, y, bw, bh = (int(round(v)) for v in box)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), _GALLERY_COLORS[kind], th)

        for box in g_sub.loc[~matched, ["x", "y", "w", "h"]].to_numpy():
            draw(box, "Miss")
        focus_centers = []
        for box, kind in zip(p_sub[["x", "y", "w", "h"]].to_numpy(), types):
            draw(box, kind)
            if kind == focus:
                focus_centers.append((box[0] + box[2] / 2, box[1] + box[3] / 2))
        if focus == "Miss":
            mb = g_sub.loc[~matched].copy()
            mb = mb[(mb["w"] * mb["h"]) < 32**2]
            focus_centers = list(zip(mb["x"] + mb["w"] / 2, mb["y"] + mb["h"] / 2))
        cx, cy, cw, ch = _best_crop(np.array(focus_centers, dtype=np.float64).reshape(-1, 2), w, h)

        # label wrong-class boxes inside the crop: "predicted (true)"
        if focus == "Cls":
            for box, kind, gc, pc in zip(p_sub[["x", "y", "w", "h"]].to_numpy(), types, near_cls,
                                         p_sub["cls"].to_numpy()):
                if kind == "Cls" and cx <= box[0] < cx + cw and cy <= box[1] < cy + ch:
                    txt = f"{CLASS_NAMES[pc]} ({CLASS_NAMES[gc]})"
                    org = (int(box[0]), int(box[1]) - 2)
                    (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.22 * th, 1)
                    cv2.rectangle(img, (org[0], org[1] - tht - 1), (org[0] + tw, org[1] + 1), (0, 0, 0), -1)
                    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.22 * th, (255, 255, 255), 1, cv2.LINE_AA)

        crop = img[cy:cy + ch, cx:cx + cw].copy()
        full = img.copy()
        cv2.rectangle(full, (cx, cy), (cx + cw, cy + ch), (255, 255, 255), th * 3)
        scale = 1280.0 / max(w, h)
        full = cv2.resize(full, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        axes[r, 0].imshow(full)
        axes[r, 1].imshow(crop)
        for ax in axes[r]:
            ax.set_axis_off()
        axes[r, 0].set_title(
            f"Case {r + 1}: {label} | {key}\n"
            f"{st['miss']}/{st['n_gt']} GT missed ({st['miss_small']} small), "
            f"{st['Cls']} wrong-class, {st['Bkg']} background FPs",
            fontsize=10, loc="left")
        axes[r, 1].set_title("zoom on the white rectangle" + (" — labels: predicted (true)" if focus == "Cls" else ""),
                             fontsize=10, loc="left")

    handles = [patches.Patch(facecolor="none", edgecolor=np.array(c) / 255.0, linewidth=2, label=l)
               for l, c in [("correct detection", _GALLERY_COLORS["TP"]), ("missed GT", _GALLERY_COLORS["Miss"]),
                            ("wrong class", _GALLERY_COLORS["Cls"]), ("poor localisation", _GALLERY_COLORS["Loc"]),
                            ("duplicate", _GALLERY_COLORS["Dupe"]), ("background FP", _GALLERY_COLORS["Bkg"])]]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Failure cases — {family_label} (val, detections with score ≥ {conf_thresh:.2f})",
                 fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.04)
    return save_photo_figure(fig, out_path)
