"""Multi-model and single-model box fusion: NMS, Soft-NMS, and Weighted Box Fusion (WBF).

Credits: Uses `ensemble_boxes` (Solovyev et al., "Weighted boxes fusion: Ensembling boxes from different object detection models", Image and Vision Computing, 2021).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from ensemble_boxes import nms, soft_nms, weighted_boxes_fusion

from vdet.constants import MAX_SUBMISSION_DETS
from vdet.data.manifest import load_manifest
from vdet.infer.predictors import (
    clip_and_filter_boxes_xywh,
    empty_predictions_df,
    normalize_predictions_df,
)


def _df_to_normalized_xyxy(
    df: pd.DataFrame,
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a single-frame prediction DataFrame (`x, y, w, h` in pixels) to `[0, 1]` `xyxy` arrays for `ensemble_boxes`."""
    if df.empty:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    x1 = np.clip(df["x"].to_numpy(dtype=np.float32) / width, 0.0, 1.0)
    y1 = np.clip(df["y"].to_numpy(dtype=np.float32) / height, 0.0, 1.0)
    x2 = np.clip((df["x"].to_numpy(dtype=np.float32) + df["w"].to_numpy(dtype=np.float32)) / width, 0.0, 1.0)
    y2 = np.clip((df["y"].to_numpy(dtype=np.float32) + df["h"].to_numpy(dtype=np.float32)) / height, 0.0, 1.0)

    valid = (x2 > x1 + 1e-6) & (y2 > y1 + 1e-6)
    if not np.any(valid):
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    boxes = np.stack([x1[valid], y1[valid], x2[valid], y2[valid]], axis=1)
    scores = np.clip(df["score"].to_numpy(dtype=np.float32)[valid], 1e-6, 1.0)
    labels = df["cls"].to_numpy(dtype=np.int32)[valid]
    return boxes, scores, labels


# Default class-aware WBF IoU thresholds tailored to VisDrone-VID object geometry:
# - Crowded/deformable small classes (pedestrian=0, people=1, bicycle=2): 0.48 to merge cross-model jitter
# - Rigid dense vehicles (car=3, van=4, truck=5): 0.57 to avoid merging adjacent parked vehicles
# - Three-wheelers (tricycle=6, awning-tricycle=7): 0.52
DEFAULT_CLASS_IOU_THR: dict[int, float] = {
    0: 0.48,  # pedestrian
    1: 0.47,  # people
    2: 0.48,  # bicycle
    3: 0.57,  # car
    4: 0.57,  # van
    5: 0.58,  # truck
    6: 0.52,  # tricycle
    7: 0.52,  # awning-tricycle
}

# Default per-class score multipliers to protect rare classes under the 500-det/frame budget
DEFAULT_CLASS_SCORE_MULTS: dict[int, float] = {
    0: 1.00,  # pedestrian
    1: 1.08,  # people
    2: 1.08,  # bicycle
    3: 0.92,  # car (abundant; prevent low-score car clutter from crowding out rare classes at maxDets=500)
    4: 1.06,  # van
    5: 1.06,  # truck
    6: 1.10,  # tricycle
    7: 1.15,  # awning-tricycle
}


def fuse_predictions(
    preds_list: Sequence[pd.DataFrame],
    weights: Sequence[float] | None = None,
    method: str = "wbf",
    iou_thr: float = 0.55,
    class_iou_thr: dict[int, float] | None = None,
    class_score_mults: dict[int, float] | None = None,
    skip_box_thr: float = 0.001,
    sigma: float = 0.5,
    conf_type: str = "avg",
    max_det: int = MAX_SUBMISSION_DETS,
    manifest_df: pd.DataFrame | None = None,
    source_tag: str | None = None,
) -> pd.DataFrame:
    """Fuse one or more prediction tables per frame using `wbf`, `soft_nms`, or `nms`.

    Supports optional `class_iou_thr` (per-class WBF IoU thresholds) and `class_score_mults`
    (per-class score multipliers to prioritize rare classes under the 500-det/frame cap).
    """
    if not preds_list:
        return empty_predictions_df()

    m_df = manifest_df if manifest_df is not None else load_manifest()
    dims_map: dict[str, tuple[float, float]] = {
        str(r.image_key): (float(r.width), float(r.height))
        for r in m_df[["image_key", "width", "height"]].itertuples(index=False)
    }

    n_models = len(preds_list)
    model_weights = [float(w) for w in weights] if weights is not None else [1.0] * n_models
    if len(model_weights) != n_models:
        raise ValueError(f"Expected {n_models} weights, got {len(model_weights)}")

    # Group each model's predictions by image_key once for fast O(1) lookup
    grouped_models: list[dict[str, pd.DataFrame]] = []
    all_keys_set: set[str] = set()
    for df in preds_list:
        if df is None or df.empty:
            grouped_models.append({})
            continue
        if float(skip_box_thr) > 0.0:
            df = df[df["score"] >= float(skip_box_thr)]
        grp = {str(k): sub for k, sub in df.groupby("image_key", sort=False)}
        grouped_models.append(grp)
        all_keys_set.update(grp.keys())

    if not all_keys_set:
        return empty_predictions_df()

    method_lower = method.lower()
    tag = source_tag or f"fuse_{method_lower}"
    out_frames: list[pd.DataFrame] = []

    # Group classes by identical IoU threshold so we only invoke WBF once per distinct IoU bucket
    iou_buckets: dict[float, set[int]] = {}
    if class_iou_thr:
        for cls_id in range(8):
            thr = round(float(class_iou_thr.get(cls_id, iou_thr)), 4)
            iou_buckets.setdefault(thr, set()).add(cls_id)

    for key in sorted(all_keys_set):
        orig_w, orig_h = dims_map.get(key, (1920.0, 1080.0))
        boxes_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []

        for grp in grouped_models:
            sub = grp.get(key)
            if sub is None or sub.empty:
                boxes_list.append(np.zeros((0, 4), dtype=np.float32))
                scores_list.append(np.zeros((0,), dtype=np.float32))
                labels_list.append(np.zeros((0,), dtype=np.int32))
            else:
                b, s, l = _df_to_normalized_xyxy(sub, orig_w, orig_h)
                boxes_list.append(b)
                scores_list.append(s)
                labels_list.append(l)

        if all(len(b) == 0 for b in boxes_list):
            continue

        if method_lower == "wbf" and iou_buckets:
            fb_parts: list[np.ndarray] = []
            fs_parts: list[np.ndarray] = []
            fl_parts: list[np.ndarray] = []
            for bucket_iou, cls_set in iou_buckets.items():
                b_sub: list[np.ndarray] = []
                s_sub: list[np.ndarray] = []
                l_sub: list[np.ndarray] = []
                has_any = False
                for b, s, l in zip(boxes_list, scores_list, labels_list):
                    if len(l) == 0:
                        b_sub.append(b)
                        s_sub.append(s)
                        l_sub.append(l)
                    else:
                        mask = np.isin(l, list(cls_set))
                        if np.any(mask):
                            has_any = True
                        b_sub.append(b[mask])
                        s_sub.append(s[mask])
                        l_sub.append(l[mask])
                if not has_any:
                    continue
                fb, fs, fl = weighted_boxes_fusion(
                    b_sub,
                    s_sub,
                    l_sub,
                    weights=model_weights,
                    iou_thr=float(bucket_iou),
                    skip_box_thr=float(skip_box_thr),
                    conf_type=conf_type,
                )
                if len(fb) > 0:
                    fb_parts.append(fb)
                    fs_parts.append(fs)
                    fl_parts.append(fl)
            if not fb_parts:
                continue
            f_boxes = np.concatenate(fb_parts, axis=0)
            f_scores = np.concatenate(fs_parts, axis=0)
            f_labels = np.concatenate(fl_parts, axis=0)
        elif method_lower == "wbf":
            f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=model_weights,
                iou_thr=float(iou_thr),
                skip_box_thr=float(skip_box_thr),
                conf_type=conf_type,
            )
        elif method_lower in {"soft_nms", "softnms"}:
            f_boxes, f_scores, f_labels = soft_nms(
                boxes_list,
                scores_list,
                labels_list,
                weights=model_weights,
                iou_thr=float(iou_thr),
                sigma=float(sigma),
                thresh=float(skip_box_thr),
            )
        elif method_lower == "nms":
            f_boxes, f_scores, f_labels = nms(
                boxes_list,
                scores_list,
                labels_list,
                weights=model_weights,
                iou_thr=float(iou_thr),
            )
        else:
            raise ValueError(f"Unsupported fusion method: {method!r}")

        if len(f_boxes) == 0:
            continue

        f_labels_int = f_labels.astype(np.int8)
        f_scores_f32 = f_scores.astype(np.float32)
        if class_score_mults:
            mults = np.ones_like(f_scores_f32)
            for c_id, mult_val in class_score_mults.items():
                mults[f_labels_int == int(c_id)] = float(mult_val)
            f_scores_f32 = np.clip(f_scores_f32 * mults, 1e-6, 1.0)

        raw_xywh = np.stack(
            [
                f_boxes[:, 0] * orig_w,
                f_boxes[:, 1] * orig_h,
                (f_boxes[:, 2] - f_boxes[:, 0]) * orig_w,
                (f_boxes[:, 3] - f_boxes[:, 1]) * orig_h,
            ],
            axis=1,
        )
        clipped_xywh, valid = clip_and_filter_boxes_xywh(raw_xywh, orig_w, orig_h)
        if not np.any(valid):
            continue

        out_frames.append(
            pd.DataFrame(
                {
                    "image_key": key,
                    "cls": f_labels_int[valid],
                    "score": f_scores_f32[valid],
                    "x": clipped_xywh[:, 0],
                    "y": clipped_xywh[:, 1],
                    "w": clipped_xywh[:, 2],
                    "h": clipped_xywh[:, 3],
                    "source": tag,
                }
            )
        )

    if not out_frames:
        return empty_predictions_df()
    combined = pd.concat(out_frames, ignore_index=True)
    return normalize_predictions_df(combined, conf_thresh=float(skip_box_thr), max_det_per_image=int(max_det))

