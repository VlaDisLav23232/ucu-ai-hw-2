"""Tracklet score smoothing (`s' = alpha*s + (1-alpha)*mean_topk`) and linear gap propagation (§10 U4.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vdet.constants import MAX_SUBMISSION_DETS
from vdet.data.manifest import load_manifest
from vdet.infer.predictors import (
    clip_and_filter_boxes_xywh,
    empty_predictions_df,
    normalize_predictions_df,
)
from vdet.temporal.linking import link_clip_tracklets, pairwise_iou_xywh
from vdet.temporal.motion import compute_clip_homographies


def apply_temporal_rescoring(
    preds_df: pd.DataFrame,
    manifest_df: pd.DataFrame | None = None,
    iou_link: float = 0.45,
    alpha: float = 0.70,
    topk_ratio: float = 0.50,
    min_track_len_for_boost: int = 3,
    single_frame_decay: float = 0.96,
    enable_propagation: bool = True,
    prop_min_score: float = 0.25,
    prop_min_len: int = 3,
    max_gap: int = 2,
    gap_decay: float = 0.85,
    dup_iou_thresh: float = 0.50,
    use_motion_comp: bool = False,
    data_root: Path | None = None,
    conf_thresh: float = 0.001,
    max_det: int = MAX_SUBMISSION_DETS,
    source_tag: str = "temporal_rescore",
) -> pd.DataFrame:
    """Apply per-clip tracklet rescoring and optional linear gap interpolation (§10 U4.4)."""
    if preds_df.empty:
        return empty_predictions_df()

    m_df = (manifest_df if manifest_df is not None else load_manifest()).copy()
    if "frame_idx" not in m_df.columns and "frame" in m_df.columns:
        m_df["frame_idx"] = m_df["frame"]
    meta_cols = m_df[["image_key", "clip", "frame_idx", "width", "height", "path"]].copy()

    work_df = preds_df.reset_index(drop=True).copy()
    work_df = work_df.merge(meta_cols[["image_key", "clip", "frame_idx", "width", "height"]], on="image_key", how="inner")
    if work_df.empty:
        return empty_predictions_df()

    new_scores = work_df["score"].to_numpy(dtype=np.float64).copy()
    propagated_rows: list[dict[str, object]] = []
    next_tid = 0

    # Map (clip, frame_idx) -> (image_key, width, height) for gap interpolation
    frame_lookup: dict[tuple[str, int], tuple[str, float, float]] = {
        (str(r.clip), int(r.frame_idx)): (str(r.image_key), float(r.width), float(r.height))
        for r in meta_cols.itertuples(index=False)
    }

    # Index existing boxes per (image_key, cls) to avoid interpolating duplicates
    existing_boxes_map: dict[tuple[str, int], np.ndarray] = {}
    if enable_propagation:
        for (ikey, cls_id), grp in work_df.groupby(["image_key", "cls"], sort=False):
            existing_boxes_map[(str(ikey), int(cls_id))] = grp[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)

    for clip_name, clip_df in work_df.groupby("clip", sort=False):
        homographies = None
        if use_motion_comp:
            clip_m = meta_cols[meta_cols["clip"] == clip_name]
            homographies = compute_clip_homographies(clip_m, data_root=data_root)

        tracklets = link_clip_tracklets(
            clip_df=clip_df,
            clip_name=str(clip_name),
            iou_thresh=float(iou_link),
            max_gap=int(max_gap) if enable_propagation else 0,
            homographies=homographies,
            start_tracklet_id=next_tid,
        )
        next_tid += len(tracklets)

        for trk in tracklets:
            t_len = trk.length
            if t_len == 1:
                node = trk.nodes[0]
                if node.score < 0.25 and float(single_frame_decay) < 1.0:
                    new_scores[node.row_idx] = node.score * float(single_frame_decay)
                continue

            if t_len >= int(min_track_len_for_boost):
                trk_ref_score = trk.topk_mean_score(topk_ratio=float(topk_ratio))
                for node in trk.nodes:
                    blended = float(alpha) * node.score + (1.0 - float(alpha)) * trk_ref_score
                    # Never decrease high-confidence anchor detections in a valid tracklet
                    new_scores[node.row_idx] = min(1.0, max(node.score * 0.98, blended))

            # Linear gap interpolation for strong tracklets
            if enable_propagation and t_len >= int(prop_min_len) and trk.mean_score() >= float(prop_min_score):
                for n_idx in range(len(trk.nodes) - 1):
                    n_prev = trk.nodes[n_idx]
                    n_next = trk.nodes[n_idx + 1]
                    gap = n_next.frame_idx - n_prev.frame_idx - 1
                    if 1 <= gap <= int(max_gap):
                        for step in range(1, gap + 1):
                            f_mid = n_prev.frame_idx + step
                            f_info = frame_lookup.get((str(clip_name), f_mid))
                            if f_info is None:
                                continue
                            mid_key, orig_w, orig_h = f_info
                            lam = float(step) / float(gap + 1)
                            interp_box = (1.0 - lam) * n_prev.box_xywh + lam * n_next.box_xywh
                            clipped, valid = clip_and_filter_boxes_xywh(interp_box[None, :], orig_w, orig_h)
                            if not np.any(valid):
                                continue
                            c_box = clipped[0]

                            # Skip if an existing detection of the same class already overlaps >= dup_iou_thresh
                            existing = existing_boxes_map.get((mid_key, trk.cls))
                            if existing is not None and len(existing) > 0:
                                ious = pairwise_iou_xywh(c_box[None, :], existing)
                                if float(ious.max()) >= float(dup_iou_thresh):
                                    continue

                            interp_score = ((1.0 - lam) * n_prev.score + lam * n_next.score) * (
                                float(gap_decay) ** step
                            )
                            if interp_score >= float(conf_thresh):
                                propagated_rows.append(
                                    {
                                        "image_key": mid_key,
                                        "cls": np.int8(trk.cls),
                                        "score": np.float32(min(1.0, interp_score)),
                                        "x": float(c_box[0]),
                                        "y": float(c_box[1]),
                                        "w": float(c_box[2]),
                                        "h": float(c_box[3]),
                                        "source": f"{source_tag}_prop",
                                    }
                                )

    work_df["score"] = np.clip(new_scores, 0.0, 1.0).astype(np.float32)
    work_df["source"] = source_tag
    base_cols = ["image_key", "cls", "score", "x", "y", "w", "h", "source"]
    out_df = work_df[base_cols]

    if propagated_rows:
        prop_df = pd.DataFrame(propagated_rows)[base_cols]
        out_df = pd.concat([out_df, prop_df], ignore_index=True)

    return normalize_predictions_df(out_df, conf_thresh=float(conf_thresh), max_det_per_image=int(max_det))

