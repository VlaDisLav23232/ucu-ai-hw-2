"""Tracklet score smoothing (`s' = alpha*s + (1-alpha)*mean_topk`) and linear gap propagation.

Credits: temporal rescoring of linked detections follows the idea of Seq-NMS (Han et al., arXiv:1602.08465, 2016).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vdet.constants import MAX_SUBMISSION_DETS
from vdet.data.manifest import load_manifest
from vdet.infer.predictors import (
    clip_and_filter_boxes_xywh,
    empty_predictions_df,
    normalize_predictions_df,
)
from vdet.temporal.linking import Tracklet, link_clip_tracklets, pairwise_iou_xywh


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
    use_velocity: bool = True,
    small_box_bonus: float = 0.06,
    smooth_boxes: bool = False,
    box_smooth_weight: float = 0.16,
    causal: bool = False,
    conf_thresh: float = 0.001,
    max_det: int = MAX_SUBMISSION_DETS,
    source_tag: str = "temporal_rescore",
) -> pd.DataFrame:
    """Apply per-clip tracklet rescoring, optional 3-frame box smoothing, and linear gap interpolation.

    With `causal=True` every frame only uses information from itself and *earlier* frames (online setting):
    scores are smoothed with the top-k mean / local mean of the tracklet *so far*, a detection that starts a new
    tracklet (no past support) is decayed instead of a whole length-1 tracklet, box smoothing is disabled, and
    gap interpolation is replaced by forward extrapolation: a strong tracklet that is not detected in the next
    1..`max_gap` frames gets its box projected with its own velocity and a decayed score.
    """
    if causal:
        return _apply_causal_rescoring(
            preds_df, manifest_df, iou_link, alpha, topk_ratio, min_track_len_for_boost, single_frame_decay,
            enable_propagation, prop_min_score, prop_min_len, max_gap, gap_decay, dup_iou_thresh,
            use_velocity, small_box_bonus, conf_thresh, max_det, source_tag,
        )
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
    new_boxes = work_df[["x", "y", "w", "h"]].to_numpy(dtype=np.float64).copy()
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
        tracklets = link_clip_tracklets(
            clip_df=clip_df,
            clip_name=str(clip_name),
            iou_thresh=float(iou_link),
            max_gap=int(max_gap) if enable_propagation else 0,
            use_velocity=bool(use_velocity),
            small_box_bonus=float(small_box_bonus),
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
                for n_i, node in enumerate(trk.nodes):
                    # Combine global tracklet top-k mean with local 3-frame neighborhood mean
                    lo = max(0, n_i - 2)
                    hi = min(t_len, n_i + 3)
                    local_mean = float(np.mean([trk.nodes[j].score for j in range(lo, hi)]))
                    ref_s = 0.65 * trk_ref_score + 0.35 * local_mean
                    blended = float(alpha) * node.score + (1.0 - float(alpha)) * ref_s
                    # Never decrease high-confidence anchor detections in a valid tracklet
                    new_scores[node.row_idx] = min(1.0, max(node.score * 0.985, blended))

                if smooth_boxes and float(box_smooth_weight) > 0.0 and t_len >= 3:
                    w_nb = float(box_smooth_weight)
                    for n_i in range(1, t_len - 1):
                        n_prev = trk.nodes[n_i - 1]
                        n_curr = trk.nodes[n_i]
                        n_next = trk.nodes[n_i + 1]
                        if (
                            n_curr.frame_idx - n_prev.frame_idx == 1
                            and n_next.frame_idx - n_curr.frame_idx == 1
                        ):
                            interp_mid = 0.5 * (n_prev.box_xywh + n_next.box_xywh)
                            iou_mid = float(pairwise_iou_xywh(n_curr.box_xywh[None, :], interp_mid[None, :])[0, 0])
                            if iou_mid >= 0.65:
                                new_boxes[n_curr.row_idx] = (
                                    (1.0 - w_nb) * n_curr.box_xywh + w_nb * interp_mid
                                )

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
    if smooth_boxes:
        work_df["x"] = new_boxes[:, 0].astype(np.float32)
        work_df["y"] = new_boxes[:, 1].astype(np.float32)
        work_df["w"] = new_boxes[:, 2].astype(np.float32)
        work_df["h"] = new_boxes[:, 3].astype(np.float32)
    work_df["source"] = source_tag
    base_cols = ["image_key", "cls", "score", "x", "y", "w", "h", "source"]
    out_df = work_df[base_cols]

    if propagated_rows:
        prop_df = pd.DataFrame(propagated_rows)[base_cols]
        out_df = pd.concat([out_df, prop_df], ignore_index=True)

    return normalize_predictions_df(out_df, conf_thresh=float(conf_thresh), max_det_per_image=int(max_det))



def _apply_causal_rescoring(
    preds_df: pd.DataFrame,
    manifest_df: pd.DataFrame | None,
    iou_link: float,
    alpha: float,
    topk_ratio: float,
    min_track_len_for_boost: int,
    single_frame_decay: float,
    enable_propagation: bool,
    prop_min_score: float,
    prop_min_len: int,
    max_gap: int,
    gap_decay: float,
    dup_iou_thresh: float,
    use_velocity: bool,
    small_box_bonus: float,
    conf_thresh: float,
    max_det: int,
    source_tag: str,
) -> pd.DataFrame:
    """Online (past-only) version of `apply_temporal_rescoring`; see its docstring."""
    if preds_df.empty:
        return empty_predictions_df()
    m_df = (manifest_df if manifest_df is not None else load_manifest()).copy()
    if "frame_idx" not in m_df.columns and "frame" in m_df.columns:
        m_df["frame_idx"] = m_df["frame"]
    meta_cols = m_df[["image_key", "clip", "frame_idx", "width", "height"]]
    work_df = preds_df.reset_index(drop=True).merge(meta_cols, on="image_key", how="inner")
    if work_df.empty:
        return empty_predictions_df()

    new_scores = work_df["score"].to_numpy(dtype=np.float64).copy()
    frame_lookup = {(str(r.clip), int(r.frame_idx)): (str(r.image_key), float(r.width), float(r.height))
                    for r in meta_cols.itertuples(index=False)}
    existing_boxes_map: dict[tuple[str, int], np.ndarray] = {}
    if enable_propagation:
        for (ikey, cls_id), grp in work_df.groupby(["image_key", "cls"], sort=False):
            existing_boxes_map[(str(ikey), int(cls_id))] = grp[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
    propagated_rows: list[dict[str, object]] = []
    next_tid = 0

    for clip_name, clip_df in work_df.groupby("clip", sort=False):
        # Linking is already online: each frame is matched only against tracklets built from earlier frames.
        tracklets = link_clip_tracklets(
            clip_df=clip_df, clip_name=str(clip_name), iou_thresh=float(iou_link),
            max_gap=int(max_gap) if enable_propagation else 0, use_velocity=bool(use_velocity),
            small_box_bonus=float(small_box_bonus), start_tracklet_id=next_tid,
        )
        next_tid += len(tracklets)

        for trk in tracklets:
            nodes = trk.nodes
            for n_i, node in enumerate(nodes):
                past = nodes[: n_i + 1]                      # current + earlier frames only
                if n_i == 0:
                    if node.score < 0.25 and float(single_frame_decay) < 1.0:
                        new_scores[node.row_idx] = node.score * float(single_frame_decay)
                    continue
                if len(past) >= int(min_track_len_for_boost):
                    ref = Tracklet(trk.tracklet_id, trk.clip, trk.cls, list(past)).topk_mean_score(float(topk_ratio))
                    local_mean = float(np.mean([p.score for p in past[-3:]]))
                    ref_s = 0.65 * ref + 0.35 * local_mean
                    blended = float(alpha) * node.score + (1.0 - float(alpha)) * ref_s
                    new_scores[node.row_idx] = min(1.0, max(node.score * 0.985, blended))

            if not enable_propagation:
                continue
            for n_i, node in enumerate(nodes):
                past = nodes[: n_i + 1]
                if len(past) < int(prop_min_len) or float(np.mean([p.score for p in past])) < float(prop_min_score):
                    continue
                next_f = nodes[n_i + 1].frame_idx if n_i + 1 < len(nodes) else None
                hist = Tracklet(trk.tracklet_id, trk.clip, trk.cls, list(past))
                for step in range(1, int(max_gap) + 1):
                    f_new = node.frame_idx + step
                    if next_f is not None and f_new >= next_f:
                        break
                    f_info = frame_lookup.get((str(clip_name), f_new))
                    if f_info is None:
                        break
                    key, orig_w, orig_h = f_info
                    box = hist.project_box_xywh(f_new) if use_velocity else node.box_xywh.copy()
                    clipped, valid = clip_and_filter_boxes_xywh(box[None, :], orig_w, orig_h)
                    if not np.any(valid):
                        break
                    c_box = clipped[0]
                    existing = existing_boxes_map.get((key, trk.cls))
                    if existing is not None and len(existing) > 0:
                        if float(pairwise_iou_xywh(c_box[None, :], existing).max()) >= float(dup_iou_thresh):
                            continue
                    score = node.score * float(gap_decay) ** step
                    if score >= float(conf_thresh):
                        propagated_rows.append({"image_key": key, "cls": np.int8(trk.cls), "score": np.float32(min(1.0, score)),
                                                "x": float(c_box[0]), "y": float(c_box[1]), "w": float(c_box[2]),
                                                "h": float(c_box[3]), "source": f"{source_tag}_prop"})

    work_df["score"] = np.clip(new_scores, 0.0, 1.0).astype(np.float32)
    work_df["source"] = source_tag
    base_cols = ["image_key", "cls", "score", "x", "y", "w", "h", "source"]
    out_df = work_df[base_cols]
    if propagated_rows:
        out_df = pd.concat([out_df, pd.DataFrame(propagated_rows)[base_cols]], ignore_index=True)
    return normalize_predictions_df(out_df, conf_thresh=float(conf_thresh), max_det_per_image=int(max_det))
