"""Seq-NMS dynamic-programming sequence rescoring and suppression across video clips (§10 U4.5).

Credits: Han et al., "Seq-NMS for Video Object Detection", arXiv:1602.08465, 2016.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vdet.constants import MAX_SUBMISSION_DETS
from vdet.data.manifest import load_manifest
from vdet.infer.predictors import empty_predictions_df, normalize_predictions_df
from vdet.temporal.linking import pairwise_iou_xywh


def apply_seq_nms(
    preds_df: pd.DataFrame,
    manifest_df: pd.DataFrame | None = None,
    iou_link: float = 0.45,
    nms_iou: float = 0.55,
    score_blend: float = 0.30,
    top_k_per_frame: int = 200,
    conf_thresh: float = 0.001,
    max_det: int = MAX_SUBMISSION_DETS,
    source_tag: str = "seqnms",
) -> pd.DataFrame:
    """Apply dynamic-programming Seq-NMS per clip and class (§10 U4.5)."""
    if preds_df.empty:
        return empty_predictions_df()

    m_df = (manifest_df if manifest_df is not None else load_manifest()).copy()
    if "frame_idx" not in m_df.columns and "frame" in m_df.columns:
        m_df["frame_idx"] = m_df["frame"]
    meta_cols = m_df[["image_key", "clip", "frame_idx"]]

    work_df = preds_df.reset_index(drop=True).copy()
    work_df = work_df.merge(meta_cols, on="image_key", how="inner")
    if work_df.empty:
        return empty_predictions_df()

    new_scores = work_df["score"].to_numpy(dtype=np.float64).copy()
    suppressed = np.zeros(len(work_df), dtype=bool)

    for _, clip_cls_df in work_df.groupby(["clip", "cls"], sort=False):
        # Restrict graph construction to top-K detections per frame for speed on dense scenes
        frames_sorted = sorted(clip_cls_df["frame_idx"].unique())
        if len(frames_sorted) < 2:
            continue

        frame_boxes: dict[int, np.ndarray] = {}
        frame_scores: dict[int, np.ndarray] = {}
        frame_rows: dict[int, np.ndarray] = {}
        for f_idx, f_grp in clip_cls_df.groupby("frame_idx", sort=False):
            f_sub = f_grp.nlargest(int(top_k_per_frame), "score")
            frame_boxes[int(f_idx)] = f_sub[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
            frame_scores[int(f_idx)] = f_sub["score"].to_numpy(dtype=np.float64)
            frame_rows[int(f_idx)] = f_sub.index.to_numpy(dtype=np.int64)

        # Precompute consecutive frame-pair adjacency lists (IoU >= iou_link)
        links: dict[tuple[int, int], list[int]] = {}
        for i in range(len(frames_sorted) - 1):
            f1, f2 = int(frames_sorted[i]), int(frames_sorted[i + 1])
            if f2 - f1 > 2:
                continue
            ious = pairwise_iou_xywh(frame_boxes[f1], frame_boxes[f2])
            src_idx, dst_idx = np.where(ious >= float(iou_link))
            for s_i, d_i in zip(src_idx, dst_idx):
                links.setdefault((f1, int(s_i)), []).append(int(d_i))

        if not links:
            continue

        # DP forward pass: find highest-scoring trajectories and blend sequence mean score
        used_local: set[tuple[int, int]] = set()
        dp_val: dict[tuple[int, int], float] = {}
        dp_next: dict[tuple[int, int], tuple[int, int] | None] = {}

        for f_idx in reversed(frames_sorted):
            f_i = int(f_idx)
            sc_arr = frame_scores[f_i]
            for b_i in range(len(sc_arr)):
                node = (f_i, b_i)
                best_child_sum = 0.0
                best_child: tuple[int, int] | None = None
                # Determine next frame index
                f_pos = frames_sorted.index(f_idx)
                if f_pos + 1 < len(frames_sorted):
                    f_next = int(frames_sorted[f_pos + 1])
                    for c_i in links.get(node, []):
                        child = (f_next, c_i)
                        val = dp_val.get(child, 0.0)
                        if val > best_child_sum:
                            best_child_sum = val
                            best_child = child
                dp_val[node] = float(sc_arr[b_i]) + best_child_sum
                dp_next[node] = best_child

        # Greedily extract paths in descending DP sum order, rescore, and suppress overlapping duplicates
        all_nodes = sorted(dp_val.keys(), key=lambda k: dp_val[k], reverse=True)
        for start_node in all_nodes:
            if start_node in used_local:
                continue
            path: list[tuple[int, int]] = []
            curr: tuple[int, int] | None = start_node
            while curr is not None and curr not in used_local:
                path.append(curr)
                curr = dp_next.get(curr)

            if len(path) < 2:
                continue

            seq_scores = [frame_scores[f][b] for f, b in path]
            seq_mean = float(np.mean(seq_scores))

            for f_i, b_i in path:
                used_local.add((f_i, b_i))
                g_row = int(frame_rows[f_i][b_i])
                orig_s = float(frame_scores[f_i][b_i])
                new_scores[g_row] = min(1.0, (1.0 - float(score_blend)) * orig_s + float(score_blend) * seq_mean)

                # Suppress lower-scoring boxes in the same frame that overlap >= nms_iou with the selected path box
                ious_same = pairwise_iou_xywh(frame_boxes[f_i][b_i : b_i + 1], frame_boxes[f_i])[0]
                for other_i in np.where(ious_same >= float(nms_iou))[0]:
                    other_i_int = int(other_i)
                    if other_i_int != b_i and (f_i, other_i_int) not in used_local:
                        if frame_scores[f_i][other_i_int] < orig_s:
                            used_local.add((f_i, other_i_int))
                            suppressed[int(frame_rows[f_i][other_i_int])] = True

    work_df["score"] = np.clip(new_scores, 0.0, 1.0).astype(np.float32)
    work_df["source"] = source_tag
    work_df = work_df[~suppressed]
    base_cols = ["image_key", "cls", "score", "x", "y", "w", "h", "source"]
    return normalize_predictions_df(
        work_df[base_cols], conf_thresh=float(conf_thresh), max_det_per_image=int(max_det)
    )

