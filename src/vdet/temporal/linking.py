"""Per-clip, per-class IoU tracklet linking with constant-velocity box projection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd



def pairwise_iou_xywh(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute `(N, M)` pairwise IoU matrix between two `(x, y, w, h)` float arrays."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x1_1 = boxes1[:, 0:1]
    y1_1 = boxes1[:, 1:2]
    x2_1 = x1_1 + boxes1[:, 2:3]
    y2_1 = y1_1 + boxes1[:, 3:4]

    x1_2 = boxes2[:, 0][None, :]
    y1_2 = boxes2[:, 1][None, :]
    x2_2 = x1_2 + boxes2[:, 2][None, :]
    y2_2 = y1_2 + boxes2[:, 3][None, :]

    iw = np.maximum(0.0, np.minimum(x2_1, x2_2) - np.maximum(x1_1, x1_2))
    ih = np.maximum(0.0, np.minimum(y2_1, y2_2) - np.maximum(y1_1, y1_2))
    inter = iw * ih
    area1 = boxes1[:, 2:3] * boxes1[:, 3:4]
    area2 = (boxes2[:, 2] * boxes2[:, 3])[None, :]
    union = area1 + area2 - inter + 1e-7
    return (inter / union).astype(np.float32)


@dataclass
class TrackletNode:
    """Single detection belonging to a linked temporal tracklet."""

    row_idx: int
    image_key: str
    frame_idx: int
    cls: int
    score: float
    box_xywh: np.ndarray


@dataclass
class Tracklet:
    """Sequence of temporally linked detections of the same class within a clip."""

    tracklet_id: int
    clip: str
    cls: int
    nodes: list[TrackletNode] = field(default_factory=list)

    @property
    def last_node(self) -> TrackletNode:
        return self.nodes[-1]

    @property
    def length(self) -> int:
        return len(self.nodes)

    def mean_score(self) -> float:
        if not self.nodes:
            return 0.0
        return float(np.mean([n.score for n in self.nodes]))

    def topk_mean_score(self, topk_ratio: float = 0.5, min_k: int = 2) -> float:
        if not self.nodes:
            return 0.0
        scores = np.sort(np.asarray([n.score for n in self.nodes], dtype=np.float64))[::-1]
        k = max(min_k, int(np.ceil(len(scores) * float(topk_ratio))))
        k = min(len(scores), k)
        return float(scores[:k].mean())

    def project_box_xywh(self, target_frame_idx: int) -> np.ndarray:
        """Project the last bounding box to `target_frame_idx` using constant-velocity kinematics."""
        last = self.nodes[-1]
        box = last.box_xywh.copy()
        if len(self.nodes) < 2:
            return box
        prev = self.nodes[-2]
        dt_prev = max(1, last.frame_idx - prev.frame_idx)
        dt_next = max(1, int(target_frame_idx) - last.frame_idx)
        if dt_prev > 3 or dt_next > 3:
            return box

        cx_last = last.box_xywh[0] + 0.5 * last.box_xywh[2]
        cy_last = last.box_xywh[1] + 0.5 * last.box_xywh[3]
        cx_prev = prev.box_xywh[0] + 0.5 * prev.box_xywh[2]
        cy_prev = prev.box_xywh[1] + 0.5 * prev.box_xywh[3]

        vx = (cx_last - cx_prev) / float(dt_prev)
        vy = (cy_last - cy_prev) / float(dt_prev)
        max_shift = 1.25 * max(float(last.box_xywh[2]), float(last.box_xywh[3]), 12.0)
        dx = float(np.clip(vx * dt_next, -max_shift, max_shift))
        dy = float(np.clip(vy * dt_next, -max_shift, max_shift))
        box[0] += dx
        box[1] += dy
        return box


def pairwise_similarity_xywh(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    small_box_bonus: float = 0.08,
) -> np.ndarray:
    """Compute `(N, M)` pairwise IoU + small-object center-proximity bonus for tiny aerial boxes."""
    iou = pairwise_iou_xywh(boxes1, boxes2)
    if len(boxes1) == 0 or len(boxes2) == 0 or float(small_box_bonus) <= 0.0:
        return iou

    cx1 = boxes1[:, 0:1] + 0.5 * boxes1[:, 2:3]
    cy1 = boxes1[:, 1:2] + 0.5 * boxes1[:, 3:4]
    cx2 = (boxes2[:, 0] + 0.5 * boxes2[:, 2])[None, :]
    cy2 = (boxes2[:, 1] + 0.5 * boxes2[:, 3])[None, :]

    dist2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    scale1 = np.maximum(boxes1[:, 2:3], boxes1[:, 3:4])
    scale2 = np.maximum(boxes2[:, 2], boxes2[:, 3])[None, :]
    mean_scale = np.maximum(8.0, 0.5 * (scale1 + scale2))

    # Only apply soft center proximity bonus when both boxes have similar scale and are small (<48 px)
    size_ratio = np.minimum(scale1, scale2) / (np.maximum(scale1, scale2) + 1e-6)
    is_small = (mean_scale < 48.0).astype(np.float32)
    prox = np.exp(-dist2 / (1.5 * (mean_scale**2))) * (size_ratio > 0.6).astype(np.float32) * is_small
    return np.clip(iou + float(small_box_bonus) * prox, 0.0, 1.0).astype(np.float32)


def link_clip_tracklets(
    clip_df: pd.DataFrame,
    clip_name: str,
    iou_thresh: float = 0.45,
    max_gap: int = 2,
    use_velocity: bool = True,
    small_box_bonus: float = 0.06,
    start_tracklet_id: int = 0,
) -> list[Tracklet]:
    """Link detections in `clip_df` (must include `frame_idx`) per class into `Tracklet` objects."""
    if clip_df.empty:
        return []

    tracklets: list[Tracklet] = []
    next_tid = int(start_tracklet_id)

    for cls_id, cls_df in clip_df.groupby("cls", sort=False):
        active: list[Tracklet] = []
        finished: list[Tracklet] = []

        for frame_idx, frame_df in cls_df.groupby("frame_idx", sort=True):
            f_idx = int(frame_idx)
            # Expire tracklets whose last seen frame is older than max_gap + 1
            still_active: list[Tracklet] = []
            for trk in active:
                if f_idx - trk.last_node.frame_idx <= int(max_gap) + 1:
                    still_active.append(trk)
                else:
                    finished.append(trk)
            active = still_active

            boxes_curr = frame_df[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
            scores_curr = frame_df["score"].to_numpy(dtype=np.float64)
            keys_curr = frame_df["image_key"].astype(str).tolist()
            row_indices = frame_df.index.to_numpy(dtype=np.int64)

            n_curr = len(boxes_curr)
            matched_curr = np.zeros(n_curr, dtype=bool)

            if active and n_curr > 0:
                prev_boxes = np.stack([t.last_node.box_xywh for t in active], axis=0)

                iou_mat = pairwise_similarity_xywh(prev_boxes, boxes_curr, small_box_bonus=small_box_bonus)
                if use_velocity:
                    vel_boxes = np.stack([t.project_box_xywh(f_idx) for t in active], axis=0)
                    vel_mat = pairwise_similarity_xywh(vel_boxes, boxes_curr, small_box_bonus=small_box_bonus)
                    iou_mat = np.maximum(iou_mat, vel_mat)
                # Weight IoU slightly by score consistency to break ties cleanly in crowds
                matched_trk = np.zeros(len(active), dtype=bool)
                # Greedy assignment in descending IoU order (above iou_thresh)
                cand_t, cand_c = np.where(iou_mat >= float(iou_thresh))
                if len(cand_t) > 0:
                    order = np.argsort(iou_mat[cand_t, cand_c])[::-1]
                    for idx in order:
                        ti = int(cand_t[idx])
                        ci = int(cand_c[idx])
                        if matched_trk[ti] or matched_curr[ci]:
                            continue
                        matched_trk[ti] = True
                        matched_curr[ci] = True
                        active[ti].nodes.append(
                            TrackletNode(
                                row_idx=int(row_indices[ci]),
                                image_key=keys_curr[ci],
                                frame_idx=f_idx,
                                cls=int(cls_id),
                                score=float(scores_curr[ci]),
                                box_xywh=boxes_curr[ci].copy(),
                            )
                        )

            # Start new tracklets for unmatched detections
            for ci in range(n_curr):
                if not matched_curr[ci]:
                    new_trk = Tracklet(
                        tracklet_id=next_tid,
                        clip=clip_name,
                        cls=int(cls_id),
                        nodes=[
                            TrackletNode(
                                row_idx=int(row_indices[ci]),
                                image_key=keys_curr[ci],
                                frame_idx=f_idx,
                                cls=int(cls_id),
                                score=float(scores_curr[ci]),
                                box_xywh=boxes_curr[ci].copy(),
                            )
                        ],
                    )
                    next_tid += 1
                    active.append(new_trk)

        finished.extend(active)
        tracklets.extend(finished)

    return tracklets

