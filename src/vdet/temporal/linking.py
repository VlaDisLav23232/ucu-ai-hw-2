"""Per-clip, per-class IoU tracklet linking with optional camera-motion compensation (§10 U4.4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vdet.temporal.motion import warp_boxes_xywh


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


def link_clip_tracklets(
    clip_df: pd.DataFrame,
    clip_name: str,
    iou_thresh: float = 0.45,
    max_gap: int = 2,
    homographies: dict[tuple[int, int], np.ndarray] | None = None,
    start_tracklet_id: int = 0,
) -> list[Tracklet]:
    """Link detections in `clip_df` (must include `frame_idx`) per class into `Tracklet` objects (§10 U4.4)."""
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
                # Project each active tracklet's last box forward to f_idx if homographies are available
                prev_boxes = np.stack([t.last_node.box_xywh for t in active], axis=0)
                if homographies:
                    for t_i, trk in enumerate(active):
                        last_f = trk.last_node.frame_idx
                        H_Step = homographies.get((last_f, f_idx))
                        if H_Step is not None:
                            prev_boxes[t_i : t_i + 1] = warp_boxes_xywh(prev_boxes[t_i : t_i + 1], H_Step)

                iou_mat = pairwise_iou_xywh(prev_boxes, boxes_curr)
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

