"""Unit tests for U4.4–U4.6: temporal tracklet linking, rescoring, gap propagation, Seq-NMS, and motion warping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vdet.temporal.motion import warp_boxes_xywh
from vdet.temporal.rescoring import apply_temporal_rescoring
from vdet.temporal.seqnms import apply_seq_nms


def test_temporal_rescoring_and_gap_propagation() -> None:
    """Tracklet rescoring boosts consistent multi-frame detections and interpolates missing intermediate frames."""
    manifest_df = pd.DataFrame(
        {
            "image_key": ["c1_0000001", "c1_0000002", "c1_0000003", "c1_0000004"],
            "clip": ["c1", "c1", "c1", "c1"],
            "frame_idx": [1, 2, 3, 4],
            "width": [1920, 1920, 1920, 1920],
            "height": [1080, 1080, 1080, 1080],
            "path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        }
    )
    # Object present in frames 1, 2, and 4 (missing in frame 3 -> 1-frame gap!)
    preds_df = pd.DataFrame(
        {
            "image_key": ["c1_0000001", "c1_0000002", "c1_0000004"],
            "cls": [2, 2, 2],
            "score": [0.85, 0.35, 0.85],
            "x": [100.0, 104.0, 112.0],
            "y": [200.0, 200.0, 200.0],
            "w": [40.0, 40.0, 40.0],
            "h": [30.0, 30.0, 30.0],
            "source": ["base", "base", "base"],
        }
    )

    rescored = apply_temporal_rescoring(
        preds_df,
        manifest_df=manifest_df,
        iou_link=0.45,
        alpha=0.60,
        enable_propagation=True,
        prop_min_score=0.25,
        prop_min_len=3,
        max_gap=2,
    )
    # Frame 3 gap should be interpolated -> 4 total detections
    assert len(rescored) == 4
    f2_score = float(rescored[rescored["image_key"] == "c1_0000002"]["score"].iloc[0])
    assert f2_score > 0.35  # Boosted by tracklet top-k mean!

    f3_row = rescored[rescored["image_key"] == "c1_0000003"].iloc[0]
    assert abs(float(f3_row["x"]) - 108.0) < 1.0


def test_seq_nms_and_homography_warp() -> None:
    """Seq-NMS links multi-frame boxes and suppresses lower-scoring duplicates; homography translation shifts boxes."""
    manifest_df = pd.DataFrame(
        {
            "image_key": ["c1_0000001", "c1_0000002"],
            "clip": ["c1", "c1"],
            "frame_idx": [1, 2],
            "width": [1920, 1920],
            "height": [1080, 1080],
        }
    )
    preds_df = pd.DataFrame(
        {
            "image_key": ["c1_0000001", "c1_0000001", "c1_0000002"],
            "cls": [2, 2, 2],
            "score": [0.90, 0.40, 0.85],
            "x": [100.0, 101.0, 102.0],
            "y": [200.0, 201.0, 200.0],
            "w": [50.0, 50.0, 50.0],
            "h": [40.0, 40.0, 40.0],
            "source": ["base", "base", "base"],
        }
    )
    out = apply_seq_nms(preds_df, manifest_df=manifest_df, iou_link=0.45, nms_iou=0.50)
    assert len(out) == 2  # Duplicate in frame 1 suppressed

    # Pure translation homography (+10 px in x, -5 px in y)
    H = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    boxes = np.array([[100.0, 200.0, 50.0, 40.0]], dtype=np.float64)
    warped = warp_boxes_xywh(boxes, H)
    assert np.allclose(warped, np.array([[110.0, 195.0, 50.0, 40.0]]))

