"""Unit tests for U4.1–U4.3: WBF / Soft-NMS / NMS fusion and SAHI tile window generation."""

from __future__ import annotations

import pandas as pd

from vdet.infer.fusion import fuse_predictions
from vdet.infer.tiling import compute_slice_windows


def test_fuse_predictions_wbf_softnms_nms() -> None:
    """WBF, Soft-NMS, and NMS fuse overlapping predictions across models while preserving valid schema."""
    manifest_df = pd.DataFrame(
        {
            "image_key": ["clip1_0000001"],
            "width": [1920],
            "height": [1080],
        }
    )
    preds_m1 = pd.DataFrame(
        {
            "image_key": ["clip1_0000001", "clip1_0000001"],
            "cls": [2, 0],
            "score": [0.80, 0.60],
            "x": [100.0, 400.0],
            "y": [200.0, 300.0],
            "w": [50.0, 20.0],
            "h": [40.0, 50.0],
            "source": ["m1", "m1"],
        }
    )
    preds_m2 = pd.DataFrame(
        {
            "image_key": ["clip1_0000001"],
            "cls": [2],
            "score": [0.90],
            "x": [102.0],
            "y": [198.0],
            "w": [48.0],
            "h": [42.0],
            "source": ["m2"],
        }
    )

    for method in ("wbf", "soft_nms", "nms"):
        fused = fuse_predictions(
            [preds_m1, preds_m2],
            weights=[1.0, 1.5],
            method=method,
            iou_thr=0.55,
            manifest_df=manifest_df,
        )
        assert not fused.empty
        assert set(fused.columns) == {"image_key", "cls", "score", "x", "y", "w", "h", "source"}
        # The two overlapping class-2 boxes should merge into 1 fused box, leaving 2 total boxes (cls 2 and cls 0)
        if method in {"wbf", "nms"}:
            assert len(fused) == 2


def test_compute_slice_windows_coverage() -> None:
    """Slice windows cover the entire image dimensions `[0, W] x [0, H]` with overlap."""
    windows = compute_slice_windows(1920, 1080, slice_w=800, slice_h=800, overlap_ratio=0.25)
    assert len(windows) == 6  # 3 horizontal x 2 vertical
    assert min(w[0] for w in windows) == 0
    assert min(w[1] for w in windows) == 0
    assert max(w[2] for w in windows) == 1920
    assert max(w[3] for w in windows) == 1080

