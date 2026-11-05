"""Unit tests for offline cache builder and GT-protected Gaussian blur (U0.4)."""

from __future__ import annotations

import numpy as np

from vdet.data.cache import apply_gt_protected_blur


def test_gt_protected_blur_never_modifies_labelled_gt_pixels() -> None:
    """Pixels inside any labelled GT box (plus 2 px dilation) remain 100% identical after blur fill."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    orig = img.copy()

    # Large ignore region covering [40, 40, 200, 160]
    ignore_xywh = np.array([[40.0, 40.0, 200.0, 160.0]], dtype=np.float32)
    # Two GT boxes: one completely inside the ignore region, one overlapping its boundary
    gt_xywh = np.array(
        [
            [80.0, 70.0, 30.0, 40.0],
            [220.0, 100.0, 40.0, 50.0],
        ],
        dtype=np.float32,
    )

    out = apply_gt_protected_blur(
        img,
        ignore_xywh=ignore_xywh,
        gt_xywh=gt_xywh,
        gt_dilate_px=2,
        feather_px=6,
    )

    # 1. Verify GT boxes (+2px dilation) are completely untouched
    for gx, gy, gw, gh in gt_xywh:
        x1 = int(gx) - 2
        y1 = int(gy) - 2
        x2 = int(gx + gw) + 2
        y2 = int(gy + gh) + 2
        assert np.array_equal(out[y1:y2, x1:x2], orig[y1:y2, x1:x2])

    # 2. Verify ignore region interior away from GT boxes IS blurred (changed)
    assert not np.array_equal(out[130:180, 130:190], orig[130:180, 130:190])

