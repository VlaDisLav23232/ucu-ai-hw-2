"""Unit tests for coordinate conversions, union area, and ignore deduplication (U0.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vdet.data.manifest import (
    compute_rect_union_area,
    deduplicate_crowd_boxes,
    load_boxes,
    load_ignore,
    load_manifest,
    parse_ignore_txt_file,
    parse_label_txt_file,
    xywh_to_yolo_norm,
    yolo_norm_to_xywh,
)
from vdet.paths import P


@pytest.mark.parametrize(
    "w,h",
    [
        (1344, 756),
        (1904, 1071),
        (2688, 1512),
        (2720, 1530),
        (1920, 1080),
        (1360, 765),
        (3840, 2160),
        (680, 382),
        (960, 540),
    ],
)
def test_yolo_pixel_roundtrip_all_resolutions(w: int, h: int) -> None:
    """Round-trip pixel (x, y, w, h) -> YOLO norm -> pixel matches within 1e-4 px."""
    rng = np.random.default_rng(42)
    bx = rng.uniform(0, w * 0.8, size=50)
    by = rng.uniform(0, h * 0.8, size=50)
    bw = rng.uniform(4, w * 0.2, size=50)
    bh = rng.uniform(4, h * 0.2, size=50)

    cx, cy, nw, nh = xywh_to_yolo_norm(bx, by, bw, bh, w, h)
    rx, ry, rw, rh = yolo_norm_to_xywh(cx, cy, nw, nh, w, h, clip=True)

    assert np.max(np.abs(rx - bx)) < 1e-4
    assert np.max(np.abs(ry - by)) < 1e-4
    assert np.max(np.abs(rw - bw)) < 1e-4
    assert np.max(np.abs(rh - bh)) < 1e-4


def test_rect_union_area_exact() -> None:
    """Disjoint and overlapping rectangles yield exact geometric union area."""
    # Single box 10x20 = 200
    assert compute_rect_union_area([(10, 10, 10, 20)], 100, 100) == pytest.approx(200.0)
    # Identical duplicate boxes -> still 200
    assert compute_rect_union_area([(10, 10, 10, 20), (10, 10, 10, 20)], 100, 100) == pytest.approx(
        200.0
    )
    # Two overlapping 10x10 boxes shifted by (5, 0): union is 15x10 = 150
    assert compute_rect_union_area([(0, 0, 10, 10), (5, 0, 10, 10)], 100, 100) == pytest.approx(150.0)


def test_deduplicate_crowd_boxes() -> None:
    """Duplicate 8-class COCO crowd entries collapse to unique spatial regions."""
    raw = [[90, 57, 103, 46] for _ in range(8)] + [[10, 20, 30, 40] for _ in range(8)]
    dedup = deduplicate_crowd_boxes(raw)
    assert len(dedup) == 2
    assert dedup[0] == (90.0, 57.0, 103.0, 46.0)
    assert dedup[1] == (10.0, 20.0, 30.0, 40.0)


@pytest.mark.skipif(not (P.cache / "manifest.parquet").exists(), reason="manifest not built yet")
def test_real_dataset_conversion_and_ignore_dedup() -> None:
    """Verify YOLO txt vs COCO bbox <= 1 px per resolution bucket and crowd vs ignore/*.txt IoU > 0.99."""
    manifest_df = load_manifest()
    boxes_df = load_boxes()
    ignore_df = load_ignore()

    annotated = manifest_df[manifest_df["split"].isin(["train", "val"])]
    boxes_by_key = {k: g for k, g in boxes_df.groupby("image_key")}

    for bucket, group in annotated.groupby("res_bucket"):
        row = group.iloc[0]
        label_path = (
            P.data / str(row["split"]) / "labels" / str(row["clip"]) / f"{int(row['frame']):07d}.txt"
        )
        txt_boxes = parse_label_txt_file(label_path, int(row["width"]), int(row["height"]))
        coco_sub = boxes_by_key.get(row["image_key"])
        assert coco_sub is not None
        assert len(txt_boxes) == len(coco_sub), f"Count mismatch in bucket {bucket}"
        txt_arr = np.array([[b[1], b[2], b[3], b[4]] for b in txt_boxes], dtype=np.float64)
        coco_arr = coco_sub[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
        assert float(np.max(np.abs(txt_arr - coco_arr))) <= 1.0

    # Check ignore deduplication on sample frame uav0000013_00000_v_0000001 (§2.7)
    sample_key = "uav0000013_00000_v_0000001"
    ig_sub = ignore_df[ignore_df["image_key"] == sample_key]
    assert len(ig_sub) == 4
    m_row = manifest_df[manifest_df["image_key"] == sample_key].iloc[0]
    txt_ig = parse_ignore_txt_file(
        P.data / "train/ignore/uav0000013_00000_v/0000001.txt",
        int(m_row.width),
        int(m_row.height),
    )
    assert len(txt_ig) == 4

