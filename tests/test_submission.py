"""Unit tests for submission formatting, round-trip parsing, and strict validation (U0.6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vdet.infer.predictors import empty_predictions_df
from vdet.submission import from_csv, to_csv, validate


@pytest.fixture()
def mini_test_images_csv(tmp_path: Path) -> Path:
    """Create a synthetic test_images.csv with 4 frames across 2 resolutions."""
    df = pd.DataFrame(
        [
            {
                "image_id": "clip_001_0000001",
                "file_name": "test/images/clip_001/0000001.jpg",
                "clip": "clip_001",
                "frame": 1,
                "width": 1920,
                "height": 1080,
            },
            {
                "image_id": "clip_001_0000002",
                "file_name": "test/images/clip_001/0000002.jpg",
                "clip": "clip_001",
                "frame": 2,
                "width": 1920,
                "height": 1080,
            },
            {
                "image_id": "clip_002_0000001",
                "file_name": "test/images/clip_002/0000001.jpg",
                "clip": "clip_002",
                "frame": 1,
                "width": 680,
                "height": 382,
            },
            {
                "image_id": "clip_002_0000002",
                "file_name": "test/images/clip_002/0000002.jpg",
                "clip": "clip_002",
                "frame": 2,
                "width": 680,
                "height": 382,
            },
        ]
    )
    csv_path = tmp_path / "test_images.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_empty_submission_passes_validate(tmp_path: Path, mini_test_images_csv: Path) -> None:
    """An empty prediction table produces a valid empty submission."""
    out_csv = tmp_path / "empty_sub.csv"
    to_csv(empty_predictions_df(), out_csv, test_images_csv=mini_test_images_csv)
    stats = validate(out_csv, test_images_csv=mini_test_images_csv)
    assert stats["valid"] is True
    assert stats["rows"] == 4
    assert stats["total_detections"] == 0


def test_synthetic_submission_and_lossless_roundtrip(
    tmp_path: Path, mini_test_images_csv: Path
) -> None:
    """Synthetic random predictions pass validate and round-trip CSV -> table -> CSV losslessly."""
    rng = np.random.default_rng(42)
    rows = []
    for img_id, w, h in [
        ("clip_001_0000001", 1920, 1080),
        ("clip_002_0000001", 680, 382),
    ]:
        for _ in range(25):
            rows.append(
                {
                    "image_key": img_id,
                    "cls": int(rng.integers(0, 8)),
                    "score": round(float(rng.uniform(0.05, 0.99)), 4),
                    "x": round(float(rng.uniform(5.0, w * 0.7)), 2),
                    "y": round(float(rng.uniform(5.0, h * 0.7)), 2),
                    "w": round(float(rng.uniform(10.0, w * 0.2)), 2),
                    "h": round(float(rng.uniform(10.0, h * 0.2)), 2),
                }
            )
    preds_df = pd.DataFrame(rows)

    csv1 = tmp_path / "sub1.csv"
    csv2 = tmp_path / "sub2.csv"
    to_csv(preds_df, csv1, test_images_csv=mini_test_images_csv, decimals=2)
    stats = validate(csv1, test_images_csv=mini_test_images_csv)
    assert stats["valid"] is True
    assert stats["total_detections"] == 50

    parsed_df = from_csv(csv1)
    to_csv(parsed_df, csv2, test_images_csv=mini_test_images_csv, decimals=2)
    assert csv1.read_text(encoding="utf-8") == csv2.read_text(encoding="utf-8")

    # Verify numeric 2-decimal tolerance on parsed DataFrame
    orig_sorted = preds_df.sort_values(["image_key", "score"], ascending=[True, False]).reset_index(
        drop=True
    )
    parsed_sorted = parsed_df.sort_values(
        ["image_key", "score"], ascending=[True, False]
    ).reset_index(drop=True)
    for col in ("x", "y", "w", "h"):
        assert np.max(np.abs(orig_sorted[col] - parsed_sorted[col])) <= 0.01


def test_validator_rejects_broken_submissions(
    tmp_path: Path, mini_test_images_csv: Path
) -> None:
    """Deliberately broken submissions (bad order, >500 dets, cls=8, NaN, w<=0, out of bounds) fail."""
    base_csv = tmp_path / "base.csv"
    to_csv(empty_predictions_df(), base_csv, test_images_csv=mini_test_images_csv)
    base_df = pd.read_csv(base_csv, dtype=str, keep_default_na=False)

    # 1. Bad row order
    bad_order = base_df.iloc[[1, 0, 2, 3]].copy()
    p_order = tmp_path / "bad_order.csv"
    bad_order.to_csv(p_order, index=False)
    with pytest.raises(ValueError, match="ordering"):
        validate(p_order, test_images_csv=mini_test_images_csv)

    # 2. > 500 detections in a single row
    too_many = base_df.copy()
    too_many.loc[0, "PredictionString"] = " ".join(["0 0.5 10 10 20 20"] * 501)
    p_many = tmp_path / "too_many.csv"
    too_many.to_csv(p_many, index=False)
    with pytest.raises(ValueError, match="exceeds max_dets"):
        validate(p_many, test_images_csv=mini_test_images_csv)

    # 3. Invalid class 8
    bad_cls = base_df.copy()
    bad_cls.loc[0, "PredictionString"] = "8 0.9 10 10 20 20"
    p_cls = tmp_path / "bad_cls.csv"
    bad_cls.to_csv(p_cls, index=False)
    with pytest.raises(ValueError, match="out of valid range"):
        validate(p_cls, test_images_csv=mini_test_images_csv)

    # 4. NaN token
    bad_nan = base_df.copy()
    bad_nan.loc[0, "PredictionString"] = "0 0.9 nan 10 20 20"
    p_nan = tmp_path / "bad_nan.csv"
    bad_nan.to_csv(p_nan, index=False)
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate(p_nan, test_images_csv=mini_test_images_csv)

    # 5. Non-positive width
    bad_w = base_df.copy()
    bad_w.loc[0, "PredictionString"] = "0 0.9 10 10 0 20"
    p_w = tmp_path / "bad_w.csv"
    bad_w.to_csv(p_w, index=False)
    with pytest.raises(ValueError, match="non-positive box size"):
        validate(p_w, test_images_csv=mini_test_images_csv)

    # 6. Out of image bounds
    bad_bounds = base_df.copy()
    bad_bounds.loc[2, "PredictionString"] = "0 0.9 600 100 150 50"  # width is 680, 600+150=750 > 681
    p_bounds = tmp_path / "bad_bounds.csv"
    bad_bounds.to_csv(p_bounds, index=False)
    with pytest.raises(ValueError, match="out of image bounds"):
        validate(p_bounds, test_images_csv=mini_test_images_csv)

