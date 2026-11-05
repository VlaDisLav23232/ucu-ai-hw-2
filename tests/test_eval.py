"""Mandatory sanity tests for the local COCO evaluator (§3.2, U0.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vdet.constants import NUM_CLASSES
from vdet.eval.coco_eval import evaluate


def _make_synthetic_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a synthetic 2-frame dataset with dense boxes and ignore regions for fast unit tests."""
    manifest_df = pd.DataFrame(
        [
            {
                "image_key": "clip_test_0000001",
                "split": "val",
                "clip": "clip_test",
                "frame": 1,
                "path": "val/images/clip_test/0000001.jpg",
                "width": 1920,
                "height": 1080,
                "n_boxes": 300.0,
                "n_ignore": 2,
                "ignore_area_frac": 0.1,
                "res_bucket": "1920x1080",
                "cls_counts": [300, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "image_key": "clip_test_0000002",
                "split": "val",
                "clip": "clip_test",
                "frame": 2,
                "path": "val/images/clip_test/0000002.jpg",
                "width": 1920,
                "height": 1080,
                "n_boxes": 16.0,
                "n_ignore": 1,
                "ignore_area_frac": 0.05,
                "res_bucket": "1920x1080",
                "cls_counts": [2] * NUM_CLASSES,
            },
        ]
    )

    # Frame 1: 300 non-overlapping small boxes of class 0 on a 25x12 grid in top half of image
    box_rows = []
    idx = 0
    for r in range(12):
        for c in range(25):
            box_rows.append(
                {
                    "image_key": "clip_test_0000001",
                    "split": "val",
                    "clip": "clip_test",
                    "frame": 1,
                    "cls": 0,
                    "x": float(20 + c * 60),
                    "y": float(20 + r * 35),
                    "w": 24.0,
                    "h": 20.0,
                    "track_id": idx + 1,
                    "occlusion": 0,
                    "truncation": 0,
                    "orig_category_id": 1,
                }
            )
            idx += 1

    # Frame 2: 2 boxes for each of the 8 classes (16 boxes total)
    for cls_id in range(NUM_CLASSES):
        for rep in range(2):
            box_rows.append(
                {
                    "image_key": "clip_test_0000002",
                    "split": "val",
                    "clip": "clip_test",
                    "frame": 2,
                    "cls": cls_id,
                    "x": float(50 + cls_id * 150 + rep * 50),
                    "y": 100.0,
                    "w": 36.0,
                    "h": 36.0,
                    "track_id": 1000 + cls_id * 2 + rep,
                    "occlusion": 0,
                    "truncation": 0,
                    "orig_category_id": cls_id + 1,
                }
            )

    boxes_df = pd.DataFrame(box_rows)

    # Ignore regions placed in the bottom half (y >= 600) away from labelled GT boxes
    ignore_df = pd.DataFrame(
        [
            {
                "image_key": "clip_test_0000001",
                "split": "val",
                "clip": "clip_test",
                "frame": 1,
                "x": 100.0,
                "y": 650.0,
                "w": 400.0,
                "h": 300.0,
            },
            {
                "image_key": "clip_test_0000001",
                "split": "val",
                "clip": "clip_test",
                "frame": 1,
                "x": 700.0,
                "y": 650.0,
                "w": 300.0,
                "h": 250.0,
            },
            {
                "image_key": "clip_test_0000002",
                "split": "val",
                "clip": "clip_test",
                "frame": 2,
                "x": 200.0,
                "y": 600.0,
                "w": 500.0,
                "h": 350.0,
            },
        ]
    )
    return manifest_df, boxes_df, ignore_df


def test_sanity_1_gt_as_predictions_gives_ap_one() -> None:
    """Sanity check 1 (§3.2): GT boxes submitted as predictions (score 1.0) give AP = 1.000."""
    manifest_df, boxes_df, ignore_df = _make_synthetic_dataset()
    preds_df = boxes_df[["image_key", "cls", "x", "y", "w", "h"]].copy()
    preds_df["score"] = np.float32(1.0)

    res = evaluate(
        preds_df,
        split="val",
        max_dets=[1, 10, 500],
        manifest_df=manifest_df,
        boxes_df=boxes_df,
        ignore_df=ignore_df,
    )
    assert res["AP"] == pytest.approx(1.0, abs=1e-4)
    assert res["AP50"] == pytest.approx(1.0, abs=1e-4)
    assert res["AR500"] == pytest.approx(1.0, abs=1e-4)


def test_sanity_2_predictions_inside_ignore_add_zero_fp() -> None:
    """Sanity check 2 (§3.2): Predictions placed only inside ignore regions add zero FP across all classes."""
    manifest_df, boxes_df, ignore_df = _make_synthetic_dataset()

    # Keep 75% of GT boxes with varying scores so AP is strictly in (0, 1) and sensitive to FPs
    rng = np.random.default_rng(42)
    base_preds = boxes_df.iloc[::2][["image_key", "cls", "x", "y", "w", "h"]].copy()
    base_preds["score"] = rng.uniform(0.3, 0.95, size=len(base_preds)).astype(np.float32)

    res_without_ig_preds = evaluate(
        base_preds,
        split="val",
        max_dets=[1, 10, 500],
        manifest_df=manifest_df,
        boxes_df=boxes_df,
        ignore_df=ignore_df,
    )

    # Add 40 high-confidence predictions across all 8 classes strictly inside the ignore regions
    ig_preds_rows = []
    for cls_id in range(NUM_CLASSES):
        for k in range(5):
            ig_preds_rows.append(
                {
                    "image_key": "clip_test_0000001",
                    "cls": cls_id,
                    "score": np.float32(0.99),
                    "x": float(120 + k * 40),
                    "y": float(680 + cls_id * 20),
                    "w": 25.0,
                    "h": 18.0,
                }
            )
            ig_preds_rows.append(
                {
                    "image_key": "clip_test_0000002",
                    "cls": cls_id,
                    "score": np.float32(0.99),
                    "x": float(220 + k * 45),
                    "y": float(620 + cls_id * 25),
                    "w": 28.0,
                    "h": 20.0,
                }
            )

    combined_preds = pd.concat([base_preds, pd.DataFrame(ig_preds_rows)], ignore_index=True)
    res_with_ig_preds = evaluate(
        combined_preds,
        split="val",
        max_dets=[1, 10, 500],
        manifest_df=manifest_df,
        boxes_df=boxes_df,
        ignore_df=ignore_df,
    )

    assert res_with_ig_preds["AP"] == pytest.approx(res_without_ig_preds["AP"], abs=1e-5)
    assert res_with_ig_preds["AP50"] == pytest.approx(res_without_ig_preds["AP50"], abs=1e-5)


def test_sanity_3_dense_scene_maxdets_100_vs_500() -> None:
    """Sanity check 3 (§3.2): With 300 GT detections in a dense scene, AP differs between maxDets=100 and 500."""
    manifest_df, boxes_df, ignore_df = _make_synthetic_dataset()
    rng = np.random.default_rng(123)
    preds_df = boxes_df[["image_key", "cls", "x", "y", "w", "h"]].copy()
    preds_df["score"] = rng.uniform(0.1, 0.99, size=len(preds_df)).astype(np.float32)

    res_100 = evaluate(
        preds_df,
        split="val",
        max_dets=[1, 10, 100],
        manifest_df=manifest_df,
        boxes_df=boxes_df,
        ignore_df=ignore_df,
    )
    res_500 = evaluate(
        preds_df,
        split="val",
        max_dets=[1, 10, 500],
        manifest_df=manifest_df,
        boxes_df=boxes_df,
        ignore_df=ignore_df,
    )

    assert res_500["AP"] > res_100["AP"] + 0.05
    assert res_500["AP"] == pytest.approx(1.0, abs=1e-4)
    assert res_500["ap_person"] == pytest.approx(1.0, abs=1e-4)
    assert res_100["ap_person"] < 0.40


def test_sanity_4_empty_predictions_give_zero_ap() -> None:
    """Sanity check 4 (§3.2): Empty predictions give AP = 0."""
    manifest_df, boxes_df, ignore_df = _make_synthetic_dataset()
    empty_df = pd.DataFrame(columns=["image_key", "cls", "score", "x", "y", "w", "h"])

    res = evaluate(
        empty_df,
        split="val",
        max_dets=[1, 10, 500],
        manifest_df=manifest_df,
        boxes_df=boxes_df,
        ignore_df=ignore_df,
    )
    assert res["AP"] == 0.0
    assert res["AP50"] == 0.0
    assert res["AR500"] == 0.0


def test_pycocotools_crosscheck_agrees_within_tolerance() -> None:
    """Cross-check faster-coco-eval against pycocotools on noisy predictions (must agree within 0.002)."""
    manifest_df, boxes_df, ignore_df = _make_synthetic_dataset()
    rng = np.random.default_rng(7)
    preds_df = boxes_df[["image_key", "cls", "x", "y", "w", "h"]].copy()
    preds_df["x"] += rng.normal(0.0, 2.5, size=len(preds_df))
    preds_df["y"] += rng.normal(0.0, 2.5, size=len(preds_df))
    preds_df["score"] = rng.uniform(0.05, 0.99, size=len(preds_df)).astype(np.float32)

    fast_res = evaluate(
        preds_df,
        split="val",
        max_dets=[1, 10, 500],
        backend="faster_coco_eval",
        manifest_df=manifest_df,
        boxes_df=boxes_df,
        ignore_df=ignore_df,
    )
    pycoco_res = evaluate(
        preds_df,
        split="val",
        max_dets=[1, 10, 500],
        backend="pycocotools",
        manifest_df=manifest_df,
        boxes_df=boxes_df,
        ignore_df=ignore_df,
    )

    for k in ("AP", "AP50", "AP75", "AR500"):
        assert abs(fast_res[k] - pycoco_res[k]) <= 0.002

