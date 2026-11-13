"""Base Predictor interface, coordinate unscaling helpers, and prediction table I/O."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from vdet.constants import MAX_SUBMISSION_DETS, NUM_CLASSES

PREDICTION_COLUMNS: tuple[str, ...] = ("image_key", "cls", "score", "x", "y", "w", "h")


def empty_predictions_df(include_optional: bool = False) -> pd.DataFrame:
    """Create an empty prediction DataFrame conforming to the schema."""
    data = {
        "image_key": pd.Series(dtype="object"),
        "cls": pd.Series(dtype="int8"),
        "score": pd.Series(dtype="float32"),
        "x": pd.Series(dtype="float32"),
        "y": pd.Series(dtype="float32"),
        "w": pd.Series(dtype="float32"),
        "h": pd.Series(dtype="float32"),
    }
    if include_optional:
        data["source"] = pd.Series(dtype="object")
        data["track"] = pd.Series(dtype="int32")
    return pd.DataFrame(data)


def clip_and_filter_boxes_xywh(
    boxes_xywh: np.ndarray,
    orig_w: int | float,
    orig_h: int | float,
    min_size: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip [N, 4] top-left (x, y, w, h) boxes to [0, orig_w] x [0, orig_h] and return valid mask."""
    if len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=bool)

    arr = np.asarray(boxes_xywh, dtype=np.float64)
    x1 = np.clip(arr[:, 0], 0.0, float(orig_w))
    y1 = np.clip(arr[:, 1], 0.0, float(orig_h))
    x2 = np.clip(arr[:, 0] + arr[:, 2], 0.0, float(orig_w))
    y2 = np.clip(arr[:, 1] + arr[:, 3], 0.0, float(orig_h))
    w = x2 - x1
    h = y2 - y1

    valid = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(w) & np.isfinite(h) & (w > min_size) & (h > min_size)
    clipped = np.stack([x1, y1, w, h], axis=1).astype(np.float32)
    return clipped[valid], valid


def unscale_xywh(
    boxes_xywh: np.ndarray,
    resized_w: int | float,
    resized_h: int | float,
    orig_w: int | float,
    orig_h: int | float,
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Map [N, 4] (x, y, w, h) boxes from resized (resized_w, resized_h) back to original image pixels."""
    if len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=bool)
    arr = np.asarray(boxes_xywh, dtype=np.float64).copy()
    sx = float(orig_w) / float(resized_w)
    sy = float(orig_h) / float(resized_h)
    arr[:, 0] *= sx
    arr[:, 1] *= sy
    arr[:, 2] *= sx
    arr[:, 3] *= sy
    if clip:
        return clip_and_filter_boxes_xywh(arr, orig_w, orig_h)
    valid = (arr[:, 2] > 0) & (arr[:, 3] > 0)
    return arr[valid].astype(np.float32), valid


def unletterbox_xywh(
    boxes_xywh: np.ndarray,
    input_w: int | float,
    input_h: int | float,
    orig_w: int | float,
    orig_h: int | float,
    pad_left: float | None = None,
    pad_top: float | None = None,
    gain: float | None = None,
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Map [N, 4] (x, y, w, h) boxes from letterboxed (input_w, input_h) coordinates to original pixels."""
    if len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=bool)

    if gain is None:
        gain = min(float(input_w) / float(orig_w), float(input_h) / float(orig_h))
    if pad_left is None:
        pad_left = (float(input_w) - float(orig_w) * gain) / 2.0
    if pad_top is None:
        pad_top = (float(input_h) - float(orig_h) * gain) / 2.0

    arr = np.asarray(boxes_xywh, dtype=np.float64).copy()
    arr[:, 0] = (arr[:, 0] - pad_left) / gain
    arr[:, 1] = (arr[:, 1] - pad_top) / gain
    arr[:, 2] = arr[:, 2] / gain
    arr[:, 3] = arr[:, 3] / gain
    if clip:
        return clip_and_filter_boxes_xywh(arr, orig_w, orig_h)
    valid = (arr[:, 2] > 0) & (arr[:, 3] > 0)
    return arr[valid].astype(np.float32), valid


def normalize_predictions_df(
    df: pd.DataFrame,
    conf_thresh: float = 0.001,
    max_det_per_image: int = MAX_SUBMISSION_DETS,
) -> pd.DataFrame:
    """Enforce column types, filter invalid boxes/scores, sort descending, and cap per image."""
    if df is None or df.empty:
        return empty_predictions_df()

    missing = [col for col in PREDICTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Prediction DataFrame missing required columns: {missing}")

    out = df.copy()
    out["image_key"] = out["image_key"].astype(str)
    out["cls"] = out["cls"].astype("int8")
    out["score"] = out["score"].astype("float32")
    out["x"] = out["x"].astype("float32")
    out["y"] = out["y"].astype("float32")
    out["w"] = out["w"].astype("float32")
    out["h"] = out["h"].astype("float32")

    valid_mask = (
        out["cls"].between(0, NUM_CLASSES - 1)
        & np.isfinite(out["score"])
        & (out["score"] >= float(conf_thresh))
        & (out["score"] <= 1.0)
        & np.isfinite(out["x"])
        & np.isfinite(out["y"])
        & np.isfinite(out["w"])
        & np.isfinite(out["h"])
        & (out["w"] > 0.0)
        & (out["h"] > 0.0)
    )
    out = out[valid_mask]
    if out.empty:
        return empty_predictions_df()

    out = out.sort_values(["image_key", "score"], ascending=[True, False])
    if max_det_per_image > 0:
        out = out.groupby("image_key", sort=False).head(max_det_per_image)

    return out.reset_index(drop=True)


def save_predictions_parquet(
    preds_df: pd.DataFrame,
    out_path: str | Path,
    conf_thresh: float = 0.001,
    max_det_per_image: int = MAX_SUBMISSION_DETS,
) -> Path:
    """Normalize and write a prediction DataFrame to a zstd-compressed Parquet file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean_df = normalize_predictions_df(
        preds_df, conf_thresh=conf_thresh, max_det_per_image=max_det_per_image
    )
    clean_df.to_parquet(p, index=False, compression="zstd")
    return p


def load_predictions_parquet(path: str | Path) -> pd.DataFrame:
    """Load a prediction table Parquet file."""
    return pd.read_parquet(Path(path))


class Predictor(ABC):
    """Abstract base interface for all frame-level object detectors."""

    @abstractmethod
    def predict(self, frames: pd.DataFrame | Sequence[str]) -> pd.DataFrame:
        """Predict bounding boxes on original images and return a prediction DataFrame."""
        raise NotImplementedError

