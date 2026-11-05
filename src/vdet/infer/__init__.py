"""Inference package: predictor interfaces, coordinate unscaling, tiling, TTA, and box fusion."""

from vdet.infer.predictors import (
    PREDICTION_COLUMNS,
    Predictor,
    clip_and_filter_boxes_xywh,
    empty_predictions_df,
    load_predictions_parquet,
    normalize_predictions_df,
    save_predictions_parquet,
    unletterbox_xywh,
    unscale_xywh,
)

__all__ = [
    "PREDICTION_COLUMNS",
    "Predictor",
    "clip_and_filter_boxes_xywh",
    "empty_predictions_df",
    "load_predictions_parquet",
    "normalize_predictions_df",
    "save_predictions_parquet",
    "unletterbox_xywh",
    "unscale_xywh",
]

