"""Fixed experimental protocol P0 and helpers to resolve in-notebook experiment configs.

Every experiment config in the notebooks is layered on top of this protocol.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from vdet.paths import P
from vdet.utils import deep_merge

# Fixed Experimental Protocol P0 — shared by every training run.
PROTOCOL: dict[str, Any] = {
    "protocol_id": "P0",
    "seed": 42,
    "data": {
        "train_split": "train",
        "holdout_split": "val",
        "frame_step_k": 5,
        "val_proxy_k": 5,
        "ignore_policy": "drop",  # fixed by the ignore-policy ablation (notebook 02)
        "cache_long_side": 1920,
        "cache_name": "k5_drop_L1920",
        "jpeg_quality": 95,
    },
    "input": {"imgsz": 1280, "min_size": 720, "max_size": 1280},
    "training": {
        "pretrained": "coco",
        "time_budget_h": 1.5,
        "warmup_epochs": 1.0,
        "close_mosaic_frac": 0.10,
    },
    "augmentation": {
        "hflip": 0.5,
        "vflip": 0.0,
        "degrees": 0.0,
        "scale_min": 0.5,
        "scale_max": 1.0,
        "hsv_h": 0.015,
        "hsv_s": 0.4,
        "hsv_v": 0.3,
    },
    "inference": {
        "conf_thresh": 0.001,
        "slim_conf_thresh": 0.05,
        "nms_iou": 0.6,
        "max_det": 500,
        "imgsz": 1280,
    },
    "evaluation": {
        "evaluator": "faster-coco-eval",
        "iou_thresholds": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        "max_dets": [1, 10, 500],
        "size_buckets": {"small": [0, 1024], "medium": [1024, 9216], "large": [9216, 10_000_000_000]},
    },
    "fps_benchmark": {
        "device": "cuda",
        "batch_size": 1,
        "fp16": True,
        "imgsz": 1280,
        "warmup_frames": 20,
        "timed_frames": 200,
        "exclude_jpeg_decode": True,
    },
}


def resolve_config(cfg: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Layer an experiment config on top of the protocol: PROTOCOL <- cfg <- overrides."""
    resolved = deep_merge(copy.deepcopy(PROTOCOL), copy.deepcopy(cfg))
    if overrides:
        resolved = deep_merge(resolved, overrides)
    return resolved


def write_model_yaml(name: str, text: str) -> Path:
    """Write an Ultralytics architecture YAML (defined inline in a notebook) and return its path.

    Ultralytics needs architecture definitions as files, so notebooks keep the YAML text in a
    cell and materialise it under ``cache/model_yamls/`` right before training.
    """
    out_dir = P.cache / "model_yamls"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    out.write_text(text.strip() + "\n", encoding="utf-8")
    return out
