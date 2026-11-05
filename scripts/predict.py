#!/usr/bin/env python3
"""CLI to run inference on original val or test images for any checkpoint (.pt or .pth), with optional Tiling (U4.1) and TTA (U4.2)."""

from __future__ import annotations

import argparse
from pathlib import Path

from vdet.constants import MAX_SUBMISSION_DETS
from vdet.data.manifest import load_manifest
from vdet.infer.predictors import Predictor, save_predictions_parquet
from vdet.infer.tiling import TiledPredictor
from vdet.infer.tta import TTAPredictor
from vdet.models.tv_builders import TorchvisionPredictor
from vdet.models.ultra import UltralyticsPredictor
from vdet.paths import P
from vdet.utils import load_yaml, timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict boxes on original val/test frames.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model checkpoint (.pt or .pth).")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config for Torchvision checkpoints.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    parser.add_argument("--out", type=Path, required=True, help="Output predictions .parquet path.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=MAX_SUBMISSION_DETS)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--source-tag", type=str, default=None)
    parser.add_argument("--tiling", action="store_true", help="Enable SAHI-style sliced inference (U4.1).")
    parser.add_argument("--slice-size", type=int, default=800, help="Tile slice size in pixels.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Tile overlap ratio.")
    parser.add_argument("--tta", action="store_true", help="Enable multi-scale + hflip TTA (U4.2).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_df = load_manifest()
    sub = manifest_df[manifest_df["split"] == args.split].reset_index(drop=True)
    if args.max_frames is not None:
        sub = sub.head(args.max_frames).reset_index(drop=True)

    is_tv = args.weights.suffix.lower() in {".pth", ".ckpt"}
    if is_tv:
        cfg = load_yaml(args.config) if args.config is not None else None
        predictor: Predictor = TorchvisionPredictor(
            weights_path=args.weights,
            config=cfg,
            conf=args.conf,
            max_det=args.max_det,
            batch_size=args.batch,
            device=args.device,
            source_tag=args.source_tag or "tv",
        )
    else:
        predictor = UltralyticsPredictor(
            weights_path=args.weights,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            batch_size=args.batch,
            device=args.device,
            source_tag=args.source_tag or "ultra",
        )

    if args.tta:
        predictor = TTAPredictor(
            base_predictor=predictor,
            scales=(1.0, 1.25),
            hflip=True,
            conf_thresh=args.conf,
            max_det=args.max_det,
            source_tag=args.source_tag or "tta",
        )
    if args.tiling:
        predictor = TiledPredictor(
            base_predictor=predictor,
            slice_size=args.slice_size,
            overlap_ratio=args.overlap,
            conf_thresh=args.conf,
            max_det=args.max_det,
            source_tag=args.source_tag or "tiled",
        )

    with timer(f"predict ({args.split}, n={len(sub)} frames)"):
        preds_df = predictor.predict(sub, data_root=P.data)

    out_path = save_predictions_parquet(
        preds_df, args.out, conf_thresh=args.conf, max_det_per_image=args.max_det
    )
    size_mb = out_path.stat().st_size / (1024**2)
    print(
        f"Saved {len(preds_df)} detections across {len(sub)} frames to "
        f"{out_path.as_posix()} ({size_mb:.2f} MB)"
    )


if __name__ == "__main__":
    main()
