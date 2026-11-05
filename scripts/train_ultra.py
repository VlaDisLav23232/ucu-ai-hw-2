#!/usr/bin/env python3
"""CLI to train YOLO11 / RT-DETR models with Ultralytics under Protocol P0 (U1.1–U1.3, F3, F4)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vdet.models.ultra import train_ultralytics
from vdet.utils import load_resolved_config, timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 / RT-DETR model with vdet RunLogger.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config YAML (e.g. configs/yolo/e0_base.yaml).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run_id.")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override input image size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument(
        "--time-budget-h",
        type=float,
        default=None,
        help="Override time budget in GPU-hours (pass 0 to disable time limit).",
    )
    parser.add_argument(
        "--eval-val",
        action="store_true",
        help="Run full original-image validation with our official evaluator after training.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g. '0' or 'cpu').")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict[str, Any] = {}
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.time_budget_h is not None:
        overrides["time_budget_h"] = args.time_budget_h if args.time_budget_h > 0 else None
    if args.device is not None:
        overrides["device"] = args.device

    cfg = load_resolved_config(config_path=args.config, cli_overrides=overrides)
    do_eval_val = args.eval_val or bool(cfg.get("auto_eval_val", False))

    with timer(f"train_ultra ({args.config.name})"):
        res = train_ultralytics(
            config=cfg,
            run_id=args.run_id,
            evaluate_full_val=do_eval_val,
        )

    print(f"\n=== Run Completed: {res['run_id']} ===")
    print(f"  Run dir   : {res['run_dir'].as_posix()}")
    print(f"  Checkpoint: {res['checkpoint'].as_posix()}")
    if res["eval_metrics"]:
        em = res["eval_metrics"]
        print(
            f"  Official Val AP={em['AP']:.4f} | AP50={em['AP50']:.4f} | "
            f"APs={em['APs']:.4f} | AR500={em['AR500']:.4f}"
        )


if __name__ == "__main__":
    main()

