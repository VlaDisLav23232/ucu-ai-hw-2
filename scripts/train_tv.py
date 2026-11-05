#!/usr/bin/env python3
"""CLI to train Torchvision Faster R-CNN (F1) or RetinaNet (F2) under Protocol P0 (U2.0)."""

from __future__ import annotations

import argparse
from pathlib import Path

from vdet.engine.train_tv import train_torchvision
from vdet.utils import load_yaml, timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN / RetinaNet model under Protocol P0.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional override for run_id.")
    parser.add_argument("--cache", type=str, default=None, help="Optional override for cache dir name.")
    parser.add_argument("--batch", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional max epochs override.")
    parser.add_argument("--time-budget-h", type=float, default=None, help="Optional GPU-hour budget override.")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. '0', 'cuda', or 'cpu').")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint (.pth) to resume from.")
    parser.add_argument(
        "--eval-val",
        action="store_true",
        help="Run official evaluation on original val frames after training completes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.cache is not None:
        cfg["cache"] = args.cache
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.time_budget_h is not None:
        cfg["time_budget_h"] = args.time_budget_h
    if args.device is not None:
        cfg["device"] = args.device

    with timer(f"train_tv ({args.config.name})"):
        res = train_torchvision(
            config=cfg,
            run_id=args.run_id,
            resume_path=args.resume,
            evaluate_full_val=args.eval_val or bool(cfg.get("evaluate_full_val", False)),
        )

    print(f"\n=== Run Completed: {res['run_id']} ===")
    print(f"  Run dir   : {res['run_dir']}")
    print(f"  Checkpoint: {res['checkpoint']}")
    if res["eval_metrics"] is not None:
        em = res["eval_metrics"]
        print(
            f"  Official Val AP={em['AP']:.4f} | AP50={em['AP50']:.4f} | "
            f"APs={em['APs']:.4f} | AR500={em['AR500']:.4f}"
        )


if __name__ == "__main__":
    main()

