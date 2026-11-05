#!/usr/bin/env python3
"""CLI to evaluate a predictions parquet file with the official local evaluator (U0.5)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vdet.constants import DEFAULT_MAX_DETS
from vdet.eval.coco_eval import evaluate
from vdet.paths import P
from vdet.utils import load_json, save_json, timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction table with faster-coco-eval.")
    parser.add_argument("--preds", type=Path, required=True, help="Path to predictions .parquet.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument(
        "--backend",
        type=str,
        default="faster_coco_eval",
        choices=["faster_coco_eval", "pycocotools"],
    )
    parser.add_argument(
        "--max-dets",
        type=int,
        nargs=3,
        default=DEFAULT_MAX_DETS,
        help="COCO maxDets tuple (default: 1 10 500).",
    )
    parser.add_argument(
        "--per-res-bucket",
        action="store_true",
        help="Include per-resolution-bucket breakdown.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSON path (defaults to <run_dir>/eval_val.json if --run-dir given).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional experiment run directory to update eval_val.json and summary.json.",
    )
    parser.add_argument(
        "--subset-from-preds",
        action="store_true",
        help="Restrict GT evaluation to the unique image_keys present in --preds (for partial smoke tests).",
    )
    parser.add_argument("--cache-root", type=Path, default=P.cache)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds_df = pd.read_parquet(args.preds)
    subset_keys = (
        preds_df["image_key"].astype(str).unique().tolist()
        if args.subset_from_preds and not preds_df.empty
        else None
    )
    with timer(f"evaluate ({args.split}, backend={args.backend})"):
        metrics = evaluate(
            preds_df=preds_df,
            split=args.split,
            subset=subset_keys,
            max_dets=args.max_dets,
            backend=args.backend,
            per_res_bucket=args.per_res_bucket,
            cache_root=args.cache_root,
        )

    print(
        f"\n=== Evaluation Results ({args.split}, n={metrics['n_images']} frames) ===\n"
        f"  AP      : {metrics['AP']:.4f}\n"
        f"  AP50    : {metrics['AP50']:.4f}\n"
        f"  AP75    : {metrics['AP75']:.4f}\n"
        f"  APs/m/l : {metrics['APs']:.4f} / {metrics['APm']:.4f} / {metrics['APl']:.4f}\n"
        f"  AR@500  : {metrics['AR500']:.4f}"
    )

    out_path = args.out
    if out_path is None and args.run_dir is not None:
        out_path = args.run_dir / f"eval_{args.split}.json"
    if out_path is not None:
        save_json(metrics, out_path)
        print(f"  Saved evaluation JSON to {out_path.as_posix()}")

    if args.run_dir is not None:
        summary_path = args.run_dir / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else {}
        for k in (
            "AP",
            "AP50",
            "AP75",
            "APs",
            "APm",
            "APl",
            "AR500",
            "ap_person",
            "ap_bicycle",
            "ap_car",
            "ap_van",
            "ap_truck",
            "ap_tricycle",
            "ap_bus",
            "ap_motor",
        ):
            if k in metrics:
                summary[k] = metrics[k]
        save_json(summary, summary_path)
        print(f"  Updated run summary at {summary_path.as_posix()}")


if __name__ == "__main__":
    main()

