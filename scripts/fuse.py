#!/usr/bin/env python3
"""CLI to fuse multiple §6.3 prediction parquet files via WBF / Soft-NMS / NMS and optionally evaluate on val (U4.3)."""

from __future__ import annotations

import argparse
from pathlib import Path

from vdet.constants import DEFAULT_SEED, MAX_SUBMISSION_DETS
from vdet.eval.coco_eval import evaluate
from vdet.infer.fusion import fuse_predictions
from vdet.infer.predictors import load_predictions_parquet, save_predictions_parquet
from vdet.tracking.aggregate import aggregate_runs
from vdet.tracking.run import RunLogger
from vdet.utils import timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse multiple prediction parquets via WBF / Soft-NMS / NMS.")
    parser.add_argument("--preds", type=Path, nargs="+", required=True, help="Input prediction .parquet paths.")
    parser.add_argument("--weights", type=float, nargs="*", default=None, help="Optional per-model weights.")
    parser.add_argument("--method", type=str, default="wbf", choices=["wbf", "soft_nms", "nms"])
    parser.add_argument("--iou-thr", type=float, default=0.55, help="Fusion IoU threshold.")
    parser.add_argument("--skip-box-thr", type=float, default=0.001, help="Min confidence threshold before fusion.")
    parser.add_argument("--sigma", type=float, default=0.5, help="Gaussian decay sigma for Soft-NMS.")
    parser.add_argument("--conf-type", type=str, default="avg", choices=["avg", "max", "box_and_model_avg"])
    parser.add_argument("--max-det", type=int, default=MAX_SUBMISSION_DETS)
    parser.add_argument("--out", type=Path, required=True, help="Output fused .parquet path.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run_id to log in experiments/runs/.")
    parser.add_argument("--parent-run", type=str, default="", help="Optional parent run_id.")
    parser.add_argument("--hypothesis", type=str, default="", help="Optional hypothesis string.")
    parser.add_argument("--eval-val", action="store_true", help="Evaluate fused predictions on official val set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dfs = [load_predictions_parquet(p) for p in args.preds]
    weights = args.weights if (args.weights and len(args.weights) > 0) else [1.0] * len(dfs)

    with timer(f"fuse ({args.method}, n_models={len(dfs)}, iou={args.iou_thr})"):
        fused_df = fuse_predictions(
            preds_list=dfs,
            weights=weights,
            method=args.method,
            iou_thr=args.iou_thr,
            skip_box_thr=args.skip_box_thr,
            sigma=args.sigma,
            conf_type=args.conf_type,
            max_det=args.max_det,
            source_tag=args.run_id or f"fuse_{args.method}",
        )

    out_path = save_predictions_parquet(
        fused_df, args.out, conf_thresh=args.skip_box_thr, max_det_per_image=args.max_det
    )
    print(f"Saved {len(fused_df)} fused detections to {out_path.as_posix()}")

    if args.eval_val:
        with timer("evaluate (val)"):
            eval_metrics = evaluate(fused_df, split="val", per_res_bucket=True)
        print(
            f"=== Fused Val Results ({args.run_id or args.method}) ===\n"
            f"  AP      : {eval_metrics['AP']:.4f}\n"
            f"  AP50    : {eval_metrics['AP50']:.4f}\n"
            f"  AP75    : {eval_metrics['AP75']:.4f}\n"
            f"  APs/m/l : {eval_metrics['APs']:.4f} / {eval_metrics['APm']:.4f} / {eval_metrics['APl']:.4f}\n"
            f"  AR@500  : {eval_metrics['AR500']:.4f}"
        )
        if args.run_id:
            cfg = {
                "run_id": args.run_id,
                "unit": "U4.3",
                "family": "ensemble",
                "exp": args.method,
                "parent_run": args.parent_run,
                "hypothesis": args.hypothesis
                or f"Multi-model {args.method.upper()} fusion (iou={args.iou_thr}, weights={weights}).",
                "preds": [p.as_posix() for p in args.preds],
                "weights": weights,
                "iou_thr": args.iou_thr,
                "skip_box_thr": args.skip_box_thr,
                "conf_type": args.conf_type,
            }
            logger = RunLogger(
                run_id=args.run_id,
                config=cfg,
                seed=DEFAULT_SEED,
                hypothesis=cfg["hypothesis"],
                parent_run=args.parent_run,
                unit="U4.3",
                family="ensemble",
                exp=args.method,
            )
            save_predictions_parquet(
                fused_df, logger.run_dir / "preds_val_slim.parquet", conf_thresh=0.05
            )
            logger.finalize(eval_metrics=eval_metrics, checkpoint_uri=out_path.as_posix())
            aggregate_runs()


if __name__ == "__main__":
    main()

