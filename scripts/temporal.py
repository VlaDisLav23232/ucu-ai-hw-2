#!/usr/bin/env python3
"""CLI to apply temporal post-processing (U4.4 tracklet rescoring + gap propagation or U4.5 Seq-NMS)."""

from __future__ import annotations

import argparse
from pathlib import Path

from vdet.constants import DEFAULT_SEED, MAX_SUBMISSION_DETS
from vdet.eval.coco_eval import evaluate
from vdet.infer.predictors import load_predictions_parquet, save_predictions_parquet
from vdet.temporal.rescoring import apply_temporal_rescoring
from vdet.temporal.seqnms import apply_seq_nms
from vdet.tracking.aggregate import aggregate_runs
from vdet.tracking.run import RunLogger
from vdet.utils import timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply temporal post-processing to a prediction parquet.")
    parser.add_argument("--preds", type=Path, required=True, help="Input predictions .parquet path.")
    parser.add_argument("--out", type=Path, required=True, help="Output predictions .parquet path.")
    parser.add_argument("--method", type=str, default="rescore", choices=["rescore", "seqnms"])
    parser.add_argument("--iou-link", type=float, default=0.45, help="Tracklet linking IoU threshold.")
    parser.add_argument("--alpha", type=float, default=0.70, help="Original score weight in tracklet smoothing.")
    parser.add_argument("--topk-ratio", type=float, default=0.50, help="Top-k fraction of tracklet scores.")
    parser.add_argument("--no-propagation", action="store_true", help="Disable linear gap propagation.")
    parser.add_argument("--prop-min-score", type=float, default=0.25, help="Min mean tracklet score for gap interpolation.")
    parser.add_argument("--max-gap", type=int, default=2, help="Max missing frames to interpolate across.")
    parser.add_argument("--motion-comp", action="store_true", help="Enable ORB+RANSAC homography motion compensation (U4.6).")
    parser.add_argument("--max-det", type=int, default=MAX_SUBMISSION_DETS)
    parser.add_argument("--run-id", type=str, default=None, help="Optional run_id to log in experiments/runs/.")
    parser.add_argument("--parent-run", type=str, default="", help="Optional parent run_id.")
    parser.add_argument("--hypothesis", type=str, default="", help="Optional hypothesis string.")
    parser.add_argument("--eval-val", action="store_true", help="Evaluate temporal predictions on official val set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds_df = load_predictions_parquet(args.preds)

    with timer(f"temporal ({args.method}, iou_link={args.iou_link}, prop={not args.no_propagation})"):
        if args.method == "seqnms":
            out_df = apply_seq_nms(
                preds_df=preds_df,
                iou_link=args.iou_link,
                max_det=args.max_det,
                source_tag=args.run_id or "seqnms",
            )
        else:
            out_df = apply_temporal_rescoring(
                preds_df=preds_df,
                iou_link=args.iou_link,
                alpha=args.alpha,
                topk_ratio=args.topk_ratio,
                enable_propagation=not args.no_propagation,
                prop_min_score=args.prop_min_score,
                max_gap=args.max_gap,
                use_motion_comp=args.motion_comp,
                max_det=args.max_det,
                source_tag=args.run_id or "temporal_rescore",
            )

    out_path = save_predictions_parquet(out_df, args.out, conf_thresh=0.001, max_det_per_image=args.max_det)
    print(f"Saved {len(out_df)} temporal detections to {out_path.as_posix()}")

    if args.eval_val:
        with timer("evaluate (val)"):
            eval_metrics = evaluate(out_df, split="val", per_res_bucket=True)
        print(
            f"=== Temporal Val Results ({args.run_id or args.method}) ===\n"
            f"  AP      : {eval_metrics['AP']:.4f}\n"
            f"  AP50    : {eval_metrics['AP50']:.4f}\n"
            f"  AP75    : {eval_metrics['AP75']:.4f}\n"
            f"  APs/m/l : {eval_metrics['APs']:.4f} / {eval_metrics['APm']:.4f} / {eval_metrics['APl']:.4f}\n"
            f"  AR@500  : {eval_metrics['AR500']:.4f}"
        )
        if args.run_id:
            cfg = {
                "run_id": args.run_id,
                "unit": "U4.4" if args.method == "rescore" else "U4.5",
                "family": "temporal",
                "exp": args.method,
                "parent_run": args.parent_run,
                "hypothesis": args.hypothesis
                or f"Temporal {args.method} (iou_link={args.iou_link}, alpha={args.alpha}, prop={not args.no_propagation}).",
                "input_preds": args.preds.as_posix(),
                "method": args.method,
                "iou_link": args.iou_link,
                "alpha": args.alpha,
                "enable_propagation": not args.no_propagation,
                "motion_comp": args.motion_comp,
            }
            logger = RunLogger(
                run_id=args.run_id,
                config=cfg,
                seed=DEFAULT_SEED,
                hypothesis=cfg["hypothesis"],
                parent_run=args.parent_run,
                unit=cfg["unit"],
                family="temporal",
                exp=args.method,
            )
            save_predictions_parquet(
                out_df, logger.run_dir / "preds_val_slim.parquet", conf_thresh=0.05
            )
            logger.finalize(eval_metrics=eval_metrics, checkpoint_uri=out_path.as_posix())
            aggregate_runs()


if __name__ == "__main__":
    main()

