#!/usr/bin/env python3
"""End-to-end M4 Pipeline Runner: WBF Ensembling + Temporal Tracklet Rescoring/Propagation + Waterfall Plot + Final Submissions (U4.3–U4.8)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from vdet.constants import DEFAULT_SEED, MAX_SUBMISSION_DETS
from vdet.data.manifest import load_manifest
from vdet.eval.coco_eval import evaluate
from vdet.infer.fusion import fuse_predictions
from vdet.infer.predictors import (
    load_predictions_parquet,
    save_predictions_parquet,
)
from vdet.models.tv_builders import TorchvisionPredictor
from vdet.models.ultra import UltralyticsPredictor
from vdet.paths import P
from vdet.submission import to_csv, validate
from vdet.temporal.rescoring import apply_temporal_rescoring
from vdet.tracking.aggregate import aggregate_runs
from vdet.tracking.plots import pipeline_waterfall
from vdet.tracking.run import RunLogger
from vdet.utils import load_json, load_yaml, save_json, timer


def _ensure_split_predictions(
    model_spec: dict[str, Any],
    split: str,
    manifest_df: pd.DataFrame,
) -> Path | None:
    """Return path to `runs/<run_id>/preds_<split>.parquet`, generating it from checkpoint if needed."""
    rid = str(model_spec["run_id"])
    pred_path = P.runs / rid / f"preds_{split}.parquet"
    if pred_path.exists():
        return pred_path

    w_path = P.root / str(model_spec["weights"])
    if not w_path.exists():
        return None

    sub_m = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
    imgsz = int(model_spec.get("imgsz", 1280))
    batch = int(model_spec.get("batch", 4))

    print(f"[pipeline] Generating {split} predictions for {rid} from {w_path.name}...")
    if w_path.suffix.lower() in {".pth", ".ckpt"}:
        cfg_p = P.root / str(model_spec["config"]) if model_spec.get("config") else None
        cfg = load_yaml(cfg_p) if (cfg_p and cfg_p.exists()) else None
        predictor = TorchvisionPredictor(
            weights_path=w_path,
            config=cfg,
            conf=0.001,
            max_det=MAX_SUBMISSION_DETS,
            batch_size=batch,
            source_tag=rid,
        )
    else:
        predictor = UltralyticsPredictor(
            weights_path=w_path,
            imgsz=imgsz,
            conf=0.001,
            iou=0.6,
            max_det=MAX_SUBMISSION_DETS,
            batch_size=batch,
            source_tag=rid,
        )

    with timer(f"predict ({rid}, {split}, n={len(sub_m)})"):
        df = predictor.predict(sub_m, data_root=P.data)
    save_predictions_parquet(df, pred_path, conf_thresh=0.001, max_det_per_image=MAX_SUBMISSION_DETS)
    return pred_path


def _log_pipeline_stage(
    run_id: str,
    unit: str,
    family: str,
    exp: str,
    parent_run: str,
    hypothesis: str,
    config: dict[str, Any],
    preds_val_df: pd.DataFrame,
    eval_metrics: dict[str, Any],
    ckpt_uri: str,
) -> dict[str, Any]:
    logger = RunLogger(
        run_id=run_id,
        config=config,
        seed=DEFAULT_SEED,
        hypothesis=hypothesis,
        parent_run=parent_run,
        unit=unit,
        family=family,
        exp=exp,
    )
    save_predictions_parquet(
        preds_val_df,
        logger.run_dir / "preds_val_slim.parquet",
        conf_thresh=0.05,
    )
    return logger.finalize(eval_metrics=eval_metrics, checkpoint_uri=ckpt_uri)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute M4 final ensemble + temporal pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=P.configs / "pipeline" / "final.yaml",
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--predict-test",
        action="store_true",
        help="Generate missing test predictions and build final Kaggle submissions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    manifest_df = load_manifest()

    models_cfg = cfg.get("models", [])
    fus_cfg = cfg.get("fusion", {})
    tmp_cfg = cfg.get("temporal", {})

    # 1. Collect all available component validation predictions
    val_dfs: list[pd.DataFrame] = []
    val_weights: list[float] = []
    active_models: list[dict[str, Any]] = []

    for m_spec in models_cfg:
        val_p = _ensure_split_predictions(m_spec, split="val", manifest_df=manifest_df)
        if val_p is not None and val_p.exists():
            val_dfs.append(load_predictions_parquet(val_p))
            val_weights.append(float(m_spec.get("wbf_weight", 1.0)))
            active_models.append(m_spec)

    if len(val_dfs) < 2:
        raise RuntimeError(f"Need at least 2 models with val predictions for ensembling; found {len(val_dfs)}.")

    active_ids = [str(m["run_id"]) for m in active_models]
    print(f"[pipeline] Active ensemble models ({len(active_ids)}): {active_ids} with weights {val_weights}")

    out_dir = P.runs / "pipeline-final"
    out_dir.mkdir(parents=True, exist_ok=True)
    wbf_val_path = out_dir / "preds_val_wbf.parquet"
    rescore_only_val_path = out_dir / "preds_val_temporal_rescore.parquet"
    final_val_path = out_dir / "preds_val_final.parquet"

    if (
        wbf_val_path.exists()
        and rescore_only_val_path.exists()
        and final_val_path.exists()
        and (P.exp_runs_dir / "pipeline-final" / "summary.json").exists()
        and (P.images / "pipeline_waterfall.png").exists()
    ):
        print("[pipeline] Validation stages (ensemble-wbf, temporal-rescore, pipeline-final) already completed; reusing results.")
    else:
        if wbf_val_path.exists():
            print(f"[pipeline] Reusing existing WBF val predictions from {wbf_val_path}")
            wbf_val_df = load_predictions_parquet(wbf_val_path)
        else:
            with timer(f"U4.3 WBF Fusion (val, {len(val_dfs)} models)"):
                wbf_val_df = fuse_predictions(
                    preds_list=val_dfs,
                    weights=val_weights,
                    method=str(fus_cfg.get("method", "wbf")),
                    iou_thr=float(fus_cfg.get("iou_thr", 0.55)),
                    skip_box_thr=float(fus_cfg.get("skip_box_thr", 0.001)),
                    conf_type=str(fus_cfg.get("conf_type", "avg")),
                    max_det=int(fus_cfg.get("max_det", MAX_SUBMISSION_DETS)),
                    manifest_df=manifest_df,
                    source_tag="ensemble-wbf",
                )
            save_predictions_parquet(wbf_val_df, wbf_val_path, conf_thresh=0.001)
        wbf_eval = evaluate(wbf_val_df, split="val", per_res_bucket=True)
        _log_pipeline_stage(
            run_id="ensemble-wbf",
            unit="U4.3",
            family="pipeline",
            exp="wbf",
            parent_run="rtdetr-e0",
            hypothesis=f"Cross-family WBF ensemble of {active_ids} (weights={val_weights}, iou=0.55).",
            config={"models": active_ids, "weights": val_weights, **fus_cfg},
            preds_val_df=wbf_val_df,
            eval_metrics=wbf_eval,
            ckpt_uri=wbf_val_path.as_posix(),
        )
        print(
            f"  -> [U4.3 WBF Ensemble] Val AP={wbf_eval['AP']:.4f} | AP50={wbf_eval['AP50']:.4f} | "
            f"APs={wbf_eval['APs']:.4f} | AR500={wbf_eval['AR500']:.4f}"
        )

        # 3. Run U4.4 Temporal Ablation Arm 1: Tracklet Rescoring ONLY (no gap propagation) on val
        with timer("U4.4 Temporal Tracklet Rescoring Only (val)"):
            rescore_only_val_df = apply_temporal_rescoring(
                preds_df=wbf_val_df,
                manifest_df=manifest_df,
                iou_link=float(tmp_cfg.get("iou_link", 0.45)),
                alpha=float(tmp_cfg.get("alpha", 0.70)),
                topk_ratio=float(tmp_cfg.get("topk_ratio", 0.50)),
                single_frame_decay=float(tmp_cfg.get("single_frame_decay", 0.96)),
                enable_propagation=False,
                source_tag="temporal-rescore",
            )
        save_predictions_parquet(rescore_only_val_df, rescore_only_val_path, conf_thresh=0.001)
        rescore_eval = evaluate(rescore_only_val_df, split="val", per_res_bucket=True)
        _log_pipeline_stage(
            run_id="temporal-rescore",
            unit="U4.4",
            family="pipeline",
            exp="rescore",
            parent_run="ensemble-wbf",
            hypothesis="Tracklet IoU linking + top-k score smoothing (without gap propagation) on WBF ensemble.",
            config={**tmp_cfg, "enable_propagation": False},
            preds_val_df=rescore_only_val_df,
            eval_metrics=rescore_eval,
            ckpt_uri=rescore_only_val_path.as_posix(),
        )
        print(
            f"  -> [U4.4 Temporal Rescore Only] Val AP={rescore_eval['AP']:.4f} | AP50={rescore_eval['AP50']:.4f} | "
            f"APs={rescore_eval['APs']:.4f} | AR500={rescore_eval['AR500']:.4f}"
        )

        # 4. Run U4.4 + U4.8 Final Pipeline: Tracklet Rescoring + Linear Gap Propagation on val
        with timer("U4.4/U4.8 Temporal Rescoring + Gap Propagation (val)"):
            final_val_df = apply_temporal_rescoring(
                preds_df=wbf_val_df,
                manifest_df=manifest_df,
                iou_link=float(tmp_cfg.get("iou_link", 0.45)),
                alpha=float(tmp_cfg.get("alpha", 0.70)),
                topk_ratio=float(tmp_cfg.get("topk_ratio", 0.50)),
                single_frame_decay=float(tmp_cfg.get("single_frame_decay", 0.96)),
                enable_propagation=True,
                prop_min_score=float(tmp_cfg.get("prop_min_score", 0.25)),
                prop_min_len=int(tmp_cfg.get("prop_min_len", 3)),
                max_gap=int(tmp_cfg.get("max_gap", 2)),
                gap_decay=float(tmp_cfg.get("gap_decay", 0.85)),
                source_tag="pipeline-final",
            )
        save_predictions_parquet(final_val_df, final_val_path, conf_thresh=0.001)
        final_eval = evaluate(final_val_df, split="val", per_res_bucket=True)
        _log_pipeline_stage(
            run_id="pipeline-final",
            unit="U4.8",
            family="pipeline",
            exp="final",
            parent_run="temporal-rescore",
            hypothesis=str(cfg.get("hypothesis", "WBF ensemble + temporal tracklet rescoring & gap propagation.")),
            config=cfg,
            preds_val_df=final_val_df,
            eval_metrics=final_eval,
            ckpt_uri=final_val_path.as_posix(),
        )
        print(
            f"  -> [U4.8 Pipeline Final] Val AP={final_eval['AP']:.4f} | AP50={final_eval['AP50']:.4f} | "
            f"APs={final_eval['APs']:.4f} | AR500={final_eval['AR500']:.4f}"
        )

        # 5. Also demonstrate temporal bonus on single-family F3 (yolo-e1) so single-family delta is recorded
        yolo_e1_val = P.runs / "yolo-e1" / "preds_val.parquet"
        if yolo_e1_val.exists():
            ye1_df = load_predictions_parquet(yolo_e1_val)
            ye1_temp_df = apply_temporal_rescoring(
                preds_df=ye1_df,
                manifest_df=manifest_df,
                iou_link=float(tmp_cfg.get("iou_link", 0.45)),
                alpha=float(tmp_cfg.get("alpha", 0.70)),
                enable_propagation=True,
                source_tag="yolo-e1-temporal",
            )
            ye1_temp_eval = evaluate(ye1_temp_df, split="val", per_res_bucket=False)
            _log_pipeline_stage(
                run_id="yolo-e1-temporal",
                unit="U4.4",
                family="temporal",
                exp="yolo_rescore",
                parent_run="yolo-e1",
                hypothesis="Temporal tracklet rescoring + gap propagation applied directly to single-model yolo-e1.",
                config=tmp_cfg,
                preds_val_df=ye1_temp_df,
                eval_metrics=ye1_temp_eval,
                ckpt_uri=yolo_e1_val.as_posix(),
            )

        aggregate_runs()

        # 6. Generate Waterfall Table & Plot (images/pipeline_waterfall.png)
        waterfall_steps: list[tuple[str, float]] = []
        for step_id, label in [
            ("yolo-e0", "F3 Baseline\n(YOLO11s @1280)"),
            ("yolo-e1", "+ Hi-Res 1600\n(yolo-e1)"),
            ("rtdetr-e0", "F4 Query-Based\n(RT-DETR-L)"),
            ("ensemble-wbf", "+ Multi-Model\nWBF Ensemble"),
            ("temporal-rescore", "+ Tracklet\nScore Smoothing"),
            ("pipeline-final", "+ Gap\nPropagation"),
        ]:
            s_path = P.exp_runs_dir / step_id / "summary.json"
            if s_path.exists():
                ap_val = float(load_json(s_path).get("AP", 0.0))
                waterfall_steps.append((label, ap_val))

        if waterfall_steps:
            wf_fig = pipeline_waterfall(waterfall_steps, P.images / "pipeline_waterfall.png")
            save_json(
                {"steps": [{"stage": k, "AP": v} for k, v in waterfall_steps]},
                P.experiments / "pipeline_waterfall.json",
            )
            print(f"[pipeline] Saved waterfall figure to {wf_fig}")

    # 7. Optional Test-Split Inference & Final Kaggle Submissions
    if args.predict_test:
        test_dfs: list[pd.DataFrame] = []
        test_weights: list[float] = []
        for m_spec in active_models:
            test_p = _ensure_split_predictions(m_spec, split="test", manifest_df=manifest_df)
            if test_p is not None and test_p.exists():
                test_dfs.append(load_predictions_parquet(test_p))
                test_weights.append(float(m_spec.get("wbf_weight", 1.0)))

        if len(test_dfs) >= 2:
            wbf_test_path = out_dir / "preds_test_wbf.parquet"
            if wbf_test_path.exists():
                print(f"[pipeline] Reusing existing WBF test predictions from {wbf_test_path}")
                wbf_test_df = load_predictions_parquet(wbf_test_path)
            else:
                with timer(f"U4.3 WBF Fusion (test, {len(test_dfs)} models)"):
                    wbf_test_df = fuse_predictions(
                        preds_list=test_dfs,
                        weights=test_weights,
                        method=str(fus_cfg.get("method", "wbf")),
                        iou_thr=float(fus_cfg.get("iou_thr", 0.55)),
                        skip_box_thr=float(fus_cfg.get("skip_box_thr", 0.001)),
                        conf_type=str(fus_cfg.get("conf_type", "avg")),
                        max_det=int(fus_cfg.get("max_det", MAX_SUBMISSION_DETS)),
                        manifest_df=manifest_df,
                        source_tag="ensemble-wbf",
                    )
                save_predictions_parquet(wbf_test_df, wbf_test_path, conf_thresh=0.001)

            wbf_sub_csv = P.subs / "ensemble_wbf.csv"
            to_csv(wbf_test_df, wbf_sub_csv, P.test_images_csv)
            rep_wbf = validate(wbf_sub_csv, P.test_images_csv)
            print(f"[submission] Saved & validated {wbf_sub_csv}: {rep_wbf}")

            final_test_path = out_dir / "preds_test_final.parquet"
            if final_test_path.exists():
                print(f"[pipeline] Reusing existing final test predictions from {final_test_path}")
                final_test_df = load_predictions_parquet(final_test_path)
            else:
                with timer("U4.8 Temporal Rescoring + Gap Propagation (test)"):
                    final_test_df = apply_temporal_rescoring(
                        preds_df=wbf_test_df,
                        manifest_df=manifest_df,
                        iou_link=float(tmp_cfg.get("iou_link", 0.45)),
                        alpha=float(tmp_cfg.get("alpha", 0.70)),
                        topk_ratio=float(tmp_cfg.get("topk_ratio", 0.50)),
                        single_frame_decay=float(tmp_cfg.get("single_frame_decay", 0.96)),
                        enable_propagation=True,
                        prop_min_score=float(tmp_cfg.get("prop_min_score", 0.25)),
                        prop_min_len=int(tmp_cfg.get("prop_min_len", 3)),
                        max_gap=int(tmp_cfg.get("max_gap", 2)),
                        gap_decay=float(tmp_cfg.get("gap_decay", 0.85)),
                        source_tag="pipeline-final",
                    )
                save_predictions_parquet(final_test_df, final_test_path, conf_thresh=0.001)

            final_sub_csv = P.subs / "pipeline_final.csv"
            to_csv(final_test_df, final_sub_csv, P.test_images_csv)
            rep_final = validate(final_sub_csv, P.test_images_csv)
            print(f"[submission] Saved & validated {final_sub_csv}: {rep_final}")


if __name__ == "__main__":
    main()

