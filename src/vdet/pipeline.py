"""End-to-end inference pipeline: per-model (TTA / tiled) predictions, WBF ensembling, temporal tracklet
rescoring + gap propagation, waterfall plot and final submissions.

The pipeline config is a dict defined in notebook 06.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vdet.constants import DEFAULT_SEED, MAX_SUBMISSION_DETS
from vdet.data.manifest import load_manifest
from vdet.eval.coco_eval import evaluate
from vdet.infer.fusion import (
    DEFAULT_CLASS_IOU_THR,
    DEFAULT_CLASS_SCORE_MULTS,
    fuse_predictions,
)
from vdet.infer.predictors import (
    Predictor,
    load_predictions_parquet,
    save_predictions_parquet,
)
from vdet.infer.tiling import TiledPredictor
from vdet.infer.tta import TTAPredictor
from vdet.models.tv_builders import TorchvisionPredictor
from vdet.models.ultra import UltralyticsPredictor
from vdet.paths import P
from vdet.submission import to_csv, validate
from vdet.temporal.rescoring import apply_temporal_rescoring
from vdet.tracking.aggregate import aggregate_runs
from vdet.tracking.plots import pipeline_waterfall
from vdet.tracking.run import RunLogger
from vdet.utils import load_json, load_yaml, save_json, timer


def ensure_split_predictions(
    model_spec: dict[str, Any],
    split: str,
    manifest_df: pd.DataFrame,
) -> Path | None:
    """Return path to `runs/<run_id>/preds_<split>.parquet`, generating it from checkpoint (with optional TTA/Tiling)."""
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
    use_tta = bool(model_spec.get("tta", False))
    use_tiling = bool(model_spec.get("tiling", False))

    print(
        f"[pipeline] Generating {split} predictions for {rid} from {w_path.name} "
        f"(imgsz={imgsz}, tta={use_tta}, tiling={use_tiling})..."
    )
    if w_path.suffix.lower() in {".pth", ".ckpt"}:
        raw_cfg = model_spec.get("config")
        if isinstance(raw_cfg, dict):
            cfg = raw_cfg
        else:
            cfg_p = P.root / str(raw_cfg) if raw_cfg else None
            cfg = load_yaml(cfg_p) if (cfg_p and cfg_p.exists()) else None
        base_predictor: Predictor = TorchvisionPredictor(
            weights_path=w_path,
            config=cfg,
            conf=0.001,
            max_det=MAX_SUBMISSION_DETS,
            batch_size=batch,
            source_tag=rid,
        )
    else:
        base_predictor = UltralyticsPredictor(
            weights_path=w_path,
            imgsz=imgsz,
            conf=0.001,
            iou=0.6,
            max_det=MAX_SUBMISSION_DETS,
            batch_size=batch,
            source_tag=rid,
        )

    # Optionally reuse already-computed base (unflipped full-frame) predictions to save 33-50% time
    base_rid = str(model_spec.get("base_run_id", ""))
    base_df: pd.DataFrame | None = None
    if base_rid:
        base_p = P.runs / base_rid / f"preds_{split}.parquet"
        if base_p.exists():
            base_df = load_predictions_parquet(base_p)

    with timer(f"predict ({rid}, {split}, n={len(sub_m)})"):
        if use_tiling:
            tiled_pred = TiledPredictor(
                base_predictor=base_predictor,
                slice_size=int(model_spec.get("slice_size", 960)),
                overlap_ratio=float(model_spec.get("overlap", 0.25)),
                upscale_small_frames=bool(model_spec.get("upscale_small_frames", True)),
                conf_thresh=0.001,
                max_det=MAX_SUBMISSION_DETS,
                source_tag=rid,
            )
            df = tiled_pred.predict(sub_m, data_root=P.data, full_preds_df=base_df)
        elif use_tta:
            scales = model_spec.get("tta_scales", [1.0, 1.20])
            tta_pred = TTAPredictor(
                base_predictor=base_predictor,
                scales=scales,
                hflip=bool(model_spec.get("tta_hflip", True)),
                hflip_only_at_base_scale=bool(model_spec.get("hflip_only_at_base_scale", True)),
                conf_thresh=0.001,
                max_det=MAX_SUBMISSION_DETS,
                source_tag=rid,
            )
            df = tta_pred.predict(sub_m, data_root=P.data, base_preds_df=base_df)
        else:
            df = base_predictor.predict(sub_m, data_root=P.data)

    save_predictions_parquet(df, pred_path, conf_thresh=0.001, max_det_per_image=MAX_SUBMISSION_DETS)
    return pred_path


def _log_pipeline_stage(
    run_id: str,
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
        family=family,
        exp=exp,
    )
    save_predictions_parquet(
        preds_val_df,
        logger.run_dir / "preds_val_slim.parquet",
        conf_thresh=0.05,
    )
    return logger.finalize(eval_metrics=eval_metrics, checkpoint_uri=ckpt_uri)


def run_pipeline(
    cfg: dict[str, Any],
    predict_test: bool = False,
    force_recompute: bool = False,
    overwrite_submissions: bool = False,
) -> dict[str, Any]:
    """Run the pipeline described by `cfg` (models, fusion, temporal).

    Every stage reuses its parquet output if it already exists (unless `force_recompute`).
    Submission CSVs are only (re)written if missing or if `overwrite_submissions` is set.
    Returns a dict with the stage metrics (as stored in experiments/runs/<id>/summary.json).
    """
    manifest_df = load_manifest()

    pipe_run_id = str(cfg.get("run_id", "pipeline-final"))
    wbf_run_id = str(cfg.get("wbf_run_id", "ensemble-wbf"))
    rescore_run_id = str(cfg.get("rescore_run_id", "temporal-rescore"))
    wbf_sub_name = str(cfg.get("wbf_sub_name", "ensemble_wbf.csv"))
    final_sub_name = str(cfg.get("final_sub_name", "pipeline_final.csv"))

    models_cfg = cfg.get("models", [])
    fus_cfg = cfg.get("fusion", {})
    tmp_cfg = cfg.get("temporal", {})

    # 1. Collect all available component validation predictions
    val_dfs: list[pd.DataFrame] = []
    val_weights: list[float] = []
    active_models: list[dict[str, Any]] = []

    for m_spec in models_cfg:
        val_p = ensure_split_predictions(m_spec, split="val", manifest_df=manifest_df)
        if val_p is not None and val_p.exists():
            val_dfs.append(load_predictions_parquet(val_p))
            val_weights.append(float(m_spec.get("wbf_weight", 1.0)))
            active_models.append(m_spec)

    if len(val_dfs) < 2:
        raise RuntimeError(f"Need at least 2 models with val predictions for ensembling; found {len(val_dfs)}.")

    active_ids = [str(m["run_id"]) for m in active_models]
    print(f"[pipeline] Active ensemble models ({len(active_ids)}): {active_ids} with weights {val_weights}")

    out_dir = P.runs / pipe_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    wbf_val_path = out_dir / "preds_val_wbf.parquet"
    rescore_only_val_path = out_dir / "preds_val_temporal_rescore.parquet"
    final_val_path = out_dir / "preds_val_final.parquet"
    best_params_path = out_dir / "best_pipeline_params.json"

    use_class_wbf = bool(fus_cfg.get("use_class_wbf", False))
    use_smooth_boxes = bool(tmp_cfg.get("smooth_boxes", False))
    use_velocity = bool(tmp_cfg.get("use_velocity", True))
    small_box_bonus = float(tmp_cfg.get("small_box_bonus", 0.06))

    if (
        not force_recompute
        and wbf_val_path.exists()
        and rescore_only_val_path.exists()
        and final_val_path.exists()
        and (P.exp_runs_dir / pipe_run_id / "summary.json").exists()
        and (P.images / "pipeline_waterfall.png").exists()
    ):
        print(f"[pipeline] Validation stages ({wbf_run_id}, {rescore_run_id}, {pipe_run_id}) already completed; reusing results.")
        if best_params_path.exists():
            bp = load_json(best_params_path)
            use_class_wbf = bool(bp.get("use_class_wbf", use_class_wbf))
            use_smooth_boxes = bool(bp.get("use_smooth_boxes", use_smooth_boxes))
    else:
        if not force_recompute and wbf_val_path.exists():
            print(f"[pipeline] Reusing existing WBF val predictions from {wbf_val_path}")
            wbf_val_df = load_predictions_parquet(wbf_val_path)
            wbf_eval = evaluate(wbf_val_df, split="val", per_res_bucket=True)
        else:
            with timer(f"WBF fusion (val, {len(val_dfs)} models)"):
                wbf_uni_df = fuse_predictions(
                    preds_list=val_dfs,
                    weights=val_weights,
                    method=str(fus_cfg.get("method", "wbf")),
                    iou_thr=float(fus_cfg.get("iou_thr", 0.55)),
                    skip_box_thr=float(fus_cfg.get("skip_box_thr", 0.001)),
                    conf_type=str(fus_cfg.get("conf_type", "avg")),
                    max_det=int(fus_cfg.get("max_det", MAX_SUBMISSION_DETS)),
                    manifest_df=manifest_df,
                    source_tag=wbf_run_id,
                )
            eval_uni = evaluate(wbf_uni_df, split="val", per_res_bucket=True)
            wbf_val_df = wbf_uni_df
            wbf_eval = eval_uni
            use_class_wbf = False

            if bool(fus_cfg.get("auto_select_class_wbf", True)):
                with timer(f"Class-aware WBF fusion (val, {len(val_dfs)} models)"):
                    wbf_cls_df = fuse_predictions(
                        preds_list=val_dfs,
                        weights=val_weights,
                        method=str(fus_cfg.get("method", "wbf")),
                        iou_thr=float(fus_cfg.get("iou_thr", 0.55)),
                        class_iou_thr=DEFAULT_CLASS_IOU_THR,
                        class_score_mults=DEFAULT_CLASS_SCORE_MULTS,
                        skip_box_thr=float(fus_cfg.get("skip_box_thr", 0.001)),
                        conf_type=str(fus_cfg.get("conf_type", "avg")),
                        max_det=int(fus_cfg.get("max_det", MAX_SUBMISSION_DETS)),
                        manifest_df=manifest_df,
                        source_tag=wbf_run_id,
                    )
                eval_cls = evaluate(wbf_cls_df, split="val", per_res_bucket=True)
                print(
                    f"  -> [WBF Comparison] Uniform IoU AP={eval_uni['AP']:.4f} vs Class-Aware WBF AP={eval_cls['AP']:.4f}"
                )
                if float(eval_cls["AP"]) > float(eval_uni["AP"]):
                    wbf_val_df = wbf_cls_df
                    wbf_eval = eval_cls
                    use_class_wbf = True

            save_predictions_parquet(wbf_val_df, wbf_val_path, conf_thresh=0.001)

        _log_pipeline_stage(
            run_id=wbf_run_id,
            family="pipeline",
            exp="wbf",
            parent_run="rtdetr-e0",
            hypothesis=f"Cross-family WBF ensemble of {active_ids} (weights={val_weights}, class_wbf={use_class_wbf}).",
            config={"models": active_ids, "weights": val_weights, "use_class_wbf": use_class_wbf, **fus_cfg},
            preds_val_df=wbf_val_df,
            eval_metrics=wbf_eval,
            ckpt_uri=wbf_val_path.as_posix(),
        )
        print(
            f"  -> [WBF ensemble ({wbf_run_id})] Val AP={wbf_eval['AP']:.4f} | AP50={wbf_eval['AP50']:.4f} | "
            f"APs={wbf_eval['APs']:.4f} | AR500={wbf_eval['AR500']:.4f}"
        )

        # 3. Temporal ablation arm 1: Tracklet Rescoring ONLY (no gap propagation) on val
        with timer("Temporal rescoring only (val)"):
            rescore_only_val_df = apply_temporal_rescoring(
                preds_df=wbf_val_df,
                manifest_df=manifest_df,
                iou_link=float(tmp_cfg.get("iou_link", 0.45)),
                alpha=float(tmp_cfg.get("alpha", 0.70)),
                topk_ratio=float(tmp_cfg.get("topk_ratio", 0.50)),
                single_frame_decay=float(tmp_cfg.get("single_frame_decay", 0.96)),
                enable_propagation=False,
                use_velocity=use_velocity,
                small_box_bonus=small_box_bonus,
                smooth_boxes=False,
                source_tag=rescore_run_id,
            )
        save_predictions_parquet(rescore_only_val_df, rescore_only_val_path, conf_thresh=0.001)
        rescore_eval = evaluate(rescore_only_val_df, split="val", per_res_bucket=True)
        _log_pipeline_stage(
            run_id=rescore_run_id,
            family="pipeline",
            exp="rescore",
            parent_run=wbf_run_id,
            hypothesis="Tracklet velocity+IoU linking + top-k score smoothing (without gap propagation) on WBF ensemble.",
            config={**tmp_cfg, "enable_propagation": False},
            preds_val_df=rescore_only_val_df,
            eval_metrics=rescore_eval,
            ckpt_uri=rescore_only_val_path.as_posix(),
        )
        print(
            f"  -> [Temporal rescore only ({rescore_run_id})] Val AP={rescore_eval['AP']:.4f} | AP50={rescore_eval['AP50']:.4f} | "
            f"APs={rescore_eval['APs']:.4f} | AR500={rescore_eval['AR500']:.4f}"
        )

        # 4. Final pipeline: Tracklet Rescoring + Linear Gap Propagation on val
        with timer("Temporal rescoring + gap propagation (val)"):
            final_nobs_df = apply_temporal_rescoring(
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
                use_velocity=use_velocity,
                small_box_bonus=small_box_bonus,
                smooth_boxes=False,
                source_tag=pipe_run_id,
            )
            eval_nobs = evaluate(final_nobs_df, split="val", per_res_bucket=True)

            final_bs_df = apply_temporal_rescoring(
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
                use_velocity=use_velocity,
                small_box_bonus=small_box_bonus,
                smooth_boxes=True,
                box_smooth_weight=float(tmp_cfg.get("box_smooth_weight", 0.16)),
                source_tag=pipe_run_id,
            )
            eval_bs = evaluate(final_bs_df, split="val", per_res_bucket=True)
            print(
                f"  -> [Box Smoothing Comparison] No Box Smooth AP={eval_nobs['AP']:.4f} vs 3-Frame Box Smooth AP={eval_bs['AP']:.4f}"
            )
            if float(eval_bs["AP"]) >= float(eval_nobs["AP"]):
                final_val_df = final_bs_df
                final_eval = eval_bs
                use_smooth_boxes = True
            else:
                final_val_df = final_nobs_df
                final_eval = eval_nobs
                use_smooth_boxes = False

        save_predictions_parquet(final_val_df, final_val_path, conf_thresh=0.001)
        save_json(
            {"use_class_wbf": use_class_wbf, "use_smooth_boxes": use_smooth_boxes},
            best_params_path,
        )
        _log_pipeline_stage(
            run_id=pipe_run_id,
            family="pipeline",
            exp="final",
            parent_run=rescore_run_id,
            hypothesis=str(cfg.get("hypothesis", "WBF ensemble + temporal tracklet rescoring & gap propagation.")),
            config={**cfg, "use_class_wbf": use_class_wbf, "use_smooth_boxes": use_smooth_boxes},
            preds_val_df=final_val_df,
            eval_metrics=final_eval,
            ckpt_uri=final_val_path.as_posix(),
        )
        print(
            f"  -> [Pipeline final ({pipe_run_id})] Val AP={final_eval['AP']:.4f} | AP50={final_eval['AP50']:.4f} | "
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
                use_velocity=use_velocity,
                small_box_bonus=small_box_bonus,
                source_tag="yolo-e1-temporal",
            )
            ye1_temp_eval = evaluate(ye1_temp_df, split="val", per_res_bucket=False)
            _log_pipeline_stage(
                run_id="yolo-e1-temporal",
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
            (wbf_run_id, "+ Multi-Model\nWBF Ensemble"),
            (rescore_run_id, "+ Tracklet\nScore Smoothing"),
            (pipe_run_id, "+ Gap\nPropagation"),
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
    if predict_test:
        test_dfs: list[pd.DataFrame] = []
        test_weights: list[float] = []
        for m_spec in active_models:
            test_p = ensure_split_predictions(m_spec, split="test", manifest_df=manifest_df)
            if test_p is not None and test_p.exists():
                test_dfs.append(load_predictions_parquet(test_p))
                test_weights.append(float(m_spec.get("wbf_weight", 1.0)))

        if len(test_dfs) >= 2:
            wbf_test_path = out_dir / "preds_test_wbf.parquet"
            if not force_recompute and wbf_test_path.exists():
                print(f"[pipeline] Reusing existing WBF test predictions from {wbf_test_path}")
                wbf_test_df = load_predictions_parquet(wbf_test_path)
            else:
                with timer(f"WBF fusion (test, {len(test_dfs)} models, class_wbf={use_class_wbf})"):
                    wbf_test_df = fuse_predictions(
                        preds_list=test_dfs,
                        weights=test_weights,
                        method=str(fus_cfg.get("method", "wbf")),
                        iou_thr=float(fus_cfg.get("iou_thr", 0.55)),
                        class_iou_thr=DEFAULT_CLASS_IOU_THR if use_class_wbf else None,
                        class_score_mults=DEFAULT_CLASS_SCORE_MULTS if use_class_wbf else None,
                        skip_box_thr=float(fus_cfg.get("skip_box_thr", 0.001)),
                        conf_type=str(fus_cfg.get("conf_type", "avg")),
                        max_det=int(fus_cfg.get("max_det", MAX_SUBMISSION_DETS)),
                        manifest_df=manifest_df,
                        source_tag=wbf_run_id,
                    )
                save_predictions_parquet(wbf_test_df, wbf_test_path, conf_thresh=0.001)

            wbf_sub_csv = P.subs / wbf_sub_name
            if overwrite_submissions or not wbf_sub_csv.exists():
                to_csv(wbf_test_df, wbf_sub_csv, P.test_images_csv)
            rep_wbf = validate(wbf_sub_csv, P.test_images_csv)
            print(f"[submission] Saved & validated {wbf_sub_csv}: {rep_wbf}")

            final_test_path = out_dir / "preds_test_final.parquet"
            if not force_recompute and final_test_path.exists():
                print(f"[pipeline] Reusing existing final test predictions from {final_test_path}")
                final_test_df = load_predictions_parquet(final_test_path)
            else:
                with timer(f"Temporal rescoring + gap propagation (test, smooth_boxes={use_smooth_boxes})"):
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
                        use_velocity=use_velocity,
                        small_box_bonus=small_box_bonus,
                        smooth_boxes=use_smooth_boxes,
                        box_smooth_weight=float(tmp_cfg.get("box_smooth_weight", 0.16)),
                        source_tag=pipe_run_id,
                    )
                save_predictions_parquet(final_test_df, final_test_path, conf_thresh=0.001)

            final_sub_csv = P.subs / final_sub_name
            if overwrite_submissions or not final_sub_csv.exists():
                to_csv(final_test_df, final_sub_csv, P.test_images_csv)
            rep_final = validate(final_sub_csv, P.test_images_csv)
            print(f"[submission] Saved & validated {final_sub_csv}: {rep_final}")


    stages: dict[str, Any] = {}
    for rid in (wbf_run_id, rescore_run_id, pipe_run_id):
        s_path = P.exp_runs_dir / rid / "summary.json"
        if s_path.exists():
            stages[rid] = load_json(s_path)
    return {"active_models": active_ids, "use_class_wbf": use_class_wbf, "use_smooth_boxes": use_smooth_boxes, "stages": stages}


_TEMPORAL_KWARGS = {
    "iou_link", "alpha", "topk_ratio", "single_frame_decay", "enable_propagation", "prop_min_score", "prop_min_len",
    "max_gap", "gap_decay", "use_velocity", "small_box_bonus", "smooth_boxes", "box_smooth_weight", "causal",
}


def run_temporal_variant(
    run_id: str,
    input_preds: str | Path,
    parent_run: str,
    temporal: dict[str, Any],
    hypothesis: str = "",
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Apply a temporal post-processing config to saved val predictions, evaluate it and log it as `run_id`.

    The result is cached in runs/<run_id>/preds_val.parquet and experiments/runs/<run_id>/summary.json.
    """
    out_path = P.runs / run_id / "preds_val.parquet"
    summary_path = P.exp_runs_dir / run_id / "summary.json"
    if out_path.exists() and summary_path.exists() and not force_recompute:
        return load_json(summary_path)

    kwargs = {k: v for k, v in temporal.items() if k in _TEMPORAL_KWARGS}
    with timer(f"temporal variant {run_id}"):
        out_df = apply_temporal_rescoring(
            preds_df=load_predictions_parquet(input_preds), manifest_df=load_manifest(), source_tag=run_id, **kwargs
        )
    save_predictions_parquet(out_df, out_path, conf_thresh=0.001)
    metrics = evaluate(out_df, split="val", per_res_bucket=True)
    summary = _log_pipeline_stage(
        run_id=run_id, family="temporal", exp="causal" if kwargs.get("causal") else "bidirectional",
        parent_run=parent_run, hypothesis=hypothesis or f"Temporal post-processing {kwargs} on {parent_run}.",
        config={"input_preds": Path(input_preds).as_posix(), **kwargs},
        preds_val_df=out_df, eval_metrics=metrics, ckpt_uri=out_path.as_posix(),
    )
    aggregate_runs()
    return summary
