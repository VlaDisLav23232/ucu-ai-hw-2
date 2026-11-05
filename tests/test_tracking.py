"""Unit tests for local experiment tracking, Ultralytics adapter, aggregation, and plotting (U0.7)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vdet.tracking.aggregate import RESULTS_COLUMNS, aggregate_runs
from vdet.tracking.plots import curves, map_vs_fps, map_vs_gpu_time, per_class_ap, size_breakdown
from vdet.tracking.run import RunLogger
from vdet.tracking.ultra_adapter import adapt_ultralytics_run


def test_run_logger_aggregate_and_plots(tmp_path: Path) -> None:
    """A dummy run produces a complete run dir, aggregates into results.csv, and renders plots."""
    exp_dir = tmp_path / "experiments"
    runs_root = exp_dir / "runs"
    lb_csv = exp_dir / "lb.csv"
    exp_dir.mkdir(parents=True, exist_ok=True)
    lb_csv.write_text(
        "run_id,submission_file,lb_public,date,notes\n"
        "yolo-e0-20261104-1200,submissions/yolo-e0.csv,0.3512,2026-11-04,baseline\n",
        encoding="utf-8",
    )

    logger = RunLogger(
        run_id="yolo-e0-20261104-1200",
        config={"imgsz": 1280, "k": 5, "cache": "k5_blur_L1920"},
        runs_root=runs_root,
        seed=42,
        hypothesis="YOLO11s baseline under P0",
        unit="U1.3",
        family="yolo",
        exp="e0",
    )

    for ep in range(1, 5):
        logger.log_epoch(
            {
                "epoch": ep,
                "gpu_time_h": round(ep * 0.35, 3),
                "lr": 0.001 * (0.8**ep),
                "train_loss": round(4.5 - ep * 0.4, 4),
                "loss_box": round(2.0 - ep * 0.15, 4),
                "loss_cls": round(1.5 - ep * 0.15, 4),
                "loss_dfl": round(1.0 - ep * 0.10, 4),
                "val_AP": round(0.18 + ep * 0.03, 4),
                "val_AP50": round(0.32 + ep * 0.04, 4),
                "val_APs": round(0.09 + ep * 0.02, 4),
            }
        )

    eval_metrics = {
        "AP": 0.301,
        "AP50": 0.482,
        "AP75": 0.312,
        "APs": 0.175,
        "APm": 0.388,
        "APl": 0.512,
        "AR500": 0.445,
        "ap_person": 0.28,
        "ap_bicycle": 0.19,
        "ap_car": 0.52,
        "ap_van": 0.34,
        "ap_truck": 0.29,
        "ap_tricycle": 0.22,
        "ap_bus": 0.31,
        "ap_motor": 0.26,
    }
    logger.finalize(
        eval_metrics=eval_metrics,
        speed_metrics={"fps": 38.5},
        checkpoint_uri="kaggle://vdet-checkpoints/yolo-e0-best.pt",
        outcome_note="Baseline completed cleanly.",
    )

    for fname in ("config.yaml", "env.json", "metrics.csv", "eval_val.json", "speed.json", "summary.json", "notes.md"):
        assert (logger.run_dir / fname).exists(), f"Missing {fname}"

    results_df = aggregate_runs(experiments_dir=exp_dir)
    assert len(results_df) == 1
    assert list(results_df.columns) == list(RESULTS_COLUMNS)
    assert float(results_df.iloc[0]["AP"]) == 0.301
    assert float(results_df.iloc[0]["lb_public"]) == 0.3512

    # Render plots and verify size budget
    p_curves = curves({"yolo-e0": logger.run_dir}, logger.plots_dir / "curves_yolo.png")
    p_gpu = map_vs_gpu_time({"yolo-e0": logger.run_dir}, logger.plots_dir / "map_vs_gpu_time_all.png")
    p_size = size_breakdown(results_df, logger.plots_dir / "size_breakdown.png")
    p_fps = map_vs_fps(results_df, logger.plots_dir / "map_vs_fps.png")
    p_cls = per_class_ap(results_df, logger.plots_dir / "per_class_ap.png")

    for p in (p_curves, p_gpu, p_size, p_fps, p_cls):
        assert p.exists()
        assert p.stat().st_size <= 300 * 1024


def test_ultralytics_adapter_converts_sample_csv(tmp_path: Path) -> None:
    """Ultralytics results.csv with padded headers converts cleanly to metrics.csv schema."""
    ultra_dir = tmp_path / "ultra_run"
    ultra_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = (
        "                  epoch,                   time,         train/box_loss,         train/cls_loss,         train/dfl_loss,   metrics/precision(B),      metrics/recall(B),       metrics/mAP50(B),    metrics/mAP50-95(B),                 lr/pg0\n"
        "                      1,                  360.0,                 1.4500,                 1.8200,                 1.1200,                 0.5200,                 0.4100,                 0.3800,                 0.2100,                 0.0010\n"
        "                      2,                  720.0,                 1.2500,                 1.4200,                 1.0100,                 0.6100,                 0.4700,                 0.4400,                 0.2600,                 0.0008\n"
    )
    (ultra_dir / "results.csv").write_text(sample_csv, encoding="utf-8")
    (ultra_dir / "args.yaml").write_text("imgsz: 1280\nbatch: 8\n", encoding="utf-8")

    run_dir = tmp_path / "exp_run"
    metrics_df = adapt_ultralytics_run(ultra_dir, run_dir)
    assert len(metrics_df) == 2
    assert list(metrics_df["epoch"]) == [1, 2]
    assert list(metrics_df["gpu_time_h"]) == [0.1, 0.2]
    assert list(metrics_df["val_AP"]) == [0.21, 0.26]
    assert list(metrics_df["val_AP50"]) == [0.38, 0.44]
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "config.yaml").exists()

