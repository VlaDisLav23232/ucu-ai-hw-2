"""Convert Ultralytics results.csv and artifacts into the local RunLogger format."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from vdet.tracking.run import METRICS_COLUMNS
from vdet.utils import deep_merge, load_yaml, save_yaml


def convert_ultralytics_results_csv(
    ultra_csv_path: str | Path,
    out_metrics_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Convert an Ultralytics `results.csv` into our standardized `metrics.csv` schema.

    Handles both single-column `time` (cumulative seconds in Ultralytics >=8.1 or per-epoch seconds)
    and standard loss/metric headers with stripped whitespace.
    """
    raw_df = pd.read_csv(Path(ultra_csv_path))
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    n = len(raw_df)
    if n == 0:
        out_df = pd.DataFrame(columns=list(METRICS_COLUMNS))
        if out_metrics_csv_path is not None:
            out_df.to_csv(out_metrics_csv_path, index=False)
        return out_df

    epochs = raw_df["epoch"].to_numpy(dtype=np.int64) if "epoch" in raw_df.columns else np.arange(1, n + 1)

    if "time" in raw_df.columns:
        t_vals = raw_df["time"].to_numpy(dtype=np.float64)
        # In Ultralytics 8.x, `time` is cumulative seconds if strictly non-decreasing, else per-epoch seconds
        if len(t_vals) > 1 and np.all(np.diff(t_vals) >= 0):
            cum_sec = t_vals
        else:
            cum_sec = np.cumsum(t_vals)
        gpu_time_h = np.round(cum_sec / 3600.0, 5)
    else:
        gpu_time_h = np.zeros(n, dtype=np.float64)

    def _col(candidates: list[str]) -> np.ndarray:
        for name in candidates:
            if name in raw_df.columns:
                return raw_df[name].to_numpy(dtype=np.float64)
        return np.zeros(n, dtype=np.float64)

    loss_box = _col(["train/box_loss", "train/giou_loss", "box_loss"])
    loss_cls = _col(["train/cls_loss", "cls_loss"])
    loss_dfl = _col(["train/dfl_loss", "train/l1_loss", "dfl_loss"])
    train_loss = np.round(loss_box + loss_cls + loss_dfl, 5)
    lr = _col(["lr/pg0", "lr/pg1", "lr"])
    val_ap = _col(["metrics/mAP50-95(B)", "metrics/mAP50-95"])
    val_ap50 = _col(["metrics/mAP50(B)", "metrics/mAP50"])
    val_aps = _col(["metrics/mAPs(B)", "val_APs"])

    out_df = pd.DataFrame(
        {
            "epoch": epochs,
            "gpu_time_h": gpu_time_h,
            "lr": np.round(lr, 8),
            "train_loss": train_loss,
            "loss_box": np.round(loss_box, 5),
            "loss_cls": np.round(loss_cls, 5),
            "loss_dfl": np.round(loss_dfl, 5),
            "val_AP": np.round(val_ap, 5),
            "val_AP50": np.round(val_ap50, 5),
            "val_APs": np.round(val_aps, 5),
        }
    )

    if out_metrics_csv_path is not None:
        out_path = Path(out_metrics_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

    return out_df


def adapt_ultralytics_run(
    ultra_dir: str | Path,
    run_dir: str | Path,
) -> pd.DataFrame:
    """Copy and convert Ultralytics run artifacts (`results.csv`, `args.yaml`, small plots) into `run_dir`."""
    u_dir = Path(ultra_dir)
    r_dir = Path(run_dir)
    r_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = r_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results_csv = u_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Ultralytics results.csv not found in {u_dir}")

    metrics_df = convert_ultralytics_results_csv(results_csv, r_dir / "metrics.csv")

    args_yaml = u_dir / "args.yaml"
    if args_yaml.exists():
        cfg_path = r_dir / "config.yaml"
        existing = load_yaml(cfg_path) if cfg_path.exists() else {}
        ultra_args = load_yaml(args_yaml)
        merged = deep_merge(existing, {"ultralytics_args": ultra_args})
        save_yaml(merged, cfg_path)

    for plot_name in ("PR_curve.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
        src_plot = u_dir / plot_name
        if src_plot.exists() and src_plot.stat().st_size <= 300 * 1024:
            shutil.copy2(src_plot, plots_dir / plot_name)

    return metrics_df

