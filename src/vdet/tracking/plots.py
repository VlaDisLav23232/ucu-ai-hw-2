"""Experiment diagnostic and report plotting utilities (§9.4, U0.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vdet.constants import CLASS_NAMES
from vdet.viz import save_figure_under_budget

matplotlib.use("Agg")


def _load_metrics_from_item(item: str | Path | pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Resolve a run dir, metrics.csv path, or DataFrame into (label, DataFrame)."""
    if isinstance(item, pd.DataFrame):
        return "run", item
    p = Path(item)
    if p.is_dir():
        return p.name, pd.read_csv(p / "metrics.csv")
    return p.parent.name, pd.read_csv(p)


def curves(
    runs: Sequence[str | Path | pd.DataFrame] | dict[str, str | Path | pd.DataFrame],
    out_path: str | Path,
    title: str = "Training Loss & Proxy Validation mAP",
) -> Path:
    """Plot training loss and proxy validation AP vs Epoch and vs GPU-hours (`curves_<family>.png`)."""
    if isinstance(runs, dict):
        named_runs = [(label, _load_metrics_from_item(val)[1]) for label, val in runs.items()]
    else:
        named_runs = [_load_metrics_from_item(val) for val in runs]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    for label, df in named_runs:
        if df.empty:
            continue
        ep = df["epoch"].to_numpy(dtype=np.float64)
        gpu_h = df["gpu_time_h"].to_numpy(dtype=np.float64)
        t_loss = df["train_loss"].to_numpy(dtype=np.float64)
        val_ap = df["val_AP"].to_numpy(dtype=np.float64)

        axes[0].plot(ep, t_loss, marker="o", markersize=3, linewidth=1.5, label=label)
        axes[1].plot(ep, val_ap, marker="o", markersize=3, linewidth=1.5, label=label)
        axes[2].plot(gpu_h, val_ap, marker="o", markersize=3, linewidth=1.5, label=label)

    axes[0].set_title("Train Loss vs Epoch", fontsize=9)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Train Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Proxy Val AP vs Epoch", fontsize=9)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Proxy mAP@[.5:.95]")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Proxy Val AP vs GPU-Hours", fontsize=9)
    axes[2].set_xlabel("Cumulative GPU-Hours")
    axes[2].set_ylabel("Proxy mAP@[.5:.95]")
    axes[2].grid(True, alpha=0.3)

    if named_runs:
        axes[2].legend(fontsize=8, loc="lower right")

    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def map_vs_gpu_time(
    runs: Sequence[str | Path | pd.DataFrame] | dict[str, str | Path | pd.DataFrame],
    out_path: str | Path,
    title: str = "Validation mAP vs Compute Budget (GPU-Hours)",
) -> Path:
    """Plot proxy mAP over cumulative GPU-hours across model families (`map_vs_gpu_time_all.png`)."""
    if isinstance(runs, dict):
        named_runs = [(label, _load_metrics_from_item(val)[1]) for label, val in runs.items()]
    else:
        named_runs = [_load_metrics_from_item(val) for val in runs]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for label, df in named_runs:
        if df.empty:
            continue
        ax.plot(
            df["gpu_time_h"].to_numpy(dtype=np.float64),
            df["val_AP"].to_numpy(dtype=np.float64),
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=label,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Cumulative Training Time (GPU-Hours)")
    ax.set_ylabel("Validation mAP@[0.50:0.95]")
    ax.grid(True, alpha=0.3)
    if named_runs:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def size_breakdown(
    results_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Holdout AP by Object Size Bucket (AP_s / AP_m / AP_l)",
) -> Path:
    """Plot grouped bar chart of APs, APm, and APl per model run (`size_breakdown.png`)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = results_df["run_id"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.25

    aps = pd.to_numeric(results_df["APs"], errors="coerce").fillna(0.0).to_numpy()
    apm = pd.to_numeric(results_df["APm"], errors="coerce").fillna(0.0).to_numpy()
    apl = pd.to_numeric(results_df["APl"], errors="coerce").fillna(0.0).to_numpy()

    ax.bar(x - width, aps, width, label="AP_small (<32²)")
    ax.bar(x, apm, width, label="AP_medium (32²–96²)")
    ax.bar(x + width, apl, width, label="AP_large (>96²)")

    ax.set_title(title, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("COCO AP")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def map_vs_fps(
    results_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Accuracy vs Speed Trade-off (Holdout mAP vs T4 FPS)",
) -> Path:
    """Scatter plot of holdout AP vs inference FPS (`map_vs_fps.png`)."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fps = pd.to_numeric(results_df["fps"], errors="coerce").fillna(0.0).to_numpy()
    ap = pd.to_numeric(results_df["AP"], errors="coerce").fillna(0.0).to_numpy()
    labels = results_df["run_id"].astype(str).tolist()

    ax.scatter(fps, ap, s=55, c="#2a9d8f", edgecolors="black", zorder=3)
    for x_val, y_val, lbl in zip(fps, ap, labels):
        ax.annotate(lbl, (x_val, y_val), textcoords="offset points", xytext=(5, 4), fontsize=8)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Inference Speed (FPS, batch=1, FP16 @ 1280)")
    ax.set_ylabel("Holdout mAP@[0.50:0.95]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def per_class_ap(
    results_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Per-Class Holdout AP Across Models",
) -> Path:
    """Grouped bar chart of per-class AP (`per_class_ap.png`)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    n_runs = max(1, len(results_df))
    x = np.arange(len(CLASS_NAMES))
    width = min(0.22, 0.8 / n_runs)

    for idx, row in enumerate(results_df.itertuples(index=False)):
        vals = [float(getattr(row, f"ap_{c}", 0.0) or 0.0) for c in CLASS_NAMES]
        offset = (idx - (n_runs - 1) / 2.0) * width
        ax.bar(x + offset, vals, width, label=str(row.run_id))

    ax.set_title(title, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_ylabel("AP@[0.50:0.95]")
    ax.grid(True, axis="y", alpha=0.3)
    if len(results_df) > 0:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def pipeline_waterfall(
    steps: list[tuple[str, float]],
    out_path: str | Path,
    title: str = "M4 Pipeline Progression (Holdout Val mAP@[0.50:0.95])",
) -> Path:
    """Plot step-by-step waterfall bar chart of holdout AP gains (`pipeline_waterfall.png`)."""
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    labels = [s[0] for s in steps]
    values = [float(s[1]) for s in steps]
    x = np.arange(len(labels))

    colors = ["#457b9d" if i < 3 else "#2a9d8f" for i in range(len(labels))]
    bars = ax.bar(x, values, width=0.55, color=colors, edgecolor="black", alpha=0.9)

    for idx, (bar, val) in enumerate(zip(bars, values)):
        delta_str = ""
        if idx > 0:
            diff = val - values[idx - 1]
            sign = "+" if diff >= 0 else ""
            delta_str = f"\n({sign}{diff:.4f})"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.003,
            f"{val:.4f}{delta_str}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Official Val mAP@[0.50:0.95]")
    if values:
        ax.set_ylim(0.0, max(values) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)

