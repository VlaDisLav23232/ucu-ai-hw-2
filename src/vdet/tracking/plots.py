"""Experiment diagnostic and report plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


FAMILY_COLORS = {
    "faster_rcnn": "#4C72B0",
    "retinanet": "#DD8452",
    "yolo": "#55A868",
    "rtdetr": "#C44E52",
    "pipeline": "#8172B3",
    "temporal": "#937860",
}
FAMILY_LABELS = {
    "faster_rcnn": "Faster R-CNN (two-stage, anchors)",
    "retinanet": "RetinaNet (one-stage, anchors)",
    "yolo": "YOLO11 (anchor-free)",
    "rtdetr": "RT-DETR-L (query-based)",
    "pipeline": "ensemble pipeline",
    "temporal": "temporal post-processing",
}


def _named_runs(runs) -> list[tuple[str, pd.DataFrame]]:
    if isinstance(runs, dict):
        return [(label, _load_metrics_from_item(val)[1]) for label, val in runs.items()]
    return [_load_metrics_from_item(val) for val in runs]


def curves(
    runs: Sequence[str | Path | pd.DataFrame] | dict[str, str | Path | pd.DataFrame],
    out_path: str | Path,
    title: str = "Training loss and proxy validation mAP",
) -> Path:
    """Plot training loss and proxy validation AP vs epoch and vs GPU-hours (`curves_<family>.png`)."""
    named_runs = _named_runs(runs)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.5))
    colors = plt.get_cmap("tab10").colors

    for i, (label, df) in enumerate(named_runs):
        if df.empty:
            continue
        kw = dict(marker="o", markersize=3.5, linewidth=1.5, color=colors[i % 10], label=label)
        axes[0].plot(df["epoch"], df["train_loss"], **kw)
        axes[1].plot(df["epoch"], df["val_AP"], **kw)
        axes[2].plot(df["gpu_time_h"], df["val_AP"], **kw)

    for ax, t, xl, yl in [
        (axes[0], "Train loss vs epoch", "Epoch", "Total train loss"),
        (axes[1], "Proxy val AP vs epoch", "Epoch", "Proxy mAP@[.5:.95]"),
        (axes[2], "Proxy val AP vs GPU-hours", "Cumulative GPU-hours", "Proxy mAP@[.5:.95]"),
    ]:
        ax.set_title(t, fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    if named_runs:
        axes[2].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def map_vs_gpu_time(
    runs: Sequence[str | Path | pd.DataFrame] | dict[str, str | Path | pd.DataFrame],
    out_path: str | Path,
    title: str = "Proxy validation mAP vs training budget (same 1.5 GPU-h budget)",
) -> Path:
    """Plot proxy mAP over cumulative GPU-hours, CNN (YOLO) vs query-based (RT-DETR) (`map_vs_gpu_time_all.png`)."""
    named_runs = _named_runs(runs)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    greens = ["#1b7837", "#5aae61", "#a6dba0"]
    reds = ["#b2182b", "#ef8a62", "#f4a582"]
    gi = ri = 0
    for label, df in named_runs:
        if df.empty:
            continue
        if label.startswith("rtdetr"):
            c, ls, ri = reds[ri % 3], "-", ri + 1
        else:
            c, ls, gi = greens[gi % 3], "--", gi + 1
        ax.plot(df["gpu_time_h"], df["val_AP"], marker="o", markersize=4, linewidth=1.8,
                color=c, linestyle=ls, label=label)

    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("Cumulative training time (GPU-hours, RTX 4050 Laptop)")
    ax.set_ylabel("Proxy val mAP@[0.50:0.95]")
    ax.grid(True, alpha=0.3)
    if named_runs:
        ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                  title="YOLO11 (CNN) dashed\nRT-DETR solid", title_fontsize=8.5)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def size_breakdown(
    results_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Validation AP by object size (COCO buckets)",
) -> Path:
    """Grouped bars of AP_small / AP_medium / AP_large per run (`size_breakdown.png`)."""
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    labels = results_df["run_id"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.27
    cols = [("APs", "AP_small (<32²)", "#9ecae1"), ("APm", "AP_medium (32²–96²)", "#4292c6"),
            ("APl", "AP_large (>96²)", "#08519c")]
    for j, (c, lbl, colr) in enumerate(cols):
        vals = pd.to_numeric(results_df[c], errors="coerce").fillna(0.0).to_numpy()
        bars = ax.bar(x + (j - 1) * width, vals, width, label=lbl, color=colr)
        if c == "APs":
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)

    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("COCO AP@[.5:.95]")
    ax.set_ylim(0, pd.to_numeric(results_df["APl"], errors="coerce").max() * 1.12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def _place_labels(ax, xs, ys, labels, fontsize=8):
    """Annotate points, nudging labels downwards until they do not overlap earlier ones."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    order = np.argsort(-np.asarray(ys))
    for i in order:
        dy = 3
        for _ in range(40):
            t = ax.annotate(labels[i], (xs[i], ys[i]), textcoords="offset points", xytext=(6, dy),
                            fontsize=fontsize, va="center")
            bb = t.get_window_extent(renderer).expanded(1.02, 1.1)
            if not any(bb.overlaps(o) for o in placed):
                placed.append(bb)
                break
            t.remove()
            dy -= 4
        else:
            placed.append(bb)


def map_vs_fps(
    results_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Accuracy vs speed (val mAP vs inference FPS)",
) -> Path:
    """Scatter of val AP vs FPS, coloured by detector family (`map_vs_fps.png`)."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fps = pd.to_numeric(results_df["fps"], errors="coerce").to_numpy()
    ap = pd.to_numeric(results_df["AP"], errors="coerce").to_numpy()
    fams = results_df["family"].astype(str).tolist()
    labels = results_df["run_id"].astype(str).tolist()
    for fam in dict.fromkeys(fams):
        m = np.array([f == fam for f in fams])
        ax.scatter(fps[m], ap[m], s=55, c=FAMILY_COLORS.get(fam, "grey"), edgecolors="black",
                   linewidths=0.6, zorder=3, label=FAMILY_LABELS.get(fam, fam))
    ax.set_xlim(0, np.nanmax(fps) * 1.25)
    _place_labels(ax, fps, ap, labels)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Inference speed (FPS, batch 1, FP16, RTX 4050 Laptop; run's own input size)")
    ax.set_ylabel("Val mAP@[0.50:0.95]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def per_class_ap(
    results_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Per-class validation AP",
) -> Path:
    """Heatmap (runs x classes) of per-class AP@[.5:.95] (`per_class_ap.png`)."""
    mat = np.array([[float(getattr(r, f"ap_{c}", np.nan) or 0.0) for c in CLASS_NAMES]
                    for r in results_df.itertuples(index=False)])
    runs = results_df["run_id"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(runs) + 1.4))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(0.5, float(np.nanmax(mat))), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.3 else "black")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(runs, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=11, pad=24)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="AP@[.5:.95]")
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def pipeline_waterfall(
    steps: list[tuple[str, float]],
    out_path: str | Path,
    title: str = "From single models to the final pipeline (val mAP@[0.50:0.95])",
) -> Path:
    """Plot step-by-step waterfall bar chart of holdout AP gains (`pipeline_waterfall.png`)."""
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    labels = [s[0] for s in steps]
    values = [float(s[1]) for s in steps]
    x = np.arange(len(labels))

    colors = ["#457b9d" if i < 3 else "#2a9d8f" for i in range(len(labels))]  # single models | pipeline
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
    ax.set_ylabel("Val mAP@[0.50:0.95]")
    if values:
        ax.set_ylim(0.0, max(values) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)

