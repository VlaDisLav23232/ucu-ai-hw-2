#!/usr/bin/env python3
"""Run or plot the U1.2 Ignore-Policy Ablation ('blur' vs 'drop') and export images/ignore_ablation.png."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vdet.data.manifest import load_ignore
from vdet.models.ultra import train_ultralytics
from vdet.paths import P
from vdet.utils import load_json, load_resolved_config, save_json, timer
from vdet.viz import save_figure_under_budget

matplotlib.use("Agg")


def compute_fp_rate_near_ignore_borders(
    preds_df: pd.DataFrame,
    ignore_df: pd.DataFrame,
    score_thresh: float = 0.25,
    border_margin_px: float = 24.0,
) -> float:
    """Compute share (%) of high-confidence detections falling in the [0, 24px] ring around ignore regions."""
    if preds_df is None or preds_df.empty:
        return 0.0
    sub_p = preds_df[preds_df["score"] >= score_thresh]
    if sub_p.empty:
        return 0.0

    val_ig = ignore_df[ignore_df["split"] == "val"]
    ig_by_key = {
        k: g[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
        for k, g in val_ig.groupby("image_key")
    }

    near_border_count = 0
    total_count = len(sub_p)

    for key, p_group in sub_p.groupby("image_key"):
        ig_arr = ig_by_key.get(str(key))
        if ig_arr is None or len(ig_arr) == 0:
            continue
        p_arr = p_group[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
        pcx = p_arr[:, 0] + 0.5 * p_arr[:, 2]
        pcy = p_arr[:, 1] + 0.5 * p_arr[:, 3]

        ix1 = ig_arr[:, 0][None, :]
        iy1 = ig_arr[:, 1][None, :]
        ix2 = (ig_arr[:, 0] + ig_arr[:, 2])[None, :]
        iy2 = (ig_arr[:, 1] + ig_arr[:, 3])[None, :]

        inside = (pcx[:, None] >= ix1) & (pcx[:, None] <= ix2) & (pcy[:, None] >= iy1) & (pcy[:, None] <= iy2)
        in_expanded = (
            (pcx[:, None] >= ix1 - border_margin_px)
            & (pcx[:, None] <= ix2 + border_margin_px)
            & (pcy[:, None] >= iy1 - border_margin_px)
            & (pcy[:, None] <= iy2 + border_margin_px)
        )
        in_ring = np.any(in_expanded & (~inside), axis=1)
        near_border_count += int(np.sum(in_ring))

    return round(100.0 * near_border_count / max(1, total_count), 3)


def plot_ignore_ablation(
    blur_run_dir: Path,
    drop_run_dir: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Generate `images/ignore_ablation.png` comparing `blur` and `drop` on AP, AP50, APs, and border rate."""
    blur_eval = load_json(blur_run_dir / "eval_val.json")
    drop_eval = load_json(drop_run_dir / "eval_val.json")
    ignore_df = load_ignore()

    blur_preds_path = blur_run_dir / "preds_val_slim.parquet"
    drop_preds_path = drop_run_dir / "preds_val_slim.parquet"
    blur_preds = pd.read_parquet(blur_preds_path) if blur_preds_path.exists() else pd.DataFrame()
    drop_preds = pd.read_parquet(drop_preds_path) if drop_preds_path.exists() else pd.DataFrame()

    blur_border_pct = compute_fp_rate_near_ignore_borders(blur_preds, ignore_df)
    drop_border_pct = compute_fp_rate_near_ignore_borders(drop_preds, ignore_df)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))

    # Left panel: AP, AP50, APs
    metrics_keys = ["AP", "AP50", "APs"]
    blur_vals = [float(blur_eval.get(k, 0.0)) for k in metrics_keys]
    drop_vals = [float(drop_eval.get(k, 0.0)) for k in metrics_keys]
    x = np.arange(len(metrics_keys))
    w = 0.34
    axes[0].bar(x - w / 2, blur_vals, w, label="Blur (GT-protected)", color="#2a9d8f")
    axes[0].bar(x + w / 2, drop_vals, w, label="Drop (unmodified pixels)", color="#e76f51")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_keys, fontsize=9)
    axes[0].set_ylabel("Validation Score")
    axes[0].set_title("Official Holdout Accuracy (mAP / AP50 / AP_small)", fontsize=9.5)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    # Right panel: Detections in 24px border ring around ignore regions
    axes[1].bar(
        ["Blur (GT-protected)", "Drop (unmodified)"],
        [blur_border_pct, drop_border_pct],
        color=["#2a9d8f", "#e76f51"],
        width=0.45,
    )
    axes[1].set_ylabel("Detections in 24px Ignore Border Ring (%)")
    axes[1].set_title("Border-Zone Detection Share (conf ≥ 0.25)", fontsize=9.5)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("U1.2 Ignore-Policy Ablation: GT-Protected Blur vs Drop", fontsize=10, y=1.02)
    fig.tight_layout()
    save_figure_under_budget(fig, out_path)

    winner = "blur" if float(blur_eval.get("AP", 0.0)) >= float(drop_eval.get("AP", 0.0)) else "drop"
    return {
        "blur": {
            "AP": blur_eval.get("AP"),
            "AP50": blur_eval.get("AP50"),
            "APs": blur_eval.get("APs"),
            "border_ring_pct": blur_border_pct,
        },
        "drop": {
            "AP": drop_eval.get("AP"),
            "AP50": drop_eval.get("AP50"),
            "APs": drop_eval.get("APs"),
            "border_ring_pct": drop_border_pct,
        },
        "winner": winner,
        "figure": out_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute or plot U1.2 ignore-policy ablation.")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from existing run dirs.")
    parser.add_argument("--blur-run-id", type=str, default="yolo-u12_blur")
    parser.add_argument("--drop-run-id", type=str, default="yolo-u12_drop")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--time-budget-h", type=float, default=None)
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.time_budget_h is not None:
        overrides["time_budget_h"] = args.time_budget_h

    if not args.plot_only:
        blur_cfg = load_resolved_config(
            P.configs / "yolo" / "u12_ablation_blur.yaml", cli_overrides=overrides
        )
        drop_cfg = load_resolved_config(
            P.configs / "yolo" / "u12_ablation_drop.yaml", cli_overrides=overrides
        )
        with timer("U1.2 Arm 1 (blur)"):
            train_ultralytics(blur_cfg, run_id=args.blur_run_id, evaluate_full_val=True)
        with timer("U1.2 Arm 2 (drop)"):
            train_ultralytics(drop_cfg, run_id=args.drop_run_id, evaluate_full_val=True)

    blur_dir = P.exp_runs_dir / args.blur_run_id
    drop_dir = P.exp_runs_dir / args.drop_run_id
    out_fig = P.images / "ignore_ablation.png"
    summary = plot_ignore_ablation(blur_dir, drop_dir, out_fig)
    save_json(summary, P.experiments / "u12_ignore_ablation_summary.json")
    print("\n=== U1.2 Ignore-Policy Ablation Summary ===")
    print(summary)


if __name__ == "__main__":
    main()

