"""Frame-step redundancy analysis: track displacement and IoU across lag k = 1..20."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vdet.constants import CLASS_NAMES, NUM_CLASSES
from vdet.data.manifest import load_boxes, load_ignore, load_manifest
from vdet.paths import P
from vdet.viz import save_figure_under_budget

matplotlib.use("Agg")


def compute_track_lag_redundancy(
    boxes_df: pd.DataFrame,
    lags: Sequence[int] = tuple(range(1, 21)),
    max_clips: int | None = None,
) -> pd.DataFrame:
    """Compute median/p25/p75 IoU and normalized center displacement of same `track_id` between frame t and t+k.

    Vectorized via self-join on `(clip, track_id)` where `frame_t_plus_k = frame_t + k`.
    """
    valid = boxes_df[(boxes_df["split"] == "train") & (boxes_df["track_id"] >= 0)].copy()
    if max_clips is not None:
        keep_clips = valid["clip"].unique()[:max_clips]
        valid = valid[valid["clip"].isin(keep_clips)]

    # Deduplicate rare cases where same (clip, frame, track_id) appears twice
    valid = valid.drop_duplicates(subset=["clip", "frame", "track_id"])
    base = valid[["clip", "frame", "track_id", "x", "y", "w", "h"]]

    rows: list[dict[str, float | int]] = []
    for k in lags:
        shifted = base.copy()
        shifted["frame"] = shifted["frame"] - int(k)
        merged = base.merge(
            shifted,
            on=["clip", "frame", "track_id"],
            suffixes=("_t", "_tk"),
        )
        if merged.empty:
            continue

        x1_t = merged["x_t"].to_numpy(dtype=np.float64)
        y1_t = merged["y_t"].to_numpy(dtype=np.float64)
        w_t = merged["w_t"].to_numpy(dtype=np.float64)
        h_t = merged["h_t"].to_numpy(dtype=np.float64)
        x2_t = x1_t + w_t
        y2_t = y1_t + h_t

        x1_k = merged["x_tk"].to_numpy(dtype=np.float64)
        y1_k = merged["y_tk"].to_numpy(dtype=np.float64)
        w_k = merged["w_tk"].to_numpy(dtype=np.float64)
        h_k = merged["h_tk"].to_numpy(dtype=np.float64)
        x2_k = x1_k + w_k
        y2_k = y1_k + h_k

        inter_w = np.maximum(0.0, np.minimum(x2_t, x2_k) - np.maximum(x1_t, x1_k))
        inter_h = np.maximum(0.0, np.minimum(y2_t, y2_k) - np.maximum(y1_t, y1_k))
        inter = inter_w * inter_h
        union = np.maximum(1e-9, w_t * h_t + w_k * h_k - inter)
        iou = inter / union

        cx_t, cy_t = x1_t + 0.5 * w_t, y1_t + 0.5 * h_t
        cx_k, cy_k = x1_k + 0.5 * w_k, y1_k + 0.5 * h_k
        disp_px = np.hypot(cx_k - cx_t, cy_k - cy_t)
        box_diag = np.maximum(1.0, np.hypot(w_t, h_t))
        rel_disp = disp_px / box_diag

        rows.append(
            {
                "lag_k": int(k),
                "n_pairs": int(len(merged)),
                "median_iou": round(float(np.median(iou)), 4),
                "p25_iou": round(float(np.percentile(iou, 25)), 4),
                "p75_iou": round(float(np.percentile(iou, 75)), 4),
                "median_disp_px": round(float(np.median(disp_px)), 2),
                "median_rel_disp": round(float(np.median(rel_disp)), 4),
            }
        )

    return pd.DataFrame(rows)


def generate_all_eda_figures(
    cache_root: Path | None = None,
    images_dir: Path | None = None,
    target_long_side: int = 1280,
) -> tuple[dict[str, Path], pd.DataFrame]:
    """Generate all 8 EDA figures into `images/` (all <= 300 KB)."""
    c_root = cache_root or P.cache
    img_dir = images_dir or P.images
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = load_manifest(c_root)
    boxes_df = load_boxes(c_root)
    ignore_df = load_ignore(c_root)

    saved: dict[str, Path] = {}

    # 1. Class distribution (train vs val)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    train_b = boxes_df[boxes_df["split"] == "train"]
    val_b = boxes_df[boxes_df["split"] == "val"]
    t_counts = train_b["cls"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    v_counts = val_b["cls"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    t_pct = 100.0 * t_counts / max(1, len(train_b))
    v_pct = 100.0 * v_counts / max(1, len(val_b))

    x = np.arange(NUM_CLASSES)
    width = 0.36
    ax.bar(x - width / 2, t_pct, width, label=f"Train (n={len(train_b):,})", color="#2a9d8f")
    ax.bar(x + width / 2, v_pct, width, label=f"Val (n={len(val_b):,})", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_ylabel("Share of Labelled Boxes (%)")
    ax.set_title("Class Distribution: Train vs Val (Domain Shift in Person vs Car)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    saved["class_distribution"] = save_figure_under_budget(
        fig, img_dir / "eda_class_distribution.png"
    )

    # 2. Object-size histograms (native vs 1280 scale) with COCO buckets
    m_dims = manifest_df.set_index("image_key")[["width", "height"]]
    tb_joined = train_b.join(m_dims, on="image_key")
    native_sqrt_area = np.sqrt(tb_joined["w"] * tb_joined["h"])
    scale_1280 = float(target_long_side) / np.maximum(tb_joined["width"], tb_joined["height"])
    scaled_sqrt_area = native_sqrt_area * scale_1280

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax_i, arr, lbl in [
        (axes[0], native_sqrt_area, "Native Resolution"),
        (axes[1], scaled_sqrt_area, f"Resized to Long Side = {target_long_side}px"),
    ]:
        clipped = np.clip(arr, 1.0, 250.0)
        ax_i.hist(clipped, bins=60, color="#457b9d", alpha=0.85, density=True)
        ax_i.axvline(32.0, color="#e63946", linestyle="--", linewidth=1.4, label="COCO Small (32 px)")
        ax_i.axvline(96.0, color="#f4a261", linestyle="--", linewidth=1.4, label="COCO Med/Large (96 px)")
        s_pct = 100.0 * float((arr < 32.0).mean())
        m_pct = 100.0 * float(((arr >= 32.0) & (arr <= 96.0)).mean())
        l_pct = 100.0 * float((arr > 96.0).mean())
        ax_i.set_title(
            f"{lbl}\nS: {s_pct:.1f}% | M: {m_pct:.1f}% | L: {l_pct:.1f}% (med={np.median(arr):.1f}px)",
            fontsize=9,
        )
        ax_i.set_xlabel("Box Linear Size √(w·h) [pixels]")
        ax_i.set_ylabel("Density")
        ax_i.grid(True, alpha=0.3)
        ax_i.legend(fontsize=7)
    fig.tight_layout()
    saved["object_sizes"] = save_figure_under_budget(fig, img_dir / "eda_object_sizes.png")

    # 3. Boxes/frame distribution per clip (box plot)
    annotated_m = manifest_df[manifest_df["split"].isin(["train", "val"])].copy()
    clip_medians = annotated_m.groupby("clip")["n_boxes"].median().sort_values(ascending=False)
    ordered_clips = clip_medians.index.tolist()
    clip_data = [
        annotated_m[annotated_m["clip"] == c]["n_boxes"].to_numpy() for c in ordered_clips
    ]

    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    bp = ax.boxplot(clip_data, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], ordered_clips):
        is_val = c in set(manifest_df[manifest_df["split"] == "val"]["clip"])
        patch.set_facecolor("#e76f51" if is_val else "#2a9d8f")
        patch.set_alpha(0.75)
    ax.axhline(100.0, color="#e63946", linestyle="--", linewidth=1.2, label="pycocotools default maxDets=100")
    ax.set_xticks(range(1, len(ordered_clips) + 1))
    ax.set_xticklabels(ordered_clips, rotation=90, fontsize=5.5)
    ax.set_ylabel("Labelled Boxes per Frame")
    ax.set_title("Object Density per Sequence (Teal=Train, Orange=Val; Max=180/frame)", fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    saved["density_per_clip"] = save_figure_under_budget(fig, img_dir / "eda_density_per_clip.png")

    # 4. Per-clip class mix heatmap
    clip_cls = (
        boxes_df.groupby(["clip", "cls"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=ordered_clips, columns=range(NUM_CLASSES), fill_value=0)
    )
    clip_cls_share = clip_cls.div(clip_cls.sum(axis=1).replace(0, 1), axis=0)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    im = ax.imshow(clip_cls_share.to_numpy().T, aspect="auto", cmap="viridis")
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xticks(range(len(ordered_clips)))
    ax.set_xticklabels(ordered_clips, rotation=90, fontsize=5.5)
    ax.set_title("Per-Sequence Class Composition Heatmap (Row-Normalized Share)", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Class Share")
    fig.tight_layout()
    saved["clip_class_heatmap"] = save_figure_under_budget(
        fig, img_dir / "eda_clip_class_heatmap.png"
    )

    # 5. Resolution distribution across train / val / test
    res_table = (
        manifest_df.groupby(["res_bucket", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
    )
    res_pct = 100.0 * res_table.div(res_table.sum(axis=0).replace(0, 1), axis=1)

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    rx = np.arange(len(res_pct))
    rw = 0.26
    ax.bar(rx - rw, res_pct["train"], rw, label="Train", color="#2a9d8f")
    ax.bar(rx, res_pct["val"], rw, label="Val", color="#e76f51")
    ax.bar(rx + rw, res_pct["test"], rw, label="Test", color="#457b9d")
    ax.set_xticks(rx)
    ax.set_xticklabels(res_pct.index.tolist(), rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Share of Frames (%)")
    ax.set_title("Frame Resolution Shift: Train/Val vs Downscaled Test Frames", fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    saved["resolutions"] = save_figure_under_budget(fig, img_dir / "eda_resolutions.png")

    # 6. Ignore coverage per clip
    ig_clip = (
        annotated_m.groupby("clip")["ignore_area_frac"]
        .mean()
        .reindex(ordered_clips)
        * 100.0
    )
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.bar(range(len(ordered_clips)), ig_clip.to_numpy(), color="#e9c46a", edgecolor="#264653", linewidth=0.5)
    ax.set_xticks(range(len(ordered_clips)))
    ax.set_xticklabels(ordered_clips, rotation=90, fontsize=5.5)
    ax.set_ylabel("Mean Ignore Area (% of Frame)")
    ax.set_title("Deduplicated Ignore-Region Area Coverage per Sequence", fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    saved["ignore_coverage"] = save_figure_under_budget(fig, img_dir / "eda_ignore_coverage.png")

    # 7. Frame-step redundancy (k = 1..20)
    lag_df = compute_track_lag_redundancy(boxes_df, lags=range(1, 21))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    lags = lag_df["lag_k"].to_numpy()
    med_iou = lag_df["median_iou"].to_numpy()
    p25_iou = lag_df["p25_iou"].to_numpy()
    p75_iou = lag_df["p75_iou"].to_numpy()

    axes[0].plot(lags, med_iou, marker="o", markersize=4, color="#2a9d8f", linewidth=1.8, label="Median Track IoU")
    axes[0].fill_between(lags, p25_iou, p75_iou, color="#2a9d8f", alpha=0.2, label="IQR (p25–p75)")
    axes[0].axvline(5, color="#e63946", linestyle="--", linewidth=1.4, label="Chosen step k=5")
    axes[0].set_xlabel("Frame Lag k")
    axes[0].set_ylabel("Same-Track Box IoU(t, t+k)")
    axes[0].set_title("Temporal Redundancy: Box IoU vs Frame Step k", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7.5)

    axes[1].plot(
        lags,
        lag_df["median_rel_disp"].to_numpy(),
        marker="s",
        markersize=4,
        color="#e76f51",
        linewidth=1.8,
        label="Median Center Shift / Box Diag",
    )
    axes[1].axvline(5, color="#e63946", linestyle="--", linewidth=1.4, label="Chosen step k=5")
    axes[1].set_xlabel("Frame Lag k")
    axes[1].set_ylabel("Normalized Displacement (Δc / diag)")
    axes[1].set_title("Track Center Displacement vs Frame Step k", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7.5)

    fig.tight_layout()
    saved["frame_step_redundancy"] = save_figure_under_budget(
        fig, img_dir / "eda_frame_step_redundancy.png"
    )

    # 8. Box width/height scatter at 1280 scale (for anchor design)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(tb_joined), size=min(15000, len(tb_joined)), replace=False)
    sub_b = tb_joined.iloc[sample_idx]
    s_factor = float(target_long_side) / np.maximum(sub_b["width"], sub_b["height"])
    sw = sub_b["w"].to_numpy() * s_factor
    sh = sub_b["h"].to_numpy() * s_factor

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(sw, sh, s=2, alpha=0.18, c="#1d3557", rasterized=True)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.axvline(32, color="#e63946", linestyle="--", linewidth=1.0, label="Default smallest anchor (32px)")
    ax.axhline(32, color="#e63946", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"Box Width at {target_long_side}px Long Side")
    ax.set_ylabel(f"Box Height at {target_long_side}px Long Side")
    ax.set_title("Train Box Width vs Height Distribution (@1280 Scale)", fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    saved["box_wh_scatter"] = save_figure_under_budget(fig, img_dir / "eda_box_wh_scatter.png")

    return saved, lag_df

