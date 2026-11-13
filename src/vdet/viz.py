"""Visual debugging utilities: GT/pred/ignore overlays, grids, clip strips, and cache comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vdet.constants import CLASS_COLORS_RGB, ID_TO_CLASS

matplotlib.use("Agg")


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image file from disk as an RGB uint8 NumPy array."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_figure_under_budget(
    fig: plt.Figure,
    out_path: str | Path,
    max_kb: int = 300,
    initial_dpi: int = 110,
    dpi: int | None = None,
) -> Path:
    """Save a matplotlib figure as a PNG guaranteed to stay under `max_kb` KB."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    start_dpi = int(dpi) if dpi is not None else int(initial_dpi)

    for cur_dpi in (start_dpi, 90, 75, 60, 48):
        fig.savefig(p, dpi=cur_dpi, bbox_inches="tight", pad_inches=0.06)
        if p.stat().st_size <= max_kb * 1024:
            plt.close(fig)
            return p

    # If still above budget (e.g. photographic grid), resize + max PNG compression via OpenCV
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is not None:
        scale = 0.85
        while p.stat().st_size > max_kb * 1024 and scale >= 0.35:
            nh = max(64, int(round(img.shape[0] * scale)))
            nw = max(64, int(round(img.shape[1] * scale)))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(p), resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            scale *= 0.82

    plt.close(fig)
    return p


def draw_frame_annotations_ax(
    ax: plt.Axes,
    img_rgb: np.ndarray,
    gt_boxes: pd.DataFrame | None = None,
    ignore_boxes: pd.DataFrame | None = None,
    pred_boxes: pd.DataFrame | None = None,
    score_thresh: float = 0.25,
    title: str | None = None,
    show_labels: bool = False,
) -> None:
    """Draw RGB image with hatched ignore regions, class-colored GT boxes, and dashed predictions."""
    ax.imshow(img_rgb)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8, pad=2)

    # 1. Hatched ignore regions
    if ignore_boxes is not None and not ignore_boxes.empty:
        for row in ignore_boxes.itertuples(index=False):
            rect = patches.Rectangle(
                (float(row.x), float(row.y)),
                float(row.w),
                float(row.h),
                linewidth=1.0,
                edgecolor=(1.0, 0.85, 0.1, 0.9),
                facecolor=(0.95, 0.85, 0.1, 0.18),
                hatch="///",
            )
            ax.add_patch(rect)

    # 2. Ground-truth boxes
    if gt_boxes is not None and not gt_boxes.empty:
        for row in gt_boxes.itertuples(index=False):
            cls_id = int(row.cls)
            rgb = CLASS_COLORS_RGB.get(cls_id, (255, 255, 255))
            color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
            rect = patches.Rectangle(
                (float(row.x), float(row.y)),
                float(row.w),
                float(row.h),
                linewidth=1.1,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            if show_labels:
                ax.text(
                    float(row.x),
                    max(0.0, float(row.y) - 2.0),
                    ID_TO_CLASS.get(cls_id, str(cls_id)),
                    color="white",
                    fontsize=5,
                    bbox={"facecolor": color, "alpha": 0.7, "pad": 0.5, "edgecolor": "none"},
                )

    # 3. Prediction boxes (dashed)
    if pred_boxes is not None and not pred_boxes.empty:
        filtered = pred_boxes[pred_boxes["score"] >= score_thresh]
        for row in filtered.itertuples(index=False):
            cls_id = int(row.cls)
            rgb = CLASS_COLORS_RGB.get(cls_id, (0, 255, 0))
            color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
            rect = patches.Rectangle(
                (float(row.x), float(row.y)),
                float(row.w),
                float(row.h),
                linewidth=1.1,
                linestyle="--",
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)


def render_frame_grid(
    manifest_sub: pd.DataFrame,
    boxes_df: pd.DataFrame,
    ignore_df: pd.DataFrame,
    data_root: Path,
    out_path: Path,
    ncols: int = 2,
    suptitle: str | None = None,
    max_Dim: int = 640,
) -> Path:
    """Render a grid of frames with GT and hatched ignore regions."""
    n = len(manifest_sub)
    if n == 0:
        raise ValueError("Empty manifest subset passed to render_frame_grid.")
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 2.6))
    axes_flat = np.atleast_1d(axes).ravel()

    boxes_by_key = {k: g for k, g in boxes_df.groupby("image_key")} if not boxes_df.empty else {}
    ignore_by_key = {k: g for k, g in ignore_df.groupby("image_key")} if not ignore_df.empty else {}

    for idx, row in enumerate(manifest_sub.itertuples(index=False)):
        ax = axes_flat[idx]
        img_path = data_root / str(row.path)
        img = load_rgb_image(img_path)
        h, w = img.shape[:2]
        scale = min(1.0, float(max_Dim) / float(max(w, h)))
        if scale < 1.0:
            img = cv2.resize(
                img,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        b_df = boxes_by_key.get(row.image_key, pd.DataFrame()).copy()
        i_df = ignore_by_key.get(row.image_key, pd.DataFrame()).copy()
        if scale < 1.0:
            for df in (b_df, i_df):
                if not df.empty:
                    df["x"] = df["x"] * scale
                    df["y"] = df["y"] * scale
                    df["w"] = df["w"] * scale
                    df["h"] = df["h"] * scale

        title = (
            f"{row.clip} f={row.frame} ({row.res_bucket}) | "
            f"GT={int(row.n_boxes) if pd.notna(row.n_boxes) else 0}, "
            f"ign={row.n_ignore} ({row.ignore_area_frac*100:.1f}%)"
        )
        draw_frame_annotations_ax(ax, img, gt_boxes=b_df, ignore_boxes=i_df, title=title)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_axis_off()

    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.01)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def render_clip_strip(
    manifest_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    ignore_df: pd.DataFrame,
    data_root: Path,
    clip_name: str,
    out_path: Path,
    n_frames: int = 30,
    ncols: int = 6,
    thumb_long_side: int = 320,
) -> Path:
    """Render 30 consecutive frames from a clip to inspect temporal box/ignore alignment."""
    clip_rows = (
        manifest_df[manifest_df["clip"] == clip_name]
        .sort_values("frame")
        .head(n_frames)
        .reset_index(drop=True)
    )
    if clip_rows.empty:
        raise ValueError(f"Clip '{clip_name}' not found in manifest.")

    keys = set(clip_rows["image_key"])
    sub_b = boxes_df[boxes_df["image_key"].isin(keys)]
    sub_i = ignore_df[ignore_df["image_key"].isin(keys)]
    boxes_by_key = {k: g for k, g in sub_b.groupby("image_key")} if not sub_b.empty else {}
    ignore_by_key = {k: g for k, g in sub_i.groupby("image_key")} if not sub_i.empty else {}

    total = len(clip_rows)
    nrows = (total + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 1.45))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, row in enumerate(clip_rows.itertuples(index=False)):
        ax = axes_flat[idx]
        img = load_rgb_image(data_root / str(row.path))
        h, w = img.shape[:2]
        scale = min(1.0, float(thumb_long_side) / float(max(w, h)))
        if scale < 1.0:
            img = cv2.resize(
                img,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        b_df = boxes_by_key.get(row.image_key, pd.DataFrame()).copy()
        i_df = ignore_by_key.get(row.image_key, pd.DataFrame()).copy()
        if scale < 1.0:
            for df in (b_df, i_df):
                if not df.empty:
                    df["x"] = df["x"] * scale
                    df["y"] = df["y"] * scale
                    df["w"] = df["w"] * scale
                    df["h"] = df["h"] * scale
        draw_frame_annotations_ax(
            ax,
            img,
            gt_boxes=b_df,
            ignore_boxes=i_df,
            title=f"f={row.frame} (n={int(row.n_boxes)})",
        )

    for idx in range(total, len(axes_flat)):
        axes_flat[idx].set_axis_off()

    fig.suptitle(f"Clip strip: {clip_name} ({total} consecutive frames)", fontsize=10, y=1.01)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)


def render_cache_vs_original(
    image_keys: Sequence[str],
    manifest_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    ignore_df: pd.DataFrame,
    data_root: Path,
    blur_cache_dir: Path,
    drop_cache_dir: Path | None,
    out_path: Path,
    thumb_long_side: int = 480,
) -> Path:
    """Compare original image vs cached blur (and drop) with GT boxes overlaid."""
    nrows = len(image_keys)
    ncols = 3 if (drop_cache_dir is not None and drop_cache_dir.exists()) else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 2.5))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    m_map = manifest_df.set_index("image_key")
    for r, key in enumerate(image_keys):
        row = m_map.loc[key]
        split = str(row["split"])
        orig_img = load_rgb_image(data_root / str(row["path"]))
        oh, ow = orig_img.shape[:2]
        b_df = boxes_df[boxes_df["image_key"] == key].copy()
        i_df = ignore_df[ignore_df["image_key"] == key].copy()

        # Col 0: Original
        s0 = min(1.0, float(thumb_long_side) / float(max(ow, oh)))
        img0 = (
            cv2.resize(orig_img, (int(round(ow * s0)), int(round(oh * s0))), interpolation=cv2.INTER_AREA)
            if s0 < 1.0
            else orig_img
        )
        b0 = b_df.copy()
        i0 = i_df.copy()
        for df in (b0, i0):
            if not df.empty:
                df[["x", "y", "w", "h"]] *= s0
        draw_frame_annotations_ax(
            axes[r, 0], img0, gt_boxes=b0, ignore_boxes=i0, title=f"Original: {key}"
        )

        # Col 1: Blur cache
        blur_img_path = blur_cache_dir / split / "images" / f"{key}.jpg"
        blur_img = load_rgb_image(blur_img_path)
        bh, bw = blur_img.shape[:2]
        s1 = min(1.0, float(thumb_long_side) / float(max(bw, bh)))
        img1 = (
            cv2.resize(blur_img, (int(round(bw * s1)), int(round(bh * s1))), interpolation=cv2.INTER_AREA)
            if s1 < 1.0
            else blur_img
        )
        b1 = b_df.copy()
        if not b1.empty:
            b1["x"] = b1["x"] * (bw / ow) * s1
            b1["y"] = b1["y"] * (bh / oh) * s1
            b1["w"] = b1["w"] * (bw / ow) * s1
            b1["h"] = b1["h"] * (bh / oh) * s1
        draw_frame_annotations_ax(
            axes[r, 1], img1, gt_boxes=b1, ignore_boxes=None, title=f"Cache blur (GT preserved): {key}"
        )

        # Col 2: Drop cache (if present)
        if ncols == 3 and drop_cache_dir is not None:
            drop_img_path = drop_cache_dir / split / "images" / f"{key}.jpg"
            drop_img = load_rgb_image(drop_img_path)
            dh, dw = drop_img.shape[:2]
            s2 = min(1.0, float(thumb_long_side) / float(max(dw, dh)))
            img2 = (
                cv2.resize(drop_img, (int(round(dw * s2)), int(round(dh * s2))), interpolation=cv2.INTER_AREA)
                if s2 < 1.0
                else drop_img
            )
            b2 = b_df.copy()
            if not b2.empty:
                b2["x"] = b2["x"] * (dw / ow) * s2
                b2["y"] = b2["y"] * (dh / oh) * s2
                b2["w"] = b2["w"] * (dw / ow) * s2
                b2["h"] = b2["h"] * (dh / oh) * s2
            draw_frame_annotations_ax(
                axes[r, 2], img2, gt_boxes=b2, ignore_boxes=None, title=f"Cache drop: {key}"
            )

    fig.suptitle("Original vs Cached Frames (GT-Protected Blur & Drop Policies)", fontsize=10, y=1.01)
    fig.tight_layout()
    return save_figure_under_budget(fig, out_path)

