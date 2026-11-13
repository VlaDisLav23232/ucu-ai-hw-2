"""Data-contract verification checks and diagnostic figures.

1. YOLO txt -> pixels matches COCO bbox within 1 px for every resolution bucket.
2. Deduplicated COCO crowd set equals txt ignore set per frame (IoU > 0.99).
3. Share of GT boxes with IoA > 0.5 inside ignore regions, per class.
4. Temporal-jump check on ignore regions (shift > 20% of region size between consecutive frames).
5. Export diagnostic figures to `images/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vdet.constants import CLASS_NAMES, NUM_CLASSES
from vdet.data.manifest import (
    load_boxes,
    load_ignore,
    load_manifest,
    parse_ignore_txt_file,
    parse_label_txt_file,
)
from vdet.paths import P
from vdet.utils import save_json, timer
from vdet.viz import render_cache_vs_original, render_clip_strip, render_frame_grid


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two [N, 4] and [M, 4] arrays in (x, y, w, h) format."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    ax1, ay1 = boxes_a[:, 0][:, None], boxes_a[:, 1][:, None]
    ax2, ay2 = (boxes_a[:, 0] + boxes_a[:, 2])[:, None], (boxes_a[:, 1] + boxes_a[:, 3])[:, None]
    bx1, by1 = boxes_b[:, 0][None, :], boxes_b[:, 1][None, :]
    bx2, by2 = (boxes_b[:, 0] + boxes_b[:, 2])[None, :], (boxes_b[:, 1] + boxes_b[:, 3])[None, :]

    inter_w = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    inter_h = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = inter_w * inter_h

    area_a = (boxes_a[:, 2] * boxes_a[:, 3])[:, None]
    area_b = (boxes_b[:, 2] * boxes_b[:, 3])[None, :]
    union = np.maximum(1e-9, area_a + area_b - inter)
    return inter / union


def box_ioa_max_with_ignores(gt_boxes: np.ndarray, ig_boxes: np.ndarray) -> np.ndarray:
    """For each GT box [N, 4], compute max(intersection / gt_area) across ignore regions [M, 4]."""
    if len(gt_boxes) == 0 or len(ig_boxes) == 0:
        return np.zeros(len(gt_boxes), dtype=np.float64)
    gx1, gy1 = gt_boxes[:, 0][:, None], gt_boxes[:, 1][:, None]
    gx2, gy2 = (gt_boxes[:, 0] + gt_boxes[:, 2])[:, None], (gt_boxes[:, 1] + gt_boxes[:, 3])[:, None]
    ix1, iy1 = ig_boxes[:, 0][None, :], ig_boxes[:, 1][None, :]
    ix2, iy2 = (ig_boxes[:, 0] + ig_boxes[:, 2])[None, :], (ig_boxes[:, 1] + ig_boxes[:, 3])[None, :]

    inter_w = np.maximum(0.0, np.minimum(gx2, ix2) - np.maximum(gx1, ix1))
    inter_h = np.maximum(0.0, np.minimum(gy2, iy2) - np.maximum(gy1, iy1))
    inter = inter_w * inter_h
    gt_area = np.maximum(1e-9, (gt_boxes[:, 2] * gt_boxes[:, 3])[:, None])
    return np.max(inter / gt_area, axis=1)


def check_1_yolo_vs_coco_per_bucket(
    manifest_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    data_root: Path,
    samples_per_bucket: int = 25,
) -> dict[str, dict[str, float]]:
    """Verify YOLO txt -> pixels matches COCO bbox within 1 px across every resolution bucket."""
    annotated = manifest_df[manifest_df["split"].isin(["train", "val"])]
    boxes_by_key = {k: g for k, g in boxes_df.groupby("image_key")}
    results: dict[str, dict[str, float]] = {}

    for bucket, group in annotated.groupby("res_bucket"):
        sampled = group.head(samples_per_bucket)
        max_err = 0.0
        n_checked = 0
        for row in sampled.itertuples(index=False):
            label_path = (
                data_root / str(row.split) / "labels" / str(row.clip) / f"{int(row.frame):07d}.txt"
            )
            txt_boxes = parse_label_txt_file(label_path, int(row.width), int(row.height))
            coco_sub = boxes_by_key.get(row.image_key, pd.DataFrame())
            if len(txt_boxes) != len(coco_sub):
                raise AssertionError(
                    f"Box count mismatch on {row.image_key}: txt={len(txt_boxes)}, coco={len(coco_sub)}"
                )
            if not txt_boxes:
                continue
            txt_arr = np.array([[b[1], b[2], b[3], b[4]] for b in txt_boxes], dtype=np.float64)
            coco_arr = coco_sub[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
            err = float(np.max(np.abs(txt_arr - coco_arr)))
            max_err = max(max_err, err)
            n_checked += len(txt_boxes)

        if max_err > 1.0:
            raise AssertionError(
                f"Resolution bucket {bucket} max coordinate error {max_err:.4f} px > 1.0 px"
            )
        results[str(bucket)] = {"max_abs_px_err": round(max_err, 5), "boxes_checked": n_checked}
    return results


def check_2_crowd_dedup_vs_ignore_txt(
    manifest_df: pd.DataFrame,
    ignore_df: pd.DataFrame,
    data_root: Path,
    max_frames: int = 500,
) -> dict[str, Any]:
    """Verify deduplicated COCO crowd boxes match ignore/*.txt boxes with IoU > 0.99."""
    ig_frames = manifest_df[(manifest_df["split"] == "train") & (manifest_df["n_ignore"] > 0)].head(
        max_frames
    )
    ig_by_key = {k: g for k, g in ignore_df.groupby("image_key")}

    min_iou = 1.0
    checked_regions = 0
    for row in ig_frames.itertuples(index=False):
        txt_path = (
            data_root / str(row.split) / "ignore" / str(row.clip) / f"{int(row.frame):07d}.txt"
        )
        txt_regs = parse_ignore_txt_file(txt_path, int(row.width), int(row.height))
        coco_regs = ig_by_key.get(row.image_key, pd.DataFrame())
        if len(txt_regs) != len(coco_regs):
            raise AssertionError(
                f"Ignore region count mismatch on {row.image_key}: txt={len(txt_regs)}, dedup_coco={len(coco_regs)}"
            )
        if not txt_regs:
            continue
        t_arr = np.asarray(txt_regs, dtype=np.float64)
        c_arr = coco_regs[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
        ious = box_iou_matrix(t_arr, c_arr)
        best_iou = float(np.min(np.max(ious, axis=1)))
        min_iou = min(min_iou, best_iou)
        checked_regions += len(txt_regs)

    if min_iou <= 0.99:
        raise AssertionError(f"Minimum crowd-vs-txt IoU {min_iou:.4f} <= 0.99")
    return {
        "frames_checked": len(ig_frames),
        "regions_checked": checked_regions,
        "min_matched_iou": round(min_iou, 6),
    }


def check_3_gt_inside_ignore_per_class(
    boxes_df: pd.DataFrame,
    ignore_df: pd.DataFrame,
    ioa_thresh: float = 0.5,
) -> dict[str, dict[str, float | int]]:
    """Report share of GT boxes with IoA > 0.5 inside ignore regions, per class."""
    train_b = boxes_df[boxes_df["split"] == "train"]
    train_i = ignore_df[ignore_df["split"] == "train"]
    ig_by_key = {
        k: g[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
        for k, g in train_i.groupby("image_key")
    }

    total_per_cls = np.zeros(NUM_CLASSES, dtype=np.int64)
    inside_per_cls = np.zeros(NUM_CLASSES, dtype=np.int64)

    for key, b_group in train_b.groupby("image_key"):
        cls_arr = b_group["cls"].to_numpy(dtype=np.int64)
        for c in range(NUM_CLASSES):
            total_per_cls[c] += int(np.sum(cls_arr == c))
        ig_arr = ig_by_key.get(key)
        if ig_arr is None or len(ig_arr) == 0:
            continue
        gt_arr = b_group[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
        max_ioa = box_ioa_max_with_ignores(gt_arr, ig_arr)
        overlapped = max_ioa > ioa_thresh
        if np.any(overlapped):
            for c in range(NUM_CLASSES):
                inside_per_cls[c] += int(np.sum(overlapped & (cls_arr == c)))

    report: dict[str, dict[str, float | int]] = {}
    for c, name in enumerate(CLASS_NAMES):
        tot = int(total_per_cls[c])
        ins = int(inside_per_cls[c])
        share = float(ins / tot) if tot > 0 else 0.0
        report[name] = {
            "total_boxes": tot,
            "inside_ignore_ioa_gt_0_5": ins,
            "share_pct": round(share * 100.0, 3),
        }
    return report


def check_4_ignore_temporal_jumps(
    manifest_df: pd.DataFrame,
    ignore_df: pd.DataFrame,
    shift_thresh: float = 0.20,
) -> dict[str, Any]:
    """Flag consecutive frames where an ignore region shifts > 20% of its size."""
    train_i = ignore_df[ignore_df["split"] == "train"]
    ig_by_key = {
        k: g[["x", "y", "w", "h"]].to_numpy(dtype=np.float64)
        for k, g in train_i.groupby("image_key")
    }
    train_m = manifest_df[manifest_df["split"] == "train"].sort_values(["clip", "frame"])

    flagged_transitions = 0
    total_transitions = 0
    examples: list[dict[str, Any]] = []

    for clip, group in train_m.groupby("clip"):
        keys = group["image_key"].tolist()
        frames = group["frame"].tolist()
        for idx in range(1, len(keys)):
            prev_ig = ig_by_key.get(keys[idx - 1])
            curr_ig = ig_by_key.get(keys[idx])
            if prev_ig is None or curr_ig is None or len(prev_ig) == 0 or len(curr_ig) == 0:
                continue
            total_transitions += 1
            # Match each current ignore region to its closest previous region by center distance / size
            pcx = prev_ig[:, 0] + 0.5 * prev_ig[:, 2]
            pcy = prev_ig[:, 1] + 0.5 * prev_ig[:, 3]
            ccx = curr_ig[:, 0] + 0.5 * curr_ig[:, 2]
            ccy = curr_ig[:, 1] + 0.5 * curr_ig[:, 3]
            dist = np.hypot(ccx[:, None] - pcx[None, :], ccy[:, None] - pcy[None, :])
            size_ref = np.maximum(
                1.0,
                np.sqrt(curr_ig[:, 2][:, None] * curr_ig[:, 3][:, None]),
            )
            rel_shift = np.min(dist / size_ref, axis=1)
            max_rel_shift = float(np.max(rel_shift))
            if max_rel_shift > shift_thresh:
                flagged_transitions += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "clip": str(clip),
                            "frame_prev": int(frames[idx - 1]),
                            "frame_curr": int(frames[idx]),
                            "max_rel_shift": round(max_rel_shift, 3),
                        }
                    )

    return {
        "total_ignore_frame_transitions": total_transitions,
        "flagged_transitions_gt_20pct": flagged_transitions,
        "flagged_rate_pct": round(
            100.0 * flagged_transitions / max(1, total_transitions), 2
        ),
        "sample_flagged": examples,
    }


def export_u03_figures(
    manifest_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    ignore_df: pd.DataFrame,
    data_root: Path,
    images_dir: Path,
    cache_root: Path,
) -> list[Path]:
    """Export all data-check figures to images/ (guaranteed <= 300 KB each)."""
    saved: list[Path] = []
    annotated = manifest_df[manifest_df["split"].isin(["train", "val"])]

    # 1. gt_grid_<bucket>.png (one per resolution bucket in train/val)
    for bucket, group in annotated.groupby("res_bucket"):
        sample = group.sort_values("n_boxes", ascending=False).head(4)
        out = images_dir / f"gt_grid_{bucket}.png"
        render_frame_grid(
            sample,
            boxes_df[boxes_df["image_key"].isin(sample["image_key"])],
            ignore_df[ignore_df["image_key"].isin(sample["image_key"])],
            data_root=data_root,
            out_path=out,
            ncols=2,
            suptitle=f"GT + Ignore Grid — Resolution Bucket {bucket}",
        )
        saved.append(out)

    # 2. ignore_worst_frames.png (top 6 frames with highest ignore_area_frac)
    worst_ig = (
        annotated.sort_values("ignore_area_frac", ascending=False)
        .groupby("clip")
        .head(1)
        .head(6)
    )
    out_worst = images_dir / "ignore_worst_frames.png"
    render_frame_grid(
        worst_ig,
        boxes_df[boxes_df["image_key"].isin(worst_ig["image_key"])],
        ignore_df[ignore_df["image_key"].isin(worst_ig["image_key"])],
        data_root=data_root,
        out_path=out_worst,
        ncols=3,
        suptitle="Highest Ignore-Region Coverage Frames (One per Clip)",
    )
    saved.append(out_worst)

    # 3. clip_strip_<clip>.png (30 consecutive frames from a dense validation clip)
    val_clips = manifest_df[manifest_df["split"] == "val"]["clip"].unique()
    target_clip = str(val_clips[0]) if len(val_clips) > 0 else str(annotated["clip"].iloc[0])
    out_strip = images_dir / f"clip_strip_{target_clip}.png"
    render_clip_strip(
        manifest_df,
        boxes_df,
        ignore_df,
        data_root=data_root,
        clip_name=target_clip,
        out_path=out_strip,
        n_frames=30,
        ncols=6,
    )
    saved.append(out_strip)

    # 4. cache_vs_original.png (if k5_blur_L1920 cache exists)
    blur_dir = cache_root / "k5_blur_L1920"
    drop_dir = cache_root / "k5_drop_L1920"
    if blur_dir.exists() and (blur_dir / "index.parquet").exists():
        idx_df = pd.read_parquet(blur_dir / "index.parquet")
        train_keys = set(idx_df[idx_df["split"] == "train"]["image_key"])
        cand = manifest_df[
            manifest_df["image_key"].isin(train_keys) & (manifest_df["n_ignore"] > 0)
        ].sort_values("ignore_area_frac", ascending=False)
        sample_keys = cand.groupby("clip").head(1).head(3)["image_key"].tolist()
        if sample_keys:
            out_cache = images_dir / "cache_vs_original.png"
            render_cache_vs_original(
                sample_keys,
                manifest_df,
                boxes_df,
                ignore_df,
                data_root=data_root,
                blur_cache_dir=blur_dir,
                drop_cache_dir=drop_dir if drop_dir.exists() else None,
                out_path=out_cache,
            )
            saved.append(out_cache)

    return saved


def run_data_verification(
    data_root: Path | None = None,
    cache_root: Path | None = None,
    images_dir: Path | None = None,
    export_figures: bool = True,
) -> dict[str, Any]:
    """Run all data checks, export figures, and save `cache/verify_data_report.json`."""
    data_root = data_root or P.data
    cache_root = cache_root or P.cache
    images_dir = images_dir or P.images

    manifest_df = load_manifest(cache_root)
    boxes_df = load_boxes(cache_root)
    ignore_df = load_ignore(cache_root)

    with timer("Check 1: YOLO txt vs COCO bbox per resolution bucket"):
        c1 = check_1_yolo_vs_coco_per_bucket(manifest_df, boxes_df, data_root)
    with timer("Check 2: Deduplicated COCO crowd vs ignore/*.txt"):
        c2 = check_2_crowd_dedup_vs_ignore_txt(manifest_df, ignore_df, data_root)
    with timer("Check 3: GT boxes with IoA > 0.5 inside ignore regions per class"):
        c3 = check_3_gt_inside_ignore_per_class(boxes_df, ignore_df)
    with timer("Check 4: Ignore region temporal jump check"):
        c4 = check_4_ignore_temporal_jumps(manifest_df, ignore_df)

    saved_figs: list[Path] = []
    if export_figures:
        with timer("Check 5: Export diagnostic figures to images/"):
            saved_figs = export_u03_figures(
                manifest_df, boxes_df, ignore_df,
                data_root=data_root, images_dir=images_dir, cache_root=cache_root,
            )

    report = {
        "check_1_yolo_vs_coco_buckets": c1,
        "check_2_crowd_vs_txt": c2,
        "check_3_gt_inside_ignore_per_class": c3,
        "check_4_ignore_temporal_jumps": c4,
        "saved_figures": [p.name for p in saved_figs],
    }
    save_json(report, cache_root / "verify_data_report.json")
    return report
