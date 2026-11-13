"""Offline cache builder: frame subsampling, long-side resize, and GT-preserving ignore policies."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from vdet.constants import COCO_CATEGORIES, YOLO_TO_COCO_ID
from vdet.data.export_ultra import write_ultralytics_data_yaml
from vdet.data.manifest import load_boxes, load_ignore, load_manifest, xywh_to_yolo_norm
from vdet.paths import P
from vdet.utils import get_git_info, save_json, save_yaml, sha256_file


def apply_gt_protected_blur(
    img_bgr: np.ndarray,
    ignore_xywh: np.ndarray,
    gt_xywh: np.ndarray,
    gt_dilate_px: int = 2,
    feather_px: int = 6,
) -> np.ndarray:
    """Apply soft-edged Gaussian blur inside (ignore regions \\ dilated GT boxes).

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image (H, W, 3) in BGR uint8.
    ignore_xywh : np.ndarray
        Array of shape (M, 4) with [x, y, w, h] in image pixel coordinates.
    gt_xywh : np.ndarray
        Array of shape (N, 4) with [x, y, w, h] in image pixel coordinates.
    gt_dilate_px : int
        Dilation margin (in pixels) around every labelled GT box to protect from blurring.
    feather_px : int
        Soft edge feathering width (in pixels) for the ignore mask.
    """
    if len(ignore_xywh) == 0:
        return img_bgr

    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    # 1. Fill ignore regions in mask and blur ROI per region with adaptive sigma
    blurred_full = img_bgr.copy()
    for row in ignore_xywh:
        ix1 = max(0, int(np.floor(row[0])))
        iy1 = max(0, int(np.floor(row[1])))
        ix2 = min(w, int(np.ceil(row[0] + row[2])))
        iy2 = min(h, int(np.ceil(row[1] + row[3])))
        rw = ix2 - ix1
        rh = iy2 - iy1
        if rw <= 1 or rh <= 1:
            continue
        mask[iy1:iy2, ix1:ix2] = 1.0

        sigma = max(8.0, 0.25 * float(min(rw, rh)))
        pad = int(np.ceil(2.0 * sigma))
        rx1 = max(0, ix1 - pad)
        ry1 = max(0, iy1 - pad)
        rx2 = min(w, ix2 + pad)
        ry2 = min(h, iy2 + pad)
        roi = img_bgr[ry1:ry2, rx1:rx2]
        roi_blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma, sigmaY=sigma)
        blurred_full[iy1:iy2, ix1:ix2] = roi_blur[iy1 - ry1 : iy2 - ry1, ix1 - rx1 : ix2 - rx1]

    if not np.any(mask > 0):
        return img_bgr

    # 2. Carve out dilated GT boxes before feathering
    gt_protect = np.zeros((h, w), dtype=bool)
    for row in gt_xywh:
        gx1 = max(0, int(np.floor(row[0])) - gt_dilate_px)
        gy1 = max(0, int(np.floor(row[1])) - gt_dilate_px)
        gx2 = min(w, int(np.ceil(row[0] + row[2])) + gt_dilate_px)
        gy2 = min(h, int(np.ceil(row[1] + row[3])) + gt_dilate_px)
        if gx2 > gx1 and gy2 > gy1:
            gt_protect[gy1:gy2, gx1:gx2] = True

    mask[gt_protect] = 0.0

    # 3. Feather mask edges (~6 px) and strictly re-zero inside dilated GT boxes
    if feather_px > 0:
        f_sigma = max(1.0, float(feather_px) / 2.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=f_sigma, sigmaY=f_sigma)
        mask[gt_protect] = 0.0

    alpha = mask[:, :, None]
    blended = alpha * blurred_full.astype(np.float32) + (1.0 - alpha) * img_bgr.astype(np.float32)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _process_single_frame(task: dict[str, Any]) -> dict[str, Any]:
    """Worker function to resize, optionally blur ignore regions, and write JPEG + YOLO label."""
    src_path = Path(task["src_path"])
    dst_img_path = Path(task["dst_img_path"])
    dst_lbl_path = Path(task["dst_lbl_path"])
    orig_w = int(task["orig_w"])
    orig_h = int(task["orig_h"])
    long_side = int(task["long_side"])
    policy = str(task["ignore_policy"])
    split = str(task["split"])
    jpeg_quality = int(task["jpeg_quality"])
    gt_dilate_px = int(task["gt_dilate_px"])
    feather_px = int(task["feather_px"])
    gt_boxes = np.asarray(task["gt_boxes"], dtype=np.float32)  # [N, 5]: cls, x, y, w, h
    ig_boxes = np.asarray(task["ig_boxes"], dtype=np.float32)  # [M, 4]: x, y, w, h

    scale = min(1.0, float(long_side) / float(max(orig_w, orig_h)))
    cache_w = int(round(orig_w * scale))
    cache_h = int(round(orig_h * scale))

    # Fast path: if scale == 1.0 and no pixel modifications needed (drop policy or val split or no ignores)
    need_blur = (split == "train") and (policy == "blur") and (len(ig_boxes) > 0)
    img_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image {src_path}")

    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (cache_w, cache_h), interpolation=cv2.INTER_AREA)

    if need_blur:
        scaled_ig = ig_boxes * scale if len(ig_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        scaled_gt = (
            gt_boxes[:, 1:5] * scale if len(gt_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        )
        img_bgr = apply_gt_protected_blur(
            img_bgr,
            ignore_xywh=scaled_ig,
            gt_xywh=scaled_gt,
            gt_dilate_px=gt_dilate_px,
            feather_px=feather_px,
        )

    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_img_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    # Write normalised YOLO labels (invariant to uniform scaling)
    dst_lbl_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if len(gt_boxes) > 0:
        cls_ids = gt_boxes[:, 0].astype(int)
        cx, cy, nw, nh = xywh_to_yolo_norm(
            gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3], gt_boxes[:, 4], orig_w, orig_h
        )
        for c, x_c, y_c, w_n, h_n in zip(cls_ids, cx, cy, nw, nh):
            lines.append(f"{c} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
    dst_lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "image_key": task["image_key"],
        "split": split,
        "cache_path": f"{split}/images/{task['image_key']}.jpg",
        "orig_w": orig_w,
        "orig_h": orig_h,
        "cache_w": cache_w,
        "cache_h": cache_h,
        "scale": float(scale),
    }


def _build_cached_coco_json(
    index_sub: pd.DataFrame,
    boxes_df: pd.DataFrame,
    ignore_df: pd.DataFrame,
    out_json_path: Path,
) -> Path:
    """Build COCO ground-truth JSON in cached coordinates for torchvision training/validation."""
    images_list: list[dict[str, Any]] = []
    anns_list: list[dict[str, Any]] = []

    boxes_by_key = {k: g for k, g in boxes_df.groupby("image_key")} if not boxes_df.empty else {}
    ignore_by_key = {k: g for k, g in ignore_df.groupby("image_key")} if not ignore_df.empty else {}

    ann_id = 1
    for img_id, row in enumerate(index_sub.itertuples(index=False), start=1):
        key = str(row.image_key)
        scale = float(row.scale)
        cw = int(row.cache_w)
        ch = int(row.cache_h)

        images_list.append(
            {
                "id": img_id,
                "image_key": key,
                "file_name": str(row.cache_path),
                "width": cw,
                "height": ch,
                "orig_width": int(row.orig_w),
                "orig_height": int(row.orig_h),
                "scale": scale,
            }
        )

        b_df = boxes_by_key.get(key)
        if b_df is not None and not b_df.empty:
            for b in b_df.itertuples(index=False):
                sx = float(b.x) * scale
                sy = float(b.y) * scale
                sw = float(b.w) * scale
                sh = float(b.h) * scale
                if sw <= 0 or sh <= 0:
                    continue
                anns_list.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": YOLO_TO_COCO_ID[int(b.cls)],
                        "bbox": [round(sx, 2), round(sy, 2), round(sw, 2), round(sh, 2)],
                        "area": round(sw * sh, 2),
                        "iscrowd": 0,
                        "track_id": int(b.track_id),
                        "occlusion": int(b.occlusion),
                        "truncation": int(b.truncation),
                    }
                )
                ann_id += 1

        i_df = ignore_by_key.get(key)
        if i_df is not None and not i_df.empty:
            for ig in i_df.itertuples(index=False):
                sx = float(ig.x) * scale
                sy = float(ig.y) * scale
                sw = float(ig.w) * scale
                sh = float(ig.h) * scale
                if sw <= 0 or sh <= 0:
                    continue
                anns_list.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [round(sx, 2), round(sy, 2), round(sw, 2), round(sh, 2)],
                        "area": round(sw * sh, 2),
                        "iscrowd": 1,
                        "ignore": 1,
                    }
                )
                ann_id += 1

    coco_dict = {
        "info": {"description": f"vdet cached COCO dataset ({out_json_path.stem})"},
        "categories": COCO_CATEGORIES,
        "images": images_list,
        "annotations": anns_list,
    }
    return save_json(coco_dict, out_json_path)


def build_cache_from_recipe(
    recipe: dict[str, Any],
    data_root: Path | None = None,
    cache_root: Path | None = None,
    max_frames_per_split: int | None = None,
) -> Path:
    """Build an offline cache directory according to `recipe`."""
    d_root = data_root or P.data
    c_root = cache_root or P.cache
    cache_name = str(recipe.get("cache_name", "k5_blur_L1920"))
    out_dir = c_root / cache_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_k = int(recipe.get("frame_step_k", 5))
    val_k = int(recipe.get("val_frame_step_k", 5))
    long_side = int(recipe.get("long_side", 1920))
    policy = str(recipe.get("ignore_policy", "blur")).lower()
    if policy not in {"blur", "drop"}:
        raise ValueError(f"Unsupported ignore_policy '{policy}'; expected 'blur' or 'drop'.")
    gt_dilate_px = int(recipe.get("gt_dilate_px", 2))
    feather_px = int(recipe.get("feather_px", 6))
    jpeg_quality = int(recipe.get("jpeg_quality", 95))
    num_workers = int(recipe.get("num_workers", 8))
    splits = list(recipe.get("splits", ["train", "val"]))

    manifest_path = c_root / "manifest.parquet"
    manifest_df = load_manifest(c_root)
    boxes_df = load_boxes(c_root)
    ignore_df = load_ignore(c_root)

    boxes_by_key = {
        k: g[["cls", "x", "y", "w", "h"]].to_numpy(dtype=np.float32)
        for k, g in boxes_df.groupby("image_key")
    }
    ignore_by_key = {
        k: g[["x", "y", "w", "h"]].to_numpy(dtype=np.float32)
        for k, g in ignore_df.groupby("image_key")
    }

    tasks: list[dict[str, Any]] = []
    for split in splits:
        step_k = train_k if split == "train" else val_k
        sub = manifest_df[
            (manifest_df["split"] == split) & ((manifest_df["frame"] - 1) % step_k == 0)
        ].sort_values(["clip", "frame"])
        if max_frames_per_split is not None:
            sub = sub.head(max_frames_per_split)

        for row in sub.itertuples(index=False):
            key = str(row.image_key)
            tasks.append(
                {
                    "image_key": key,
                    "split": split,
                    "src_path": (d_root / str(row.path)).as_posix(),
                    "dst_img_path": (out_dir / split / "images" / f"{key}.jpg").as_posix(),
                    "dst_lbl_path": (out_dir / split / "labels" / f"{key}.txt").as_posix(),
                    "orig_w": int(row.width),
                    "orig_h": int(row.height),
                    "long_side": long_side,
                    "ignore_policy": policy,
                    "jpeg_quality": jpeg_quality,
                    "gt_dilate_px": gt_dilate_px,
                    "feather_px": feather_px,
                    "gt_boxes": boxes_by_key.get(key, np.zeros((0, 5), dtype=np.float32)),
                    "ig_boxes": ignore_by_key.get(key, np.zeros((0, 4), dtype=np.float32)),
                }
            )

    if num_workers > 1 and len(tasks) > 16:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            index_rows = list(pool.map(_process_single_frame, tasks, chunksize=32))
    else:
        index_rows = [_process_single_frame(t) for t in tasks]

    index_df = pd.DataFrame(index_rows)
    index_df.to_parquet(out_dir / "index.parquet", index=False, compression="zstd")

    # Export Ultralytics data.yaml
    write_ultralytics_data_yaml(out_dir)

    # Export cached COCO JSONs for torchvision path
    for split in splits:
        sub_idx = index_df[index_df["split"] == split]
        sub_keys = set(sub_idx["image_key"])
        _build_cached_coco_json(
            sub_idx,
            boxes_df[boxes_df["image_key"].isin(sub_keys)],
            ignore_df[ignore_df["image_key"].isin(sub_keys)],
            out_dir / f"coco_{split}.json",
        )

    # Write recipe.yaml with git hash and manifest sha256
    recipe_meta = {
        **recipe,
        "git": get_git_info(),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else "unknown",
        "counts": {
            split: int((index_df["split"] == split).sum()) for split in splits
        },
    }
    save_yaml(recipe_meta, out_dir / "recipe.yaml")
    return out_dir

