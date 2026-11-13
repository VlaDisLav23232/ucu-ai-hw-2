"""Build and load manifest.parquet, boxes.parquet, and deduplicated ignore.parquet."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vdet.constants import (
    COCO_TO_YOLO_ID,
    EXPECTED_SPLIT_BOXES,
    EXPECTED_SPLIT_CLIPS,
    EXPECTED_SPLIT_FRAMES,
    NUM_CLASSES,
)
from vdet.paths import P


def yolo_norm_to_xywh(
    cx: float | np.ndarray,
    cy: float | np.ndarray,
    w: float | np.ndarray,
    h: float | np.ndarray,
    img_w: int | float,
    img_h: int | float,
    clip: bool = True,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Convert normalised YOLO center-format (cx, cy, w, h) in [0, 1] to top-left pixel (x, y, w, h)."""
    pw = w * img_w
    ph = h * img_h
    px = (cx * img_w) - 0.5 * pw
    py = (cy * img_h) - 0.5 * ph
    if clip:
        x2 = np.clip(px + pw, 0.0, float(img_w))
        y2 = np.clip(py + ph, 0.0, float(img_h))
        px = np.clip(px, 0.0, float(img_w))
        py = np.clip(py, 0.0, float(img_h))
        pw = np.maximum(0.0, x2 - px)
        ph = np.maximum(0.0, y2 - py)
    return px, py, pw, ph


def xywh_to_yolo_norm(
    x: float | np.ndarray,
    y: float | np.ndarray,
    w: float | np.ndarray,
    h: float | np.ndarray,
    img_w: int | float,
    img_h: int | float,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Convert top-left pixel (x, y, w, h) to normalised YOLO (cx, cy, w, h) in [0, 1]."""
    cx = (x + 0.5 * w) / float(img_w)
    cy = (y + 0.5 * h) / float(img_h)
    nw = w / float(img_w)
    nh = h / float(img_h)
    return (
        np.clip(cx, 0.0, 1.0),
        np.clip(cy, 0.0, 1.0),
        np.clip(nw, 0.0, 1.0),
        np.clip(nh, 0.0, 1.0),
    )


def compute_rect_union_area(
    boxes_xywh: list[tuple[float, float, float, float]] | np.ndarray,
    img_w: int,
    img_h: int,
) -> float:
    """Compute exact geometric union area of axis-aligned [x, y, w, h] rectangles clipped to image bounds."""
    if len(boxes_xywh) == 0:
        return 0.0
    arr = np.asarray(boxes_xywh, dtype=np.float64)
    x1 = np.clip(arr[:, 0], 0.0, float(img_w))
    y1 = np.clip(arr[:, 1], 0.0, float(img_h))
    x2 = np.clip(arr[:, 0] + arr[:, 2], 0.0, float(img_w))
    y2 = np.clip(arr[:, 1] + arr[:, 3], 0.0, float(img_h))

    valid = (x2 > x1) & (y2 > y1)
    if not np.any(valid):
        return 0.0
    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    if len(x1) == 1:
        return float((x2[0] - x1[0]) * (y2[0] - y1[0]))

    xs = np.unique(np.concatenate([x1, x2]))
    ys = np.unique(np.concatenate([y1, y2]))
    if len(xs) < 2 or len(ys) < 2:
        return 0.0

    # Coordinate-compressed cell coverage: (N, X-1, 1) & (N, 1, Y-1)
    cov_x = (xs[:-1][None, :] >= x1[:, None]) & (xs[1:][None, :] <= x2[:, None])
    cov_y = (ys[:-1][None, :] >= y1[:, None]) & (ys[1:][None, :] <= y2[:, None])
    covered = np.any(cov_x[:, :, None] & cov_y[:, None, :], axis=0)

    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    cell_areas = dx[:, None] * dy[None, :]
    return float(np.sum(cell_areas[covered]))


def deduplicate_crowd_boxes(
    crowd_bboxes: list[list[float] | tuple[float, float, float, float]],
    decimals: int = 2,
) -> list[tuple[float, float, float, float]]:
    """Deduplicate COCO crowd [x, y, w, h] entries while preserving order."""
    seen: set[tuple[float, float, float, float]] = set()
    unique: list[tuple[float, float, float, float]] = []
    for bbox in crowd_bboxes:
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if w <= 0 or h <= 0:
            continue
        key = (round(x, decimals), round(y, decimals), round(w, decimals), round(h, decimals))
        if key not in seen:
            seen.add(key)
            unique.append((x, y, w, h))
    return unique


def parse_ignore_txt_file(txt_path: Path, img_w: int, img_h: int) -> list[tuple[float, float, float, float]]:
    """Read normalised `cx cy w h` lines from an ignore/*.txt file into pixel (x, y, w, h) tuples."""
    if not txt_path.exists():
        return []
    regions: list[tuple[float, float, float, float]] = []
    with txt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            cx, cy, nw, nh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            px, py, pw, ph = yolo_norm_to_xywh(cx, cy, nw, nh, img_w, img_h, clip=True)
            if float(pw) > 0 and float(ph) > 0:
                regions.append((float(px), float(py), float(pw), float(ph)))
    return regions


def parse_label_txt_file(
    txt_path: Path, img_w: int, img_h: int
) -> list[tuple[int, float, float, float, float]]:
    """Read normalised `cls cx cy w h` lines from a labels/*.txt file into (cls, x, y, w, h) tuples."""
    if not txt_path.exists():
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    with txt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            px, py, pw, ph = yolo_norm_to_xywh(cx, cy, nw, nh, img_w, img_h, clip=True)
            if float(pw) > 0 and float(ph) > 0:
                boxes.append((cls_id, float(px), float(py), float(pw), float(ph)))
    return boxes


def _process_annotated_split(
    split: str,
    data_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Process a labelled split ('train' or 'val') from annotations_coco.json and ignore/*.txt."""
    split_dir = data_root / split
    coco_path = split_dir / "annotations_coco.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"Missing COCO annotations file: {coco_path}")

    with coco_path.open("r", encoding="utf-8") as fh:
        coco = json.load(fh)

    anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    crowds_by_img: dict[int, list[list[float]]] = defaultdict(list)
    for ann in coco["annotations"]:
        img_id = int(ann["image_id"])
        if int(ann.get("iscrowd", 0)) == 1 or int(ann.get("ignore", 0)) == 1:
            crowds_by_img[img_id].append(ann["bbox"])
        else:
            anns_by_img[img_id].append(ann)

    manifest_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    ignore_rows: list[dict[str, Any]] = []

    images_sorted = sorted(
        coco["images"],
        key=lambda im: (str(im["video"]), int(im["frame_id"])),
    )

    for im in images_sorted:
        img_id = int(im["id"])
        clip = str(im["video"])
        frame = int(im["frame_id"])
        width = int(im["width"])
        height = int(im["height"])
        image_key = f"{clip}_{frame:07d}"
        rel_path = f"{split}/images/{clip}/{frame:07d}.jpg"

        # 1. Labelled boxes (from COCO, which carries exact pixel coords + track_id/occlusion/truncation)
        img_anns = anns_by_img.get(img_id, [])
        cls_counts = [0] * NUM_CLASSES
        valid_box_count = 0

        for ann in img_anns:
            cat_id = int(ann["category_id"])
            if cat_id not in COCO_TO_YOLO_ID:
                continue
            cls_id = COCO_TO_YOLO_ID[cat_id]
            bx, by, bw, bh = (
                float(ann["bbox"][0]),
                float(ann["bbox"][1]),
                float(ann["bbox"][2]),
                float(ann["bbox"][3]),
            )
            # Clip to [0, width] x [0, height]
            x2 = min(float(width), max(0.0, bx + bw))
            y2 = min(float(height), max(0.0, by + bh))
            bx = min(float(width), max(0.0, bx))
            by = min(float(height), max(0.0, by))
            bw = x2 - bx
            bh = y2 - by
            if bw <= 0.0 or bh <= 0.0:
                continue

            cls_counts[cls_id] += 1
            valid_box_count += 1
            box_rows.append(
                {
                    "image_key": image_key,
                    "split": split,
                    "clip": clip,
                    "frame": frame,
                    "cls": cls_id,
                    "x": np.float32(bx),
                    "y": np.float32(by),
                    "w": np.float32(bw),
                    "h": np.float32(bh),
                    "track_id": int(ann.get("track_id", -1)),
                    "occlusion": int(ann.get("occlusion", 0)),
                    "truncation": int(ann.get("truncation", 0)),
                    "orig_category_id": int(ann.get("orig_category_id", -1)),
                }
            )

        # 2. Deduplicated ignore regions
        # Deduplicating COCO crowd boxes yields exact pixel coordinates while matching ignore/*.txt line count 1:1
        dedup_ignores = deduplicate_crowd_boxes(crowds_by_img.get(img_id, []))
        if not dedup_ignores:
            ignore_txt = split_dir / "ignore" / clip / f"{frame:07d}.txt"
            if ignore_txt.exists():
                dedup_ignores = parse_ignore_txt_file(ignore_txt, width, height)

        for ix, iy, iw, ih in dedup_ignores:
            ignore_rows.append(
                {
                    "image_key": image_key,
                    "split": split,
                    "clip": clip,
                    "frame": frame,
                    "x": np.float32(ix),
                    "y": np.float32(iy),
                    "w": np.float32(iw),
                    "h": np.float32(ih),
                }
            )

        union_area = compute_rect_union_area(dedup_ignores, width, height)
        ignore_area_frac = float(union_area / float(width * height)) if (width * height) > 0 else 0.0

        manifest_rows.append(
            {
                "image_key": image_key,
                "split": split,
                "clip": clip,
                "frame": frame,
                "path": rel_path,
                "width": width,
                "height": height,
                "n_boxes": float(valid_box_count),
                "n_ignore": int(len(dedup_ignores)),
                "ignore_area_frac": float(ignore_area_frac),
                "res_bucket": f"{width}x{height}",
                "cls_counts": cls_counts,
            }
        )

    return manifest_rows, box_rows, ignore_rows


def _process_test_split(data_root: Path) -> list[dict[str, Any]]:
    """Process the test split from test_images.csv."""
    test_csv = data_root / "test_images.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test_images.csv: {test_csv}")

    df_test = pd.read_csv(test_csv)
    manifest_rows: list[dict[str, Any]] = []
    zero_counts = [0] * NUM_CLASSES

    for row in df_test.itertuples(index=False):
        image_key = str(row.image_id)
        clip = str(row.clip)
        frame = int(row.frame)
        rel_path = str(row.file_name).replace("\\", "/")
        width = int(row.width)
        height = int(row.height)
        manifest_rows.append(
            {
                "image_key": image_key,
                "split": "test",
                "clip": clip,
                "frame": frame,
                "path": rel_path,
                "width": width,
                "height": height,
                "n_boxes": np.nan,
                "n_ignore": 0,
                "ignore_area_frac": 0.0,
                "res_bucket": f"{width}x{height}",
                "cls_counts": list(zero_counts),
            }
        )
    return manifest_rows


def build_manifests(
    data_root: Path | None = None,
    cache_root: Path | None = None,
    strict_verify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and write manifest.parquet, boxes.parquet, and ignore.parquet."""
    d_root = data_root or P.data
    c_root = cache_root or P.cache
    c_root.mkdir(parents=True, exist_ok=True)

    train_m, train_b, train_i = _process_annotated_split("train", d_root)
    val_m, val_b, val_i = _process_annotated_split("val", d_root)
    test_m = _process_test_split(d_root)

    manifest_df = pd.DataFrame(train_m + val_m + test_m)
    boxes_df = pd.DataFrame(train_b + val_b)
    ignore_df = pd.DataFrame(train_i + val_i)

    # Enforce schema types
    manifest_df["frame"] = manifest_df["frame"].astype("int32")
    manifest_df["width"] = manifest_df["width"].astype("int32")
    manifest_df["height"] = manifest_df["height"].astype("int32")
    manifest_df["n_boxes"] = manifest_df["n_boxes"].astype("float64")
    manifest_df["n_ignore"] = manifest_df["n_ignore"].astype("int32")
    manifest_df["ignore_area_frac"] = manifest_df["ignore_area_frac"].astype("float32")

    if not boxes_df.empty:
        boxes_df["frame"] = boxes_df["frame"].astype("int32")
        boxes_df["cls"] = boxes_df["cls"].astype("int8")
        boxes_df["x"] = boxes_df["x"].astype("float32")
        boxes_df["y"] = boxes_df["y"].astype("float32")
        boxes_df["w"] = boxes_df["w"].astype("float32")
        boxes_df["h"] = boxes_df["h"].astype("float32")
        boxes_df["track_id"] = boxes_df["track_id"].astype("int32")
        boxes_df["occlusion"] = boxes_df["occlusion"].astype("int8")
        boxes_df["truncation"] = boxes_df["truncation"].astype("int8")
        boxes_df["orig_category_id"] = boxes_df["orig_category_id"].astype("int8")

    if not ignore_df.empty:
        ignore_df["frame"] = ignore_df["frame"].astype("int32")
        ignore_df["x"] = ignore_df["x"].astype("float32")
        ignore_df["y"] = ignore_df["y"].astype("float32")
        ignore_df["w"] = ignore_df["w"].astype("float32")
        ignore_df["h"] = ignore_df["h"].astype("float32")

    if strict_verify:
        verify_manifest_counts(manifest_df, boxes_df, ignore_df)

    manifest_df.to_parquet(c_root / "manifest.parquet", index=False, compression="zstd")
    boxes_df.to_parquet(c_root / "boxes.parquet", index=False, compression="zstd")
    ignore_df.to_parquet(c_root / "ignore.parquet", index=False, compression="zstd")

    return manifest_df, boxes_df, ignore_df


def verify_manifest_counts(
    manifest_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    ignore_df: pd.DataFrame,
) -> dict[str, Any]:
    """Verify split frame counts, clip counts, box counts, and COCO metadata join rate."""
    stats: dict[str, Any] = {}
    for split, exp_frames in EXPECTED_SPLIT_FRAMES.items():
        sub = manifest_df[manifest_df["split"] == split]
        actual_frames = len(sub)
        actual_clips = sub["clip"].nunique()
        exp_clips = EXPECTED_SPLIT_CLIPS[split]
        if actual_frames != exp_frames:
            raise AssertionError(
                f"Split '{split}' frame count mismatch: got {actual_frames}, expected {exp_frames}"
            )
        if actual_clips != exp_clips:
            raise AssertionError(
                f"Split '{split}' clip count mismatch: got {actual_clips}, expected {exp_clips}"
            )
        stats[f"{split}_frames"] = actual_frames
        stats[f"{split}_clips"] = actual_clips

    for split, exp_boxes in EXPECTED_SPLIT_BOXES.items():
        sub_b = boxes_df[boxes_df["split"] == split]
        actual_boxes = len(sub_b)
        if actual_boxes != exp_boxes:
            raise AssertionError(
                f"Split '{split}' box count mismatch: got {actual_boxes}, expected {exp_boxes}"
            )
        join_rate = float((sub_b["track_id"] >= 0).mean())
        if join_rate < 0.999:
            raise AssertionError(
                f"Split '{split}' COCO metadata join rate {join_rate:.4f} < 0.999"
            )
        stats[f"{split}_boxes"] = actual_boxes
        stats[f"{split}_join_rate"] = join_rate

    train_ig_frames = int((manifest_df.query("split == 'train'")["n_ignore"] > 0).sum())
    if train_ig_frames != 18_090:
        raise AssertionError(
            f"Train frames with ignore regions mismatch: got {train_ig_frames}, expected 18090"
        )
    train_ig_boxes = int(len(ignore_df[ignore_df["split"] == "train"]))
    if train_ig_boxes != 63_843:
        raise AssertionError(
            f"Train deduplicated ignore boxes mismatch: got {train_ig_boxes}, expected 63843"
        )
    stats["train_ignore_frames"] = train_ig_frames
    stats["train_ignore_boxes"] = train_ig_boxes
    stats["val_ignore_boxes"] = int(len(ignore_df[ignore_df["split"] == "val"]))
    return stats


def load_manifest(cache_root: Path | None = None) -> pd.DataFrame:
    """Load cache/manifest.parquet."""
    c_root = cache_root or P.cache
    return pd.read_parquet(c_root / "manifest.parquet")


def load_boxes(cache_root: Path | None = None, split: str | None = None) -> pd.DataFrame:
    """Load cache/boxes.parquet, optionally filtered by split."""
    c_root = cache_root or P.cache
    df = pd.read_parquet(c_root / "boxes.parquet")
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def load_ignore(cache_root: Path | None = None, split: str | None = None) -> pd.DataFrame:
    """Load cache/ignore.parquet, optionally filtered by split."""
    c_root = cache_root or P.cache
    df = pd.read_parquet(c_root / "ignore.parquet")
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df

