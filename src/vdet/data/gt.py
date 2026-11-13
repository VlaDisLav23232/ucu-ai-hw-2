"""Build COCO ground-truth dictionaries/JSONs with deduplicated crowd regions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from vdet.constants import COCO_CATEGORIES, NUM_CLASSES, YOLO_TO_COCO_ID
from vdet.data.manifest import load_boxes, load_ignore, load_manifest
from vdet.paths import P
from vdet.utils import save_json


def build_coco_gt_dict(
    split: str = "val",
    subset_keys: Sequence[str] | set[str] | None = None,
    manifest_df: pd.DataFrame | None = None,
    boxes_df: pd.DataFrame | None = None,
    ignore_df: pd.DataFrame | None = None,
    cache_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build an in-memory COCO GT dict and `image_key -> int image_id` map for evaluation.

    Notes
    -----
    - Spatial ignore regions in `ignore.parquet` are already deduplicated per frame.
    - Because COCO evaluators (`faster-coco-eval` and `pycocotools`) evaluate each `category_id`
      (1..8) independently, each unique spatial ignore region is emitted with `iscrowd=1` for
      each `category_id` in 1..8 so that a detection of *any* class falling inside an ignore
      region is ignored.
    """
    c_root = cache_root or P.cache
    if manifest_df is None:
        manifest_df = load_manifest(c_root)
    if boxes_df is None:
        boxes_df = load_boxes(c_root, split=split)
    else:
        boxes_df = boxes_df[boxes_df["split"] == split]
    if ignore_df is None:
        ignore_df = load_ignore(c_root, split=split)
    else:
        ignore_df = ignore_df[ignore_df["split"] == split]

    m_sub = manifest_df[manifest_df["split"] == split].sort_values(["clip", "frame"])
    if subset_keys is not None:
        key_set = set(subset_keys)
        m_sub = m_sub[m_sub["image_key"].isin(key_set)]

    key_to_id: dict[str, int] = {}
    images_list: list[dict[str, Any]] = []
    for img_id, row in enumerate(m_sub.itertuples(index=False), start=1):
        key = str(row.image_key)
        key_to_id[key] = img_id
        images_list.append(
            {
                "id": img_id,
                "image_key": key,
                "file_name": str(row.path),
                "width": int(row.width),
                "height": int(row.height),
                "res_bucket": str(row.res_bucket),
                "clip": str(row.clip),
                "frame": int(row.frame),
            }
        )

    valid_keys = set(key_to_id.keys())
    b_sub = boxes_df[boxes_df["image_key"].isin(valid_keys)]
    i_sub = ignore_df[ignore_df["image_key"].isin(valid_keys)]

    annotations_list: list[dict[str, Any]] = []
    ann_id = 1

    # 1. Labelled GT boxes (iscrowd = 0)
    for row in b_sub.itertuples(index=False):
        img_id = key_to_id[str(row.image_key)]
        bx = float(row.x)
        by = float(row.y)
        bw = float(row.w)
        bh = float(row.h)
        if bw <= 0.0 or bh <= 0.0:
            continue
        annotations_list.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": YOLO_TO_COCO_ID[int(row.cls)],
                "bbox": [bx, by, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    # 2. Deduplicated spatial ignore regions (iscrowd = 1 across categories 1..8)
    for row in i_sub.itertuples(index=False):
        img_id = key_to_id[str(row.image_key)]
        ix = float(row.x)
        iy = float(row.y)
        iw = float(row.w)
        ih = float(row.h)
        if iw <= 0.0 or ih <= 0.0:
            continue
        area = iw * ih
        bbox = [ix, iy, iw, ih]
        for cat_id in range(1, NUM_CLASSES + 1):
            annotations_list.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 1,
                    "ignore": 1,
                }
            )
            ann_id += 1

    coco_dict: dict[str, Any] = {
        "info": {"description": f"vdet COCO GT ({split}, n_images={len(images_list)})"},
        "categories": COCO_CATEGORIES,
        "images": images_list,
        "annotations": annotations_list,
    }
    return coco_dict, key_to_id


def save_coco_gt_json(
    out_path: str | Path,
    split: str = "val",
    subset_keys: Sequence[str] | set[str] | None = None,
    cache_root: Path | None = None,
) -> Path:
    """Build and write a COCO GT JSON file to disk."""
    coco_dict, _ = build_coco_gt_dict(split=split, subset_keys=subset_keys, cache_root=cache_root)
    return save_json(coco_dict, out_path)

