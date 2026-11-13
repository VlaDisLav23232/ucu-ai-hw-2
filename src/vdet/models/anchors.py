"""K-means anchor fitting (1 - IoU distance) on 1280-scale train boxes for RetinaNet / Faster R-CNN."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vdet.data.manifest import load_boxes, load_manifest
from vdet.paths import P
from vdet.utils import load_yaml, save_yaml


def _wh_iou_matrix(wh1: np.ndarray, wh2: np.ndarray) -> np.ndarray:
    """Compute `(N, K)` origin-aligned width-height IoU between `wh1 (N, 2)` and `wh2 (K, 2)`."""
    w1, h1 = wh1[:, 0:1], wh1[:, 1:2]
    w2, h2 = wh2[None, :, 0], wh2[None, :, 1]
    inter = np.minimum(w1, w2) * np.minimum(h1, h2)
    union = (w1 * h1) + (w2 * h2) - inter + 1e-9
    return inter / union


def fit_kmeans_anchors(
    boxes_df: pd.DataFrame | None = None,
    manifest_df: pd.DataFrame | None = None,
    target_long_side: float = 1280.0,
    n_levels: int = 5,
    scales_per_level: int = 3,
    ratios_per_level: int = 3,
    max_samples: int = 120_000,
    max_iter: int = 100,
    seed: int = 42,
    use_cached_yaml: bool = True,
) -> dict[str, Any]:
    """Fit 5 FPN levels x 3 sizes x 3 aspect ratios via k-means (1 - IoU distance) on 1280-scale train boxes.

    Keeps `scales_per_level * ratios_per_level = 9` anchors per spatial location so
    COCO-pretrained RetinaNet v2 head weights remain shape-compatible.
    """
    cache_yaml = P.cache / f"kmeans_anchors_L{int(target_long_side)}.yaml"
    if use_cached_yaml and cache_yaml.exists() and boxes_df is None:
        cached = load_yaml(cache_yaml)
        cached["sizes"] = tuple(tuple(int(round(s)) for s in lvl) for lvl in cached["sizes"])
        cached["aspect_ratios"] = tuple(tuple(float(r) for r in lvl) for lvl in cached["aspect_ratios"])
        return cached

    b_df = boxes_df if boxes_df is not None else load_boxes()
    m_df = manifest_df if manifest_df is not None else load_manifest()

    train_m = m_df[m_df["split"] == "train"][["image_key", "width", "height"]]
    merged = b_df[b_df["split"] == "train"][["image_key", "w", "h"]].merge(
        train_m, on="image_key", how="inner"
    )
    long_side = np.maximum(merged["width"].to_numpy(dtype=np.float64), merged["height"].to_numpy(dtype=np.float64))
    scale = float(target_long_side) / np.maximum(long_side, 1.0)
    w_scaled = merged["w"].to_numpy(dtype=np.float64) * scale
    h_scaled = merged["h"].to_numpy(dtype=np.float64) * scale
    valid = (w_scaled >= 2.0) & (h_scaled >= 2.0)
    wh = np.stack([w_scaled[valid], h_scaled[valid]], axis=1)

    rng = np.random.default_rng(seed)
    if len(wh) > max_samples:
        idx = rng.choice(len(wh), size=max_samples, replace=False)
        wh = wh[idx]

    # 1. Fit K_s = n_levels * scales_per_level (15) 2D (w, h) centroids with 1 - IoU distance
    k_total = n_levels * scales_per_level
    areas = wh[:, 0] * wh[:, 1]
    quantiles = np.linspace(0.02, 0.98, k_total)
    init_idx = [int(np.argmin(np.abs(areas - np.quantile(areas, q)))) for q in quantiles]
    centroids = wh[init_idx].copy()

    for _ in range(max_iter):
        ious = _wh_iou_matrix(wh, centroids)
        assign = np.argmax(ious, axis=1)
        new_centroids = centroids.copy()
        for k_idx in range(k_total):
            mask = assign == k_idx
            if np.any(mask):
                new_centroids[k_idx] = np.median(wh[mask], axis=0)
        if np.allclose(new_centroids, centroids, rtol=1e-4):
            break
        centroids = new_centroids

    # Sort centroids by linear scale sqrt(w * h) and group into 5 FPN levels x 3 scales
    scales_1d = np.sqrt(centroids[:, 0] * centroids[:, 1])
    scales_sorted = np.sort(scales_1d)
    sizes_tuple: tuple[tuple[int, ...], ...] = tuple(
        tuple(max(4, int(round(float(s)))) for s in scales_sorted[lvl * scales_per_level : (lvl + 1) * scales_per_level])
        for lvl in range(n_levels)
    )

    # 2. Fit 3 representative aspect ratios (h / w in torchvision AnchorGenerator convention)
    ratios_raw = np.clip(wh[:, 1] / np.maximum(wh[:, 0], 1e-3), 0.25, 4.0)
    r_quantiles = np.quantile(ratios_raw, [0.18, 0.50, 0.82])
    aspect_ratios_single = tuple(round(float(r), 2) for r in r_quantiles)
    aspect_ratios_tuple: tuple[tuple[float, ...], ...] = tuple(
        aspect_ratios_single for _ in range(n_levels)
    )

    # Compute mean best IoU before vs after
    default_sizes = ((32, 40, 50), (64, 80, 101), (128, 161, 203), (256, 322, 406), (512, 645, 812))
    default_ar = (0.5, 1.0, 2.0)

    def _build_anchor_wh(sz_groups, ar_groups) -> np.ndarray:
        rows = []
        for sz_lvl, ar_lvl in zip(sz_groups, ar_groups):
            for s in sz_lvl:
                for ar in ar_lvl:
                    # torchvision AnchorGenerator: w = s / sqrt(ar), h = s * sqrt(ar)
                    rows.append([s / np.sqrt(ar), s * np.sqrt(ar)])
        return np.asarray(rows, dtype=np.float64)

    default_wh = _build_anchor_wh(default_sizes, tuple(default_ar for _ in range(n_levels)))
    fitted_wh = _build_anchor_wh(sizes_tuple, aspect_ratios_tuple)
    mean_iou_default = float(_wh_iou_matrix(wh, default_wh).max(axis=1).mean())
    mean_iou_fitted = float(_wh_iou_matrix(wh, fitted_wh).max(axis=1).mean())

    result = {
        "sizes": sizes_tuple,
        "aspect_ratios": aspect_ratios_tuple,
        "mean_best_iou_default": round(mean_iou_default, 4),
        "mean_best_iou_fitted": round(mean_iou_fitted, 4),
        "smallest_anchor_px": int(sizes_tuple[0][0]),
        "largest_anchor_px": int(sizes_tuple[-1][-1]),
    }

    if use_cached_yaml and boxes_df is None:
        serializable = {
            **result,
            "sizes": [list(lvl) for lvl in sizes_tuple],
            "aspect_ratios": [list(lvl) for lvl in aspect_ratios_tuple],
        }
        save_yaml(serializable, cache_yaml)

    return result
