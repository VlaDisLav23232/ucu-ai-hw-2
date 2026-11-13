"""COCO evaluation wrapper using faster-coco-eval (and pycocotools cross-check)."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from vdet.constants import CLASS_NAMES, DEFAULT_MAX_DETS, NUM_CLASSES, YOLO_TO_COCO_ID
from vdet.data.gt import build_coco_gt_dict
from vdet.paths import P


def _preds_df_to_coco_list(
    preds_df: pd.DataFrame,
    key_to_id: dict[str, int],
    max_det_per_image: int = 500,
) -> list[dict[str, Any]]:
    """Convert internal prediction table to COCO detection list sorted by score."""
    if preds_df is None or preds_df.empty:
        return []

    sub = preds_df[preds_df["image_key"].isin(key_to_id)].copy()
    if sub.empty:
        return []

    # Discard invalid boxes w <= 0 or h <= 0
    sub = sub[(sub["w"] > 0.0) & (sub["h"] > 0.0) & (sub["score"] > 0.0)]
    if sub.empty:
        return []

    # Sort descending by score per image and cap at max_det_per_image
    sub = sub.sort_values(["image_key", "score"], ascending=[True, False])
    if max_det_per_image > 0:
        sub = sub.groupby("image_key", sort=False).head(max_det_per_image)

    coco_dts: list[dict[str, Any]] = []
    img_ids = sub["image_key"].map(key_to_id).to_numpy(dtype=np.int64)
    cls_ids = sub["cls"].to_numpy(dtype=np.int64)
    scores = sub["score"].to_numpy(dtype=np.float64)
    xs = sub["x"].to_numpy(dtype=np.float64)
    ys = sub["y"].to_numpy(dtype=np.float64)
    ws = sub["w"].to_numpy(dtype=np.float64)
    hs = sub["h"].to_numpy(dtype=np.float64)

    for img_id, cls_id, sc, x, y, w, h in zip(img_ids, cls_ids, scores, xs, ys, ws, hs):
        if int(cls_id) not in YOLO_TO_COCO_ID:
            continue
        coco_dts.append(
            {
                "image_id": int(img_id),
                "category_id": YOLO_TO_COCO_ID[int(cls_id)],
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(sc),
            }
        )
    return coco_dts


def _run_coco_eval_core(
    coco_gt_dict: dict[str, Any],
    coco_dts: list[dict[str, Any]],
    max_dets: Sequence[int] = DEFAULT_MAX_DETS,
    backend: str = "faster_coco_eval",
) -> dict[str, Any]:
    """Run COCO evaluation on a prepared GT dict and detection list."""
    max_dets_list = sorted(int(m) for m in max_dets)
    max_det_cap = max_dets_list[-1]

    zero_result: dict[str, Any] = {
        "AP": 0.0,
        "AP50": 0.0,
        "AP75": 0.0,
        "APs": 0.0,
        "APm": 0.0,
        "APl": 0.0,
        "AR1": 0.0,
        "AR10": 0.0,
        f"AR{max_det_cap}": 0.0,
        "AR500": 0.0,
        **{f"ap_{name}": 0.0 for name in CLASS_NAMES},
    }

    if len(coco_gt_dict.get("images", [])) == 0 or len(coco_dts) == 0:
        return zero_result

    if backend == "faster_coco_eval":
        from faster_coco_eval import COCO, COCOeval_faster

        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(coco_gt_dict)
            coco_dt = coco_gt.loadRes(coco_dts)
            coco_eval = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
            coco_eval.params.maxDets = list(max_dets_list)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
    elif backend == "pycocotools":
        from pycocotools.coco import COCO as PyCOCO
        from pycocotools.cocoeval import COCOeval as PyCOCOeval

        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = PyCOCO()
            coco_gt.dataset = coco_gt_dict
            coco_gt.createIndex()
            coco_dt = coco_gt.loadRes(coco_dts)
            coco_eval = PyCOCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.params.maxDets = list(max_dets_list)
            coco_eval.evaluate()
            coco_eval.accumulate()
            # Note: pycocotools' built-in summarize() hardcodes maxDets=100 for stats[0..5].
            # Compute the 12 standard summary stats using maxDets=max_dets_list[-1] to match faster_coco_eval.
            p = coco_eval.params
            prec = coco_eval.eval["precision"]
            rec = coco_eval.eval["recall"]

            def _py_sum(ap: int = 1, iou_thr: float | None = None, area_lbl: str = "all", max_dets_val: int = max_det_cap) -> float:
                aind = [i for i, a in enumerate(p.areaRngLbl) if a == area_lbl]
                mind = [i for i, m in enumerate(p.maxDets) if m == max_dets_val]
                if ap == 1:
                    s = prec
                    if iou_thr is not None:
                        t = np.where(np.isclose(iou_thr, p.iouThrs))[0]
                        s = s[t]
                    s = s[:, :, :, aind, mind]
                else:
                    s = rec
                    if iou_thr is not None:
                        t = np.where(np.isclose(iou_thr, p.iouThrs))[0]
                        s = s[t]
                    s = s[:, :, aind, mind]
                valid_s = s[s > -1]
                return float(np.mean(valid_s)) if valid_s.size > 0 else -1.0

            coco_eval.stats = np.array(
                [
                    _py_sum(1, None, "all", max_det_cap),
                    _py_sum(1, 0.50, "all", max_det_cap),
                    _py_sum(1, 0.75, "all", max_det_cap),
                    _py_sum(1, None, "small", max_det_cap),
                    _py_sum(1, None, "medium", max_det_cap),
                    _py_sum(1, None, "large", max_det_cap),
                    _py_sum(0, None, "all", max_dets_list[0]),
                    _py_sum(0, None, "all", max_dets_list[1]),
                    _py_sum(0, None, "all", max_dets_list[2]),
                    _py_sum(0, None, "small", max_det_cap),
                    _py_sum(0, None, "medium", max_det_cap),
                    _py_sum(0, None, "large", max_det_cap),
                ],
                dtype=np.float64,
            )
    else:
        raise ValueError(f"Unknown backend '{backend}'; expected 'faster_coco_eval' or 'pycocotools'.")

    stats = [float(x) for x in coco_eval.stats]
    # Standard 12 COCO stats order:
    # 0: AP@[.5:.95], 1: AP50, 2: AP75, 3: APs, 4: APm, 5: APl,
    # 6: AR@maxDets[0], 7: AR@maxDets[1], 8: AR@maxDets[2], 9: ARs, 10: ARm, 11: ARl
    def _clean(v: float) -> float:
        return round(float(v), 6) if v >= 0.0 else 0.0

    metrics: dict[str, Any] = {
        "AP": _clean(stats[0]),
        "AP50": _clean(stats[1]),
        "AP75": _clean(stats[2]),
        "APs": _clean(stats[3]),
        "APm": _clean(stats[4]),
        "APl": _clean(stats[5]),
        "AR1": _clean(stats[6]),
        "AR10": _clean(stats[7]),
        f"AR{max_det_cap}": _clean(stats[8]),
        "AR500": _clean(stats[8]),
        "ARs": _clean(stats[9]),
        "ARm": _clean(stats[10]),
        "APl": _clean(stats[11]),
    }

    # Extract per-class AP@[0.50:0.95] at areaRng='all' (index 0) and maxDets=max_det_cap (index -1)
    # precision has shape [T, R, K, A, M]
    precisions = coco_eval.eval["precision"]
    cat_ids = list(coco_eval.params.catIds)
    for k_idx, cat_id in enumerate(cat_ids):
        cls_idx = int(cat_id) - 1
        if 0 <= cls_idx < NUM_CLASSES:
            cls_name = CLASS_NAMES[cls_idx]
            p_slice = precisions[:, :, k_idx, 0, -1]
            valid = p_slice[p_slice > -1]
            cls_ap = float(np.mean(valid)) if valid.size > 0 else 0.0
            metrics[f"ap_{cls_name}"] = round(cls_ap, 6)

    return metrics


def evaluate(
    preds_df: pd.DataFrame,
    split: str = "val",
    subset: Sequence[str] | set[str] | None = None,
    max_dets: Sequence[int] = DEFAULT_MAX_DETS,
    backend: str = "faster_coco_eval",
    per_res_bucket: bool = False,
    manifest_df: pd.DataFrame | None = None,
    boxes_df: pd.DataFrame | None = None,
    ignore_df: pd.DataFrame | None = None,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a prediction DataFrame against COCO ground truth with deduplicated crowd.

    Parameters
    ----------
    preds_df : pd.DataFrame
        Prediction table with columns `image_key, cls, score, x, y, w, h` in original pixels.
    split : str
        Dataset split to evaluate (`'val'` or `'train'`).
    subset : Sequence[str] | set[str] | None
        Optional list/set of `image_key` strings to restrict evaluation to a subset.
    max_dets : Sequence[int]
        COCO `maxDets` parameter; defaults to `[1, 10, 500]` per     backend : str
        `'faster_coco_eval'` (default) or `'pycocotools'`.
    per_res_bucket : bool
        If True, also compute AP/AP50/APs per resolution bucket (`res_bucket`).
    """
    coco_gt_dict, key_to_id = build_coco_gt_dict(
        split=split,
        subset_keys=subset,
        manifest_df=manifest_df,
        boxes_df=boxes_df,
        ignore_df=ignore_df,
        cache_root=cache_root,
    )
    max_det_cap = max(int(m) for m in max_dets)
    coco_dts = _preds_df_to_coco_list(preds_df, key_to_id, max_det_per_image=max_det_cap)

    results = _run_coco_eval_core(
        coco_gt_dict=coco_gt_dict,
        coco_dts=coco_dts,
        max_dets=max_dets,
        backend=backend,
    )
    results["split"] = split
    results["n_images"] = len(coco_gt_dict["images"])
    results["n_detections"] = len(coco_dts)
    results["max_dets"] = list(max_dets)
    results["backend"] = backend

    if per_res_bucket and len(coco_gt_dict["images"]) > 0:
        bucket_metrics: dict[str, dict[str, float]] = {}
        by_bucket: dict[str, list[str]] = {}
        for im in coco_gt_dict["images"]:
            by_bucket.setdefault(str(im["res_bucket"]), []).append(str(im["image_key"]))
        for bucket, keys in sorted(by_bucket.items()):
            b_gt, b_map = build_coco_gt_dict(
                split=split,
                subset_keys=keys,
                manifest_df=manifest_df,
                boxes_df=boxes_df,
                ignore_df=ignore_df,
                cache_root=cache_root,
            )
            b_dts = _preds_df_to_coco_list(preds_df, b_map, max_det_per_image=max_det_cap)
            b_res = _run_coco_eval_core(b_gt, b_dts, max_dets=max_dets, backend=backend)
            bucket_metrics[bucket] = {
                "n_images": len(keys),
                "AP": b_res["AP"],
                "AP50": b_res["AP50"],
                "APs": b_res["APs"],
                "AR500": b_res["AR500"],
            }
        results["per_res_bucket"] = bucket_metrics

    return results


evaluate_predictions_df = evaluate

