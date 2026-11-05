"""Model-agnostic SAHI-style sliced inference + full-frame merge (§10 U4.1).

Credits: Inspired by SAHI (Akyon et al., "Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection", ICIP 2022).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from vdet.constants import MAX_SUBMISSION_DETS
from vdet.data.manifest import load_manifest
from vdet.infer.fusion import fuse_predictions
from vdet.infer.predictors import (
    Predictor,
    clip_and_filter_boxes_xywh,
    empty_predictions_df,
    normalize_predictions_df,
)
from vdet.paths import P


def compute_slice_windows(
    width: int,
    height: int,
    slice_w: int = 800,
    slice_h: int = 800,
    overlap_ratio: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Return `(x1, y1, x2, y2)` tile windows covering a `width x height` frame with `overlap_ratio`."""
    sw = min(int(slice_w), int(width))
    sh = min(int(slice_h), int(height))
    step_x = max(1, int(round(sw * (1.0 - float(overlap_ratio)))))
    step_y = max(1, int(round(sh * (1.0 - float(overlap_ratio)))))

    xs: list[int] = []
    x = 0
    while x + sw < width:
        xs.append(x)
        x += step_x
    xs.append(max(0, width - sw))
    xs = sorted(set(xs))

    ys: list[int] = []
    y = 0
    while y + sh < height:
        ys.append(y)
        y += step_y
    ys.append(max(0, height - sh))
    ys = sorted(set(ys))

    return [(x0, y0, x0 + sw, y0 + sh) for y0 in ys for x0 in xs]


class TiledPredictor(Predictor):
    """Wraps any `Predictor` to run full-frame + overlapping tile passes and fuse them via WBF/NMS (§10 U4.1).

    Small frames (`min(W, H) < slice_size`, e.g., `680x382` or `960x540` test clips) are optionally
    upscaled `x2` before slicing so small objects occupy sufficient receptive field.
    """

    def __init__(
        self,
        base_predictor: Predictor,
        slice_size: int = 800,
        overlap_ratio: float = 0.25,
        upscale_small_frames: bool = True,
        merge_method: str = "wbf",
        merge_iou: float = 0.55,
        full_frame_weight: float = 1.5,
        tile_weight: float = 1.0,
        conf_thresh: float = 0.001,
        max_det: int = MAX_SUBMISSION_DETS,
        source_tag: str = "tiled",
    ) -> None:
        self.base_predictor = base_predictor
        self.slice_size = int(slice_size)
        self.overlap_ratio = float(overlap_ratio)
        self.upscale_small_frames = bool(upscale_small_frames)
        self.merge_method = merge_method
        self.merge_iou = float(merge_iou)
        self.full_frame_weight = float(full_frame_weight)
        self.tile_weight = float(tile_weight)
        self.conf_thresh = float(conf_thresh)
        self.max_det = int(max_det)
        self.source_tag = source_tag

    def predict(
        self,
        frames: pd.DataFrame | Sequence[str],
        data_root: Path | None = None,
    ) -> pd.DataFrame:
        d_root = data_root or P.data
        if isinstance(frames, pd.DataFrame):
            m_sub = frames.reset_index(drop=True)
        else:
            manifest_df = load_manifest()
            m_sub = manifest_df[manifest_df["image_key"].isin(set(frames))].reset_index(drop=True)

        if m_sub.empty:
            return empty_predictions_df()

        # 1. Full-frame pass
        full_preds = self.base_predictor.predict(m_sub, data_root=d_root)

        # 2. Sliced pass using a temporary directory for cropped tiles
        tile_rows: list[dict[str, object]] = []
        tile_meta: dict[str, tuple[str, int, int, float, float, float]] = {}

        with tempfile.TemporaryDirectory(prefix="vdet_tiles_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            for r in m_sub.itertuples(index=False):
                img_path = d_root / str(r.path)
                bgr = cv2.imread(img_path.as_posix(), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                orig_h, orig_w = bgr.shape[:2]
                scale = 1.0
                if self.upscale_small_frames and min(orig_w, orig_h) < self.slice_size:
                    scale = 2.0
                    bgr = cv2.resize(
                        bgr,
                        (int(orig_w * scale), int(orig_h * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )

                work_h, work_w = bgr.shape[:2]
                windows = compute_slice_windows(
                    work_w,
                    work_h,
                    slice_w=self.slice_size,
                    slice_h=self.slice_size,
                    overlap_ratio=self.overlap_ratio,
                )
                for t_idx, (x1, y1, x2, y2) in enumerate(windows):
                    crop = bgr[y1:y2, x1:x2]
                    t_key = f"{r.image_key}__tile{t_idx}"
                    rel_path = f"{t_key}.jpg"
                    cv2.imwrite((tmp_dir / rel_path).as_posix(), crop)
                    tile_rows.append(
                        {
                            "image_key": t_key,
                            "path": rel_path,
                            "width": int(x2 - x1),
                            "height": int(y2 - y1),
                        }
                    )
                    tile_meta[t_key] = (str(r.image_key), x1, y1, scale, float(orig_w), float(orig_h))

            if not tile_rows:
                return full_preds

            tile_manifest = pd.DataFrame(tile_rows)
            raw_tile_preds = self.base_predictor.predict(tile_manifest, data_root=tmp_dir)

        if raw_tile_preds.empty:
            return full_preds

        # Map tile predictions back to original frame coordinates
        mapped_frames: list[pd.DataFrame] = []
        for t_key, sub in raw_tile_preds.groupby("image_key", sort=False):
            meta = tile_meta.get(str(t_key))
            if meta is None:
                continue
            orig_key, off_x, off_y, scale, orig_w, orig_h = meta
            raw_xywh = np.stack(
                [
                    (sub["x"].to_numpy(dtype=np.float64) + off_x) / scale,
                    (sub["y"].to_numpy(dtype=np.float64) + off_y) / scale,
                    sub["w"].to_numpy(dtype=np.float64) / scale,
                    sub["h"].to_numpy(dtype=np.float64) / scale,
                ],
                axis=1,
            )
            clipped_xywh, valid = clip_and_filter_boxes_xywh(raw_xywh, orig_w, orig_h)
            if not np.any(valid):
                continue
            mapped_frames.append(
                pd.DataFrame(
                    {
                        "image_key": orig_key,
                        "cls": sub["cls"].to_numpy(dtype=np.int8)[valid],
                        "score": sub["score"].to_numpy(dtype=np.float32)[valid],
                        "x": clipped_xywh[:, 0],
                        "y": clipped_xywh[:, 1],
                        "w": clipped_xywh[:, 2],
                        "h": clipped_xywh[:, 3],
                        "source": self.source_tag,
                    }
                )
            )

        if not mapped_frames:
            return full_preds

        tile_preds_df = normalize_predictions_df(
            pd.concat(mapped_frames, ignore_index=True),
            conf_thresh=self.conf_thresh,
            max_det_per_image=self.max_det * 2,
        )

        return fuse_predictions(
            preds_list=[full_preds, tile_preds_df],
            weights=[self.full_frame_weight, self.tile_weight],
            method=self.merge_method,
            iou_thr=self.merge_iou,
            skip_box_thr=self.conf_thresh,
            max_det=self.max_det,
            manifest_df=m_sub,
            source_tag=self.source_tag,
        )

