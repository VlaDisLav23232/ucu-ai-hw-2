"""Model-agnostic Test-Time Augmentation (TTA: horizontal flip + multi-scale) with WBF merge (§10 U4.2)."""

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


class TTAPredictor(Predictor):
    """Wraps any `Predictor` to run multi-scale (`scales=(1.0, 1.25)`) + horizontal-flip TTA and fuse via WBF (§10 U4.2)."""

    def __init__(
        self,
        base_predictor: Predictor,
        scales: Sequence[float] = (1.0, 1.25),
        hflip: bool = True,
        merge_iou: float = 0.60,
        conf_thresh: float = 0.001,
        max_det: int = MAX_SUBMISSION_DETS,
        source_tag: str = "tta",
    ) -> None:
        self.base_predictor = base_predictor
        self.scales = [float(s) for s in scales]
        self.hflip = bool(hflip)
        self.merge_iou = float(merge_iou)
        self.conf_thresh = float(conf_thresh)
        self.max_det = int(max_det)
        self.source_tag = source_tag

    def _run_transformed_pass(
        self,
        m_sub: pd.DataFrame,
        d_root: Path,
        scale: float,
        flip: bool,
    ) -> pd.DataFrame:
        if abs(scale - 1.0) < 1e-4 and not flip:
            return self.base_predictor.predict(m_sub, data_root=d_root)

        aug_rows: list[dict[str, object]] = []
        meta_map: dict[str, tuple[float, float, float, bool]] = {}

        with tempfile.TemporaryDirectory(prefix="vdet_tta_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            for r in m_sub.itertuples(index=False):
                img_path = d_root / str(r.path)
                bgr = cv2.imread(img_path.as_posix(), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                orig_h, orig_w = bgr.shape[:2]
                work = bgr
                if abs(scale - 1.0) >= 1e-4:
                    work = cv2.resize(
                        work,
                        (max(32, int(round(orig_w * scale))), max(32, int(round(orig_h * scale)))),
                        interpolation=cv2.INTER_LINEAR,
                    )
                if flip:
                    work = np.ascontiguousarray(work[:, ::-1, :])

                work_h, work_w = work.shape[:2]
                rel_path = f"{r.image_key}.jpg"
                cv2.imwrite((tmp_dir / rel_path).as_posix(), work)
                aug_rows.append(
                    {
                        "image_key": str(r.image_key),
                        "path": rel_path,
                        "width": int(work_w),
                        "height": int(work_h),
                    }
                )
                meta_map[str(r.image_key)] = (float(orig_w), float(orig_h), float(work_w), flip)

            if not aug_rows:
                return empty_predictions_df()

            aug_manifest = pd.DataFrame(aug_rows)
            raw_preds = self.base_predictor.predict(aug_manifest, data_root=tmp_dir)

        if raw_preds.empty:
            return empty_predictions_df()

        mapped: list[pd.DataFrame] = []
        for key, sub in raw_preds.groupby("image_key", sort=False):
            meta = meta_map.get(str(key))
            if meta is None:
                continue
            orig_w, orig_h, work_w, is_flipped = meta
            x = sub["x"].to_numpy(dtype=np.float64)
            y = sub["y"].to_numpy(dtype=np.float64)
            w = sub["w"].to_numpy(dtype=np.float64)
            h = sub["h"].to_numpy(dtype=np.float64)
            if is_flipped:
                x = work_w - (x + w)
            raw_xywh = np.stack([x / scale, y / scale, w / scale, h / scale], axis=1)
            clipped_xywh, valid = clip_and_filter_boxes_xywh(raw_xywh, orig_w, orig_h)
            if not np.any(valid):
                continue
            mapped.append(
                pd.DataFrame(
                    {
                        "image_key": str(key),
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

        if not mapped:
            return empty_predictions_df()
        return normalize_predictions_df(
            pd.concat(mapped, ignore_index=True),
            conf_thresh=self.conf_thresh,
            max_det_per_image=self.max_det,
        )

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

        passes: list[pd.DataFrame] = []
        weights: list[float] = []
        for s in self.scales:
            passes.append(self._run_transformed_pass(m_sub, d_root, scale=s, flip=False))
            weights.append(1.2 if abs(s - 1.0) < 1e-4 else 1.0)
            if self.hflip:
                passes.append(self._run_transformed_pass(m_sub, d_root, scale=s, flip=True))
                weights.append(1.0)

        return fuse_predictions(
            preds_list=passes,
            weights=weights,
            method="wbf",
            iou_thr=self.merge_iou,
            skip_box_thr=self.conf_thresh,
            max_det=self.max_det,
            manifest_df=m_sub,
            source_tag=self.source_tag,
        )

