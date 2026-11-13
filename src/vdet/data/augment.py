"""Bbox-safe augmentations for Torchvision detection models."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class DetectionAugmentor:
    """Bbox-safe augmentation pipeline for Torchvision models under Protocol P0.

    Applies:
      - Random horizontal flip (`hflip = 0.5`)
      - Downscale-biased scale jitter (`scale_min = 0.5`, `scale_max = 1.0`)
      - Light HSV jitter (`hsv_h = 0.015`, `hsv_s = 0.4`, `hsv_v = 0.3`)
      - Strictly NO vertical flip or rotation.
    """

    def __init__(
        self,
        hflip: float = 0.5,
        scale_min: float = 0.5,
        scale_max: float = 1.0,
        hsv_h: float = 0.015,
        hsv_s: float = 0.4,
        hsv_v: float = 0.3,
        is_train: bool = True,
    ) -> None:
        self.hflip = float(hflip)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.is_train = bool(is_train)

    @classmethod
    def from_config(cls, config: dict[str, Any], is_train: bool = True) -> "DetectionAugmentor":
        aug = config.get("augmentation", {}) if isinstance(config.get("augmentation"), dict) else {}
        return cls(
            hflip=float(aug.get("hflip", 0.5)),
            scale_min=float(aug.get("scale_min", 0.5)),
            scale_max=float(aug.get("scale_max", 1.0)),
            hsv_h=float(aug.get("hsv_h", 0.015)),
            hsv_s=float(aug.get("hsv_s", 0.4)),
            hsv_v=float(aug.get("hsv_v", 0.3)),
            is_train=is_train,
        )

    def __call__(
        self,
        img_rgb: np.ndarray,
        boxes_xyxy: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Augment an `H x W x 3` uint8 RGB image and `(N, 4)` float32 `xyxy` boxes."""
        if not self.is_train:
            return img_rgb, boxes_xyxy, labels

        gen = rng if rng is not None else np.random.default_rng()
        h, w = img_rgb.shape[:2]
        out_img = img_rgb.copy()
        out_boxes = boxes_xyxy.astype(np.float32, copy=True)
        out_labels = labels.copy()

        # 1. Horizontal flip
        if self.hflip > 0.0 and gen.random() < self.hflip:
            out_img = np.ascontiguousarray(out_img[:, ::-1, :])
            if len(out_boxes) > 0:
                x1 = out_boxes[:, 0].copy()
                x2 = out_boxes[:, 2].copy()
                out_boxes[:, 0] = float(w) - x2
                out_boxes[:, 2] = float(w) - x1

        # 2. Downscale-biased scale jitter
        if self.scale_min < self.scale_max and gen.random() < 0.5:
            scale = float(gen.uniform(self.scale_min, self.scale_max))
            new_w = max(64, int(round(w * scale)))
            new_h = max(64, int(round(h * scale)))
            out_img = cv2.resize(out_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if len(out_boxes) > 0:
                sx = new_w / float(w)
                sy = new_h / float(h)
                out_boxes[:, [0, 2]] *= sx
                out_boxes[:, [1, 3]] *= sy
            h, w = new_h, new_w

        # 3. Light HSV jitter
        if (self.hsv_h > 0 or self.hsv_s > 0 or self.hsv_v > 0) and gen.random() < 0.5:
            r = gen.uniform(-1.0, 1.0, 3) * np.array([self.hsv_h, self.hsv_s, self.hsv_v]) + 1.0
            hsv = cv2.cvtColor(out_img, cv2.COLOR_RGB2HSV)
            hue, sat, val = cv2.split(hsv)
            dtype = out_img.dtype
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            hsv_aug = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            out_img = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2RGB)

        # Clip and filter degenerate boxes
        if len(out_boxes) > 0:
            out_boxes[:, [0, 2]] = np.clip(out_boxes[:, [0, 2]], 0.0, float(w))
            out_boxes[:, [1, 3]] = np.clip(out_boxes[:, [1, 3]], 0.0, float(h))
            keep = (out_boxes[:, 2] - out_boxes[:, 0] >= 1.0) & (out_boxes[:, 3] - out_boxes[:, 1] >= 1.0)
            out_boxes = out_boxes[keep]
            out_labels = out_labels[keep]

        return out_img, out_boxes, out_labels

