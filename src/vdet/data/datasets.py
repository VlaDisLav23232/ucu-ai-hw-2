"""PyTorch Dataset and DataLoader builders reading offline caches for Torchvision models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vdet.data.augment import DetectionAugmentor


def _parse_yolo_label_file(
    label_path: Path,
    width: int,
    height: int,
    label_offset: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a cached YOLO `.txt` label file (`cls cx cy w h` in `[0,1]`) into `(xyxy, labels)` arrays.

    Note: Torchvision Faster R-CNN and RetinaNet reserve `0` for background and expect
    1-indexed foreground class IDs `1..8` (`label_offset=1`, matching `YOLO_ID + 1`).
    """
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes: list[list[float]] = []
    labels: list[int] = []
    w_f = float(width)
    h_f = float(height)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0]) + int(label_offset)
        cx = float(parts[1]) * w_f
        cy = float(parts[2]) * h_f
        bw = float(parts[3]) * w_f
        bh = float(parts[4]) * h_f
        x1 = max(0.0, min(cx - 0.5 * bw, w_f))
        y1 = max(0.0, min(cy - 0.5 * bh, h_f))
        x2 = max(0.0, min(cx + 0.5 * bw, w_f))
        y2 = max(0.0, min(cy + 0.5 * bh, h_f))
        if (x2 - x1) >= 1.0 and (y2 - y1) >= 1.0:
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class VisDroneCacheDataset(Dataset):
    """Torchvision-compatible Dataset over `cache/<cache_name>/{train,val}/{images,labels}`."""

    def __init__(
        self,
        cache_dir: Path,
        split: str = "train",
        augmentor: DetectionAugmentor | None = None,
        stride: int = 1,
        max_images: int | None = None,
        label_offset: int = 1,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.augmentor = augmentor
        self.label_offset = int(label_offset)

        img_dir = self.cache_dir / split / "images"
        lbl_dir = self.cache_dir / split / "labels"
        if not img_dir.exists():
            raise FileNotFoundError(f"Cache images dir not found: {img_dir}")

        imgs = sorted(img_dir.glob("*.jpg"))
        if stride > 1:
            imgs = imgs[::stride]
        if max_images is not None:
            imgs = imgs[: int(max_images)]

        self.image_paths = imgs
        self.label_paths = [lbl_dir / f"{p.stem}.txt" for p in imgs]
        self.image_keys = [p.stem for p in imgs]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]

        bgr = cv2.imread(img_path.as_posix(), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read cached image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        boxes, labels = _parse_yolo_label_file(lbl_path, width=w, height=h, label_offset=self.label_offset)
        if self.augmentor is not None:
            rgb, boxes, labels = self.augmentor(rgb, boxes, labels)

        img_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        boxes_t = torch.from_numpy(boxes).to(dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.from_numpy(labels).to(dtype=torch.int64).reshape(-1)
        area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]) if len(boxes_t) else torch.zeros((0,), dtype=torch.float32)

        target: dict[str, Any] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": idx,
            "image_key": self.image_keys[idx],
            "area": area_t,
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
        }
        return img_tensor, target


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Standard Torchvision detection collate returning `(images_list, targets_list)`."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def build_tv_dataloaders(
    cache_dir: Path,
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from a cache directory and run config."""
    batch_size = int(config.get("batch", 4))
    workers = int(config.get("workers", 4))
    stride = int(config.get("train_subset_stride", 1))
    max_train = config.get("max_train_images", None)
    val_on_train = bool(config.get("val_on_train", False))

    train_aug = DetectionAugmentor.from_config(config, is_train=not val_on_train)
    val_aug = DetectionAugmentor.from_config(config, is_train=False)

    train_ds = VisDroneCacheDataset(
        cache_dir=cache_dir,
        split="train",
        augmentor=train_aug,
        stride=stride,
        max_images=int(max_train) if max_train is not None else None,
    )
    if val_on_train:
        val_ds = VisDroneCacheDataset(
            cache_dir=cache_dir,
            split="train",
            augmentor=val_aug,
            stride=stride,
            max_images=int(max_train) if max_train is not None else None,
        )
    else:
        max_val = config.get("max_val_images", None)
        val_ds = VisDroneCacheDataset(
            cache_dir=cache_dir,
            split="val",
            augmentor=val_aug,
            stride=1,
            max_images=int(max_val) if max_val is not None else None,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader

