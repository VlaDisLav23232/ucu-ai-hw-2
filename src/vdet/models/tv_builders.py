"""Torchvision Faster R-CNN v2 and RetinaNet v2 model builders and Predictor."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models.detection as tvd
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FastRCNNPredictor,
)
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights

from vdet.constants import MAX_SUBMISSION_DETS, NUM_CLASSES
from vdet.data.manifest import load_manifest
from vdet.infer.predictors import (
    Predictor,
    clip_and_filter_boxes_xywh,
    empty_predictions_df,
    normalize_predictions_df,
)
from vdet.models.anchors import fit_kmeans_anchors
from vdet.models.tv_patches import (
    patch_frcnn_giou_loss,
    patch_retinanet_focal_and_matcher,
)
from vdet.paths import P


def build_faster_rcnn(
    config: dict[str, Any],
    pretrained: bool = True,
) -> torch.nn.Module:
    """Build `fasterrcnn_resnet50_fpn_v2` for VisDrone (8 foreground classes + 1 background = 9)."""
    num_classes = int(config.get("num_classes", NUM_CLASSES + 1))
    min_size = int(config.get("min_size", 720))
    max_size = int(config.get("max_size", 1280))
    inf_cfg = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}

    box_detections_per_img = int(config.get("box_detections_per_img", inf_cfg.get("max_det", MAX_SUBMISSION_DETS)))
    box_score_thresh = float(config.get("box_score_thresh", inf_cfg.get("conf_thresh", 0.001)))
    box_nms_thresh = float(config.get("box_nms_thresh", inf_cfg.get("nms_iou", 0.6)))

    rpn_pre_train = int(config.get("rpn_pre_nms_top_n_train", 2000))
    rpn_pre_test = int(config.get("rpn_pre_nms_top_n_test", 1000))
    rpn_post_train = int(config.get("rpn_post_nms_top_n_train", 2000))
    rpn_post_test = int(config.get("rpn_post_nms_top_n_test", 1000))
    box_batch_size_per_image = int(config.get("box_batch_size_per_image", 512))
    box_positive_fraction = float(config.get("box_positive_fraction", 0.25))
    trainable_backbone_layers = int(config.get("trainable_backbone_layers", 3))

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = tvd.fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        box_detections_per_img=box_detections_per_img,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        rpn_pre_nms_top_n_train=rpn_pre_train,
        rpn_pre_nms_top_n_test=rpn_pre_test,
        rpn_post_nms_top_n_train=rpn_post_train,
        rpn_post_nms_top_n_test=rpn_post_test,
        box_batch_size_per_image=box_batch_size_per_image,
        box_positive_fraction=box_positive_fraction,
    )

    # Replace 91-class COCO box predictor with (8 + 1)-class predictor while keeping pretrained backbone/FPN/RPN/box_head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Optional halved RPN anchor sizes arm while keeping 3 aspect ratios per location
    rpn_anchor_sizes = config.get("rpn_anchor_sizes", None)
    if rpn_anchor_sizes is not None:
        sizes_tuple = tuple(tuple(int(s) for s in lvl) for lvl in rpn_anchor_sizes)
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes_tuple)
        model.rpn.anchor_generator = AnchorGenerator(sizes=sizes_tuple, aspect_ratios=aspect_ratios)

    # GIoU RoI box regression loss (frcnn-e2)
    if str(config.get("roi_box_loss", "smooth_l1")).lower() == "giou":
        patch_frcnn_giou_loss(model)

    return model


def build_retinanet(
    config: dict[str, Any],
    pretrained: bool = True,
) -> torch.nn.Module:
    """Build `retinanet_resnet50_fpn_v2` for VisDrone (8 foreground classes + 1 background slot = 9)."""
    num_classes = int(config.get("num_classes", NUM_CLASSES + 1))
    min_size = int(config.get("min_size", 720))
    max_size = int(config.get("max_size", 1280))
    inf_cfg = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}

    detections_per_img = int(config.get("detections_per_img", inf_cfg.get("max_det", MAX_SUBMISSION_DETS)))
    score_thresh = float(config.get("score_thresh", inf_cfg.get("conf_thresh", 0.001)))
    nms_thresh = float(config.get("nms_thresh", inf_cfg.get("nms_iou", 0.6)))
    topk_candidates = int(config.get("topk_candidates", 1000))
    trainable_backbone_layers = int(config.get("trainable_backbone_layers", 3))

    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = tvd.retinanet_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        detections_per_img=detections_per_img,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        topk_candidates=topk_candidates,
    )

    # Replace final cls_logits Conv2d (keeping pretrained backbone, FPN, 4-conv cls tower, and regression head)
    cls_head = model.head.classification_head
    num_anchors = cls_head.num_anchors
    in_channels = cls_head.cls_logits.in_channels
    new_cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    nn.init.normal_(new_cls_logits.weight, std=0.01)
    prior_prob = float(config.get("prior_probability", 0.01))
    nn.init.constant_(new_cls_logits.bias, -math.log((1.0 - prior_prob) / prior_prob))
    cls_head.cls_logits = new_cls_logits
    cls_head.num_classes = num_classes

    # K-means anchors grounded in EDA (retina-e1/e2) (9 anchors per location -> head shapes remain valid)
    if bool(config.get("use_kmeans_anchors", False)):
        anchor_cfg = config.get("kmeans_anchors", None)
        if anchor_cfg is not None and "sizes" in anchor_cfg and "aspect_ratios" in anchor_cfg:
            sizes = tuple(tuple(int(s) for s in lvl) for lvl in anchor_cfg["sizes"])
            aspect_ratios = tuple(tuple(float(r) for r in lvl) for lvl in anchor_cfg["aspect_ratios"])
        else:
            fitted = fit_kmeans_anchors(target_long_side=float(max_size))
            sizes = fitted["sizes"]
            aspect_ratios = fitted["aspect_ratios"]
        model.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

    # Focal loss (alpha, gamma) & Matcher IoU thresholds (0.4 / 0.3) + optional GIoU regression
    if (
        "focal_alpha" in config
        or "focal_gamma" in config
        or "fg_iou_thresh" in config
        or "bg_iou_thresh" in config
        or "box_loss_type" in config
    ):
        patch_retinanet_focal_and_matcher(
            model,
            focal_alpha=float(config.get("focal_alpha", 0.25)),
            focal_gamma=float(config.get("focal_gamma", 2.0)),
            fg_iou_thresh=float(config.get("fg_iou_thresh", 0.5)),
            bg_iou_thresh=float(config.get("bg_iou_thresh", 0.4)),
            box_loss_type=config.get("box_loss_type", None),
        )

    return model


def build_tv_model(
    config: dict[str, Any],
    pretrained: bool = True,
) -> torch.nn.Module:
    """Dispatch to `build_faster_rcnn` or `build_retinanet` from config `family`."""
    family = str(config.get("family", "faster_rcnn")).lower()
    if family in {"faster_rcnn", "frcnn", "f1"}:
        return build_faster_rcnn(config, pretrained=pretrained)
    if family in {"retinanet", "retina", "f2"}:
        return build_retinanet(config, pretrained=pretrained)
    raise ValueError(f"Unsupported Torchvision model family: {family!r}")


class TorchvisionPredictor(Predictor):
    """Uniform `Predictor` wrapping a Faster R-CNN or RetinaNet checkpoint on original-resolution images."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        model: torch.nn.Module | None = None,
        config: dict[str, Any] | None = None,
        conf: float = 0.001,
        max_det: int = MAX_SUBMISSION_DETS,
        batch_size: int = 4,
        device: str | torch.device | None = None,
        half: bool = True,
        source_tag: str = "tv",
    ) -> None:
        dev_str = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev_str)
        self.conf = float(conf)
        self.max_det = int(max_det)
        self.batch_size = int(batch_size)
        self.half = bool(half) and self.device.type == "cuda"
        self.source_tag = source_tag

        if model is not None:
            self.model = model.to(self.device)
            self.config = config or {}
        elif weights_path is not None:
            ckpt = torch.load(Path(weights_path), map_location="cpu", weights_only=False)
            cfg = config or ckpt.get("config", {"family": "faster_rcnn"})
            self.config = cfg
            self.model = build_tv_model(cfg, pretrained=False)
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
        else:
            raise ValueError("Either `weights_path` or `model` must be provided to TorchvisionPredictor.")

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        frames: pd.DataFrame | Sequence[str],
        data_root: Path | None = None,
    ) -> pd.DataFrame:
        """Predict on original images specified by a manifest subset DataFrame or list of image_keys."""
        d_root = data_root or P.data
        if isinstance(frames, pd.DataFrame):
            m_sub = frames.reset_index(drop=True)
        else:
            manifest_df = load_manifest()
            key_set = set(frames)
            m_sub = manifest_df[manifest_df["image_key"].isin(key_set)].reset_index(drop=True)

        if m_sub.empty:
            return empty_predictions_df()

        img_paths = [d_root / str(r.path) for r in m_sub.itertuples(index=False)]
        keys = m_sub["image_key"].astype(str).tolist()
        widths = m_sub["width"].to_numpy(dtype=np.float64)
        heights = m_sub["height"].to_numpy(dtype=np.float64)

        records: list[pd.DataFrame] = []
        n = len(img_paths)
        was_training = self.model.training
        self.model.eval()

        for start in range(0, n, self.batch_size):
            end = min(n, start + self.batch_size)
            batch_tensors: list[torch.Tensor] = []
            valid_indices: list[int] = []
            for g_idx in range(start, end):
                bgr = cv2.imread(img_paths[g_idx].as_posix(), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                t = (
                    torch.from_numpy(np.ascontiguousarray(rgb))
                    .permute(2, 0, 1)
                    .to(device=self.device, dtype=torch.float32)
                    / 255.0
                )
                batch_tensors.append(t)
                valid_indices.append(g_idx)

            if not batch_tensors:
                continue

            with torch.amp.autocast(device_type=self.device.type, enabled=self.half):
                outputs = self.model(batch_tensors)

            for g_idx, out in zip(valid_indices, outputs):
                boxes_t = out.get("boxes")
                scores_t = out.get("scores")
                labels_t = out.get("labels")
                if boxes_t is None or len(boxes_t) == 0:
                    continue

                xyxy = boxes_t.detach().cpu().numpy().astype(np.float32)
                scores = scores_t.detach().cpu().numpy().astype(np.float32)
                # Convert 1..8 Torchvision foreground IDs back to 0..7 YOLO class IDs
                cls_ids = (labels_t.detach().cpu().numpy().astype(np.int64) - 1).astype(np.int8)

                valid_cls = (cls_ids >= 0) & (cls_ids < NUM_CLASSES) & (scores >= self.conf)
                if not np.any(valid_cls):
                    continue
                xyxy = xyxy[valid_cls]
                scores = scores[valid_cls]
                cls_ids = cls_ids[valid_cls]

                raw_xywh = np.stack(
                    [
                        xyxy[:, 0],
                        xyxy[:, 1],
                        xyxy[:, 2] - xyxy[:, 0],
                        xyxy[:, 3] - xyxy[:, 1],
                    ],
                    axis=1,
                )
                orig_w = widths[g_idx]
                orig_h = heights[g_idx]
                clipped_xywh, valid_box = clip_and_filter_boxes_xywh(raw_xywh, orig_w, orig_h)
                if not np.any(valid_box):
                    continue

                records.append(
                    pd.DataFrame(
                        {
                            "image_key": keys[g_idx],
                            "cls": cls_ids[valid_box],
                            "score": scores[valid_box],
                            "x": clipped_xywh[:, 0],
                            "y": clipped_xywh[:, 1],
                            "w": clipped_xywh[:, 2],
                            "h": clipped_xywh[:, 3],
                            "source": self.source_tag,
                        }
                    )
                )

        if was_training:
            self.model.train()

        if not records:
            return empty_predictions_df()
        combined = pd.concat(records, ignore_index=True)
        return normalize_predictions_df(combined, conf_thresh=self.conf, max_det_per_image=self.max_det)

