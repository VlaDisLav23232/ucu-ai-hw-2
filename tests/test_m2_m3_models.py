"""Unit tests for M2–M3 model families: losses (NWD/GIoU), k-means anchors, tv_patches, tv_builders, and datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.models.detection._utils import BoxCoder

from vdet.data.augment import DetectionAugmentor
from vdet.models.anchors import fit_kmeans_anchors
from vdet.models.losses import (
    ciou_nwd_loss_xyxy,
    giou_loss_xyxy,
    giou_similarity_xyxy,
    nwd_loss_xyxy,
    nwd_similarity_xyxy,
)
from vdet.models.tv_builders import build_faster_rcnn, build_retinanet
from vdet.models.tv_patches import (
    decode_rel_codes_to_canonical_xyxy,
    giou_fastrcnn_loss,
)
from vdet.models.ultra import (
    make_nwd_bbox_loss,
    make_nwd_detection_trainer,
    make_rtdetr_custom_trainer,
)


def test_nwd_and_giou_losses_properties() -> None:
    """Identical boxes yield 0 loss / 1.0 similarity; shifted boxes yield positive loss and finite gradients."""
    b1 = torch.tensor([[10.0, 20.0, 30.0, 50.0], [100.0, 100.0, 108.0, 110.0]], requires_grad=True)
    b2 = b1.detach().clone()

    assert torch.allclose(nwd_similarity_xyxy(b1, b2, c=24.0), torch.ones(2), atol=1e-3)
    assert torch.allclose(giou_similarity_xyxy(b1, b2), torch.ones(2), atol=1e-5)
    assert torch.allclose(nwd_loss_xyxy(b1, b2, c=24.0), torch.zeros(2), atol=1e-3)
    assert torch.allclose(giou_loss_xyxy(b1, b2), torch.zeros(2), atol=1e-5)

    # Disjoint tiny boxes: IoU is 0, while NWD remains smooth and differentiable
    b_shifted = torch.tensor([[40.0, 60.0, 60.0, 90.0], [120.0, 120.0, 128.0, 130.0]])
    loss_blend = ciou_nwd_loss_xyxy(b1, b_shifted, lambda_nwd=0.5, c=24.0).mean()
    loss_blend.backward()
    assert loss_blend.item() > 0.0
    assert b1.grad is not None and torch.all(torch.isfinite(b1.grad))


def test_giou_fastrcnn_canonical_invariance() -> None:
    """Canonical-proposal decoding in `giou_fastrcnn_loss` matches image-space BoxCoder decoding to <1e-5."""
    weights = (10.0, 10.0, 5.0, 5.0)
    coder = BoxCoder(weights=weights)

    proposals = torch.tensor(
        [
            [50.0, 80.0, 110.0, 200.0],
            [300.0, 150.0, 340.0, 190.0],
            [12.0, 14.0, 24.0, 32.0],
        ]
    )
    gt_boxes = torch.tensor(
        [
            [55.0, 85.0, 105.0, 195.0],
            [310.0, 145.0, 355.0, 198.0],
            [10.0, 15.0, 26.0, 30.0],
        ]
    )
    pred_boxes = torch.tensor(
        [
            [52.0, 78.0, 112.0, 190.0],
            [295.0, 152.0, 345.0, 188.0],
            [14.0, 12.0, 25.0, 35.0],
        ]
    )

    target_deltas = coder.encode_single(gt_boxes, proposals)
    pred_deltas = coder.encode_single(pred_boxes, proposals)

    # Image-space GIoU vs Canonical [0,0,1,1] GIoU
    giou_img = giou_loss_xyxy(pred_boxes, gt_boxes)
    giou_can = giou_loss_xyxy(
        decode_rel_codes_to_canonical_xyxy(pred_deltas, weights=weights),
        decode_rel_codes_to_canonical_xyxy(target_deltas, weights=weights),
    )
    assert torch.allclose(giou_img, giou_can, atol=1e-5)

    # Full giou_fastrcnn_loss call with gradients
    n, num_classes = 3, 9
    class_logits = torch.randn(n, num_classes, requires_grad=True)
    box_regression = torch.randn(n, num_classes * 4, requires_grad=True)
    labels = [torch.tensor([1, 0, 4], dtype=torch.int64)]
    reg_targets = [target_deltas]

    cls_loss, box_loss = giou_fastrcnn_loss(class_logits, box_regression, labels, reg_targets)
    (cls_loss + box_loss).backward()
    assert cls_loss.item() > 0.0 and box_loss.item() >= 0.0
    assert box_regression.grad is not None and torch.all(torch.isfinite(box_regression.grad))


def test_kmeans_anchors_shape_and_improvement() -> None:
    """K-means anchor fitting produces 5 levels x 3 sizes x 3 aspect ratios (9/loc) and improves IoU."""
    rng = np.random.default_rng(42)
    m_df = pd.DataFrame(
        {
            "image_key": ["clip1_0000001", "clip1_0000002"],
            "split": ["train", "train"],
            "width": [1920, 1920],
            "height": [1080, 1080],
        }
    )
    b_df = pd.DataFrame(
        {
            "image_key": rng.choice(["clip1_0000001", "clip1_0000002"], size=600),
            "split": ["train"] * 600,
            "w": rng.uniform(12.0, 140.0, size=600),
            "h": rng.uniform(12.0, 160.0, size=600),
        }
    )
    res = fit_kmeans_anchors(boxes_df=b_df, manifest_df=m_df, use_cached_yaml=False)
    assert len(res["sizes"]) == 5
    assert all(len(lvl) == 3 for lvl in res["sizes"])
    assert len(res["aspect_ratios"]) == 5
    assert all(len(lvl) == 3 for lvl in res["aspect_ratios"])
    assert res["mean_best_iou_fitted"] > res["mean_best_iou_default"]


def test_tv_builders_and_patches_forward_backward() -> None:
    """Faster R-CNN (with GIoU patch) and RetinaNet (with k-means anchors + focal/matcher patch) run train/eval."""
    frcnn_cfg = {
        "family": "faster_rcnn",
        "num_classes": 9,
        "min_size": 320,
        "max_size": 320,
        "rpn_anchor_sizes": [[16], [32], [64], [128], [256]],
        "roi_box_loss": "giou",
    }
    frcnn = build_faster_rcnn(frcnn_cfg, pretrained=False)
    assert getattr(frcnn.roi_heads, "_vdet_giou_patched", False) is True

    img = torch.rand(3, 320, 320)
    target = {
        "boxes": torch.tensor([[20.0, 30.0, 80.0, 110.0], [120.0, 140.0, 160.0, 190.0]], dtype=torch.float32),
        "labels": torch.tensor([1, 3], dtype=torch.int64),
    }
    frcnn.train()
    losses_f = frcnn([img], [target])
    total_f = sum(losses_f.values())
    total_f.backward()
    assert torch.isfinite(total_f)

    retina_cfg = {
        "family": "retinanet",
        "num_classes": 9,
        "min_size": 320,
        "max_size": 320,
        "use_kmeans_anchors": True,
        "kmeans_anchors": {
            "sizes": [[10, 14, 18], [22, 26, 30], [34, 42, 50], [60, 75, 90], [110, 140, 180]],
            "aspect_ratios": [[0.58, 1.06, 1.92]] * 5,
        },
        "focal_alpha": 0.25,
        "focal_gamma": 1.5,
        "fg_iou_thresh": 0.4,
        "bg_iou_thresh": 0.3,
        "box_loss_type": "giou",
    }
    retina = build_retinanet(retina_cfg, pretrained=False)
    retina.train()
    losses_r = retina([img], [target])
    total_r = sum(losses_r.values())
    total_r.backward()
    assert torch.isfinite(total_r)


def test_augmentor_and_ultralytics_custom_trainers() -> None:
    """Augmentor preserves valid bbox coordinates, and custom Ultralytics NWD / RT-DETR trainers construct cleanly."""
    aug = DetectionAugmentor(hflip=1.0, scale_min=0.5, scale_max=0.5, hsv_h=0.01, hsv_s=0.2, hsv_v=0.2, is_train=True)
    img = np.full((200, 400, 3), 128, dtype=np.uint8)
    boxes = np.array([[40.0, 20.0, 120.0, 100.0]], dtype=np.float32)
    labels = np.array([2], dtype=np.int64)
    out_img, out_boxes, out_labels = aug(img, boxes, labels, rng=np.random.default_rng(0))
    assert out_img.shape[0] <= 200 and out_img.shape[1] <= 400
    assert len(out_boxes) == 1 and len(out_labels) == 1
    assert np.all(out_boxes[:, 2] > out_boxes[:, 0]) and np.all(out_boxes[:, 3] > out_boxes[:, 1])

    nwd_loss_mod = make_nwd_bbox_loss(reg_max=16, lambda_nwd=0.5, nwd_c=24.0)
    assert nwd_loss_mod.lambda_nwd == 0.5 and nwd_loss_mod.nwd_c == 24.0
    assert make_nwd_detection_trainer(0.5, 24.0) is not None
    assert make_rtdetr_custom_trainer({"class": 1.5, "bbox": 7.5, "giou": 3.0}, num_denoising=200) is not None

