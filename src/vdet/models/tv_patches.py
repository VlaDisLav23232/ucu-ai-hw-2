"""Small, unit-tested overrides for Torchvision Faster R-CNN (F1) and RetinaNet (F2) (§10 F1-E2, F2-E2)."""

from __future__ import annotations

import math
import types
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torchvision.models.detection._utils import Matcher
from torchvision.ops import box_iou, sigmoid_focal_loss

from vdet.models.losses import giou_loss_xyxy


def decode_rel_codes_to_canonical_xyxy(
    rel_codes: torch.Tensor,
    weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
    bbox_xform_clip: float = math.log(1000.0 / 16.0),
) -> torch.Tensor:
    """Decode Faster R-CNN `(dx, dy, dw, dh)` deltas relative to the canonical unit proposal `[0, 0, 1, 1]`.

    Because GIoU is analytically invariant to positive axis-aligned scaling `(W_p, H_p)` and
    translation `(cx_p, cy_p)`, decoding predicted and target deltas relative to `[0, 0, 1, 1]`
    (center `(0.5, 0.5)`, size `(1.0, 1.0)`) yields the exact same GIoU as decoding against
    the image-space proposal boxes `(x1_p, y1_p, x2_p, y2_p)`.
    """
    wx, wy, ww, wh = weights
    dx = rel_codes[:, 0] / wx
    dy = rel_codes[:, 1] / wy
    dw = torch.clamp(rel_codes[:, 2] / ww, max=bbox_xform_clip)
    dh = torch.clamp(rel_codes[:, 3] / wh, max=bbox_xform_clip)

    pred_cx = dx + 0.5
    pred_cy = dy + 0.5
    pred_w = torch.exp(dw)
    pred_h = torch.exp(dh)

    x1 = pred_cx - 0.5 * pred_w
    y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w
    y2 = pred_cy + 0.5 * pred_h
    return torch.stack((x1, y1, x2, y2), dim=-1)


def giou_fastrcnn_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: list[torch.Tensor],
    regression_targets: list[torch.Tensor],
    box_coder_weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for `torchvision.models.detection.roi_heads.fastrcnn_loss` using GIoU box loss (§10 F1-E2)."""
    labels_cat = torch.cat(labels, dim=0)
    reg_targets_cat = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels_cat)

    sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
    if sampled_pos_inds_subset.numel() == 0:
        return classification_loss, box_regression.sum() * 0.0

    labels_pos = labels_cat[sampled_pos_inds_subset]
    n_proposals = class_logits.shape[0]
    box_reg_reshaped = box_regression.reshape(n_proposals, box_regression.size(-1) // 4, 4)

    pred_deltas = box_reg_reshaped[sampled_pos_inds_subset, labels_pos]
    target_deltas = reg_targets_cat[sampled_pos_inds_subset]

    pred_boxes = decode_rel_codes_to_canonical_xyxy(pred_deltas, weights=box_coder_weights)
    target_boxes = decode_rel_codes_to_canonical_xyxy(target_deltas, weights=box_coder_weights)

    giou_losses = giou_loss_xyxy(pred_boxes, target_boxes)
    box_loss = giou_losses.sum() / max(1, labels_cat.numel())
    return classification_loss, box_loss


def patch_frcnn_giou_loss(model: torch.nn.Module) -> torch.nn.Module:
    """Patch a `FasterRCNN` instance so `roi_heads.forward` computes GIoU regression loss (§10 F1-E2)."""
    roi_heads = model.roi_heads
    weights = tuple(float(w) for w in roi_heads.box_coder.weights)

    def _forward_giou(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        if self.training:
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: list[dict[str, torch.Tensor]] = []
        losses: dict[str, torch.Tensor] = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = giou_fastrcnn_loss(
                class_logits,
                box_regression,
                labels,
                regression_targets,
                box_coder_weights=weights,
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, lbls = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            for i in range(len(boxes)):
                result.append({"boxes": boxes[i], "labels": lbls[i], "scores": scores[i]})

        return result, losses

    roi_heads.forward = types.MethodType(_forward_giou, roi_heads)
    roi_heads._vdet_giou_patched = True
    return model


def patch_retinanet_focal_and_matcher(
    model: torch.nn.Module,
    focal_alpha: float = 0.25,
    focal_gamma: float = 1.5,
    fg_iou_thresh: float = 0.4,
    bg_iou_thresh: float = 0.3,
    box_loss_type: str | None = None,
) -> torch.nn.Module:
    """Patch a `RetinaNet` instance with custom Focal Loss `(alpha, gamma)` and anchor Matcher IoU thresholds (§10 F2-E2)."""
    alpha = float(focal_alpha)
    gamma = float(focal_gamma)

    model.proposal_matcher = Matcher(
        float(fg_iou_thresh),
        float(bg_iou_thresh),
        allow_low_quality_matches=True,
    )
    model.head.classification_head.fg_iou_thresh = float(fg_iou_thresh)
    model.head.classification_head.bg_iou_thresh = float(bg_iou_thresh)
    model.head.classification_head._vdet_focal_alpha = alpha
    model.head.classification_head._vdet_focal_gamma = gamma

    def _compute_cls_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        matched_idxs: list[torch.Tensor],
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        cls_logits = head_outputs["cls_logits"]
        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            foreground_idxs = matched_idxs_per_image >= 0
            num_foreground = int(foreground_idxs.sum().item())

            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs]],
            ] = 1.0

            valid_idxs = matched_idxs_per_image != self.BETWEEN_THRESHOLDS
            loss_img = sigmoid_focal_loss(
                cls_logits_per_image[valid_idxs],
                gt_classes_target[valid_idxs],
                alpha=self._vdet_focal_alpha,
                gamma=self._vdet_focal_gamma,
                reduction="sum",
            ) / max(1, num_foreground)
            losses.append(loss_img)

        return torch.stack(losses).sum() / max(1, len(targets))

    cls_head = model.head.classification_head
    cls_head.compute_loss = types.MethodType(_compute_cls_loss, cls_head)

    if box_loss_type is not None:
        model.head.regression_head._loss_type = str(box_loss_type)

    return model


@torch.no_grad()
def compute_rpn_proposal_recall(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device | str = "cuda",
    top_k: Sequence[int] = (300, 1000, 2000),
    iou_thresh: float = 0.5,
    max_batches: int | None = 50,
) -> dict[str, float]:
    """Compute RPN class-agnostic proposal recall@K (overall and small objects <32^2 px) for F1 diagnostics (§10 F1)."""
    was_training = model.training
    model.eval()
    dev = torch.device(device)

    total_gt = 0
    total_gt_small = 0
    hits_by_k = {int(k): 0 for k in top_k}
    hits_small_by_k = {int(k): 0 for k in top_k}

    for b_idx, (images, targets) in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break
        imgs_dev = [img.to(dev) for img in images]
        images_list, _ = model.transform(imgs_dev, None)
        features = model.backbone(images_list.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        proposals, _ = model.rpn(images_list, features, None)

        orig_sizes = [(img.shape[-2], img.shape[-1]) for img in imgs_dev]
        for props_img, img_shape, orig_sz, tgt in zip(proposals, images_list.image_sizes, orig_sizes, targets):
            gt_boxes = tgt["boxes"].to(dev)
            if len(gt_boxes) == 0:
                continue
            # Rescale proposals from transformed image_shape back to dataset image size orig_sz
            sy = float(orig_sz[0]) / max(float(img_shape[0]), 1.0)
            sx = float(orig_sz[1]) / max(float(img_shape[1]), 1.0)
            p_scaled = props_img.clone()
            p_scaled[:, [0, 2]] *= sx
            p_scaled[:, [1, 3]] *= sy

            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            small_mask = gt_areas < (32.0 * 32.0)

            n_gt = len(gt_boxes)
            n_small = int(small_mask.sum().item())
            total_gt += n_gt
            total_gt_small += n_small

            for k in hits_by_k:
                p_k = p_scaled[:k]
                if len(p_k) == 0:
                    continue
                ious = box_iou(gt_boxes, p_k)
                matched = ious.max(dim=1).values >= float(iou_thresh)
                hits_by_k[k] += int(matched.sum().item())
                if n_small > 0:
                    hits_small_by_k[k] += int(matched[small_mask].sum().item())

    if was_training:
        model.train()

    out: dict[str, float] = {
        "total_gt": float(total_gt),
        "total_gt_small": float(total_gt_small),
    }
    for k in hits_by_k:
        out[f"rpn_recall@{k}"] = round(hits_by_k[k] / max(1, total_gt), 4)
        out[f"rpn_small_recall@{k}"] = round(hits_small_by_k[k] / max(1, total_gt_small), 4)
    return out

