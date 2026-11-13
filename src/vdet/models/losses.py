"""Bounding-box regression losses: GIoU, CIoU, and Normalized Wasserstein Distance (NWD).

Credits: GIoU — Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019; CIoU — Zheng et al.,
"Distance-IoU Loss", AAAI 2020; NWD — Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny
Object Detection", arXiv:2110.13389, 2021.
"""

from __future__ import annotations

import math

import torch


def boxes_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert `[..., 4]` boxes from `(x1, y1, x2, y2)` to `(cx, cy, w, h)`."""
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack((cx, cy, w, h), dim=-1)


def nwd_similarity_xyxy(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    c: float = 24.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute pairwise 1-to-1 Normalized Wasserstein Distance (NWD) similarity in `(0, 1]`.

    Models bounding boxes as 2D Gaussian distributions `N(mu, Sigma)` with
    `mu = (cx, cy)` and `Sigma = diag((w/2)^2, (h/2)^2)`:
      W_2^2 = (cx1 - cx2)^2 + (cy1 - cy2)^2 + ((w1 - w2)^2 + (h1 - h2)^2) / 4
      NWD   = exp(-sqrt(W_2^2) / C)
    """
    b1 = boxes_xyxy_to_cxcywh(boxes1)
    b2 = boxes_xyxy_to_cxcywh(boxes2)

    dcx = b1[..., 0] - b2[..., 0]
    dcy = b1[..., 1] - b2[..., 1]
    dw = (b1[..., 2] - b2[..., 2]) * 0.5
    dh = (b1[..., 3] - b2[..., 3]) * 0.5

    w2_sq = dcx.square() + dcy.square() + dw.square() + dh.square()
    w2 = torch.sqrt(w2_sq + eps)
    return torch.exp(-w2 / max(float(c), 1e-3))


def nwd_loss_xyxy(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    c: float = 24.0,
) -> torch.Tensor:
    """Return `1.0 - NWD(boxes1, boxes2)` per aligned pair."""
    return 1.0 - nwd_similarity_xyxy(boxes1, boxes2, c=c)


def giou_similarity_xyxy(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute 1-to-1 Generalized IoU (GIoU) in `[-1, 1]` for `(x1, y1, x2, y2)` tensors."""
    x1_1, y1_1, x2_1, y2_1 = boxes1.unbind(dim=-1)
    x1_2, y1_2, x2_2, y2_2 = boxes2.unbind(dim=-1)

    w1 = (x2_1 - x1_1).clamp(min=0.0)
    h1 = (y2_1 - y1_1).clamp(min=0.0)
    w2 = (x2_2 - x1_2).clamp(min=0.0)
    h2 = (y2_2 - y1_2).clamp(min=0.0)

    inter_x1 = torch.maximum(x1_1, x1_2)
    inter_y1 = torch.maximum(y1_1, y1_2)
    inter_x2 = torch.minimum(x2_1, x2_2)
    inter_y2 = torch.minimum(y2_1, y2_2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    union_area = w1 * h1 + w2 * h2 - inter_area + eps
    iou = inter_area / union_area

    enclose_x1 = torch.minimum(x1_1, x1_2)
    enclose_y1 = torch.minimum(y1_1, y1_2)
    enclose_x2 = torch.maximum(x2_1, x2_2)
    enclose_y2 = torch.maximum(y2_1, y2_2)

    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0.0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0.0)
    enclose_area = enclose_w * enclose_h + eps

    return iou - (enclose_area - union_area) / enclose_area


def giou_loss_xyxy(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """Return `1.0 - GIoU(boxes1, boxes2)` per aligned pair."""
    return 1.0 - giou_similarity_xyxy(boxes1, boxes2)


def ciou_similarity_xyxy(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute 1-to-1 Complete IoU (CIoU) for `(x1, y1, x2, y2)` tensors."""
    x1_1, y1_1, x2_1, y2_1 = boxes1.unbind(dim=-1)
    x1_2, y1_2, x2_2, y2_2 = boxes2.unbind(dim=-1)

    w1 = (x2_1 - x1_1).clamp(min=eps)
    h1 = (y2_1 - y1_1).clamp(min=eps)
    w2 = (x2_2 - x1_2).clamp(min=eps)
    h2 = (y2_2 - y1_2).clamp(min=eps)

    inter_w = (torch.minimum(x2_1, x2_2) - torch.maximum(x1_1, x1_2)).clamp(min=0.0)
    inter_h = (torch.minimum(y2_1, y2_2) - torch.maximum(y1_1, y1_2)).clamp(min=0.0)
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.maximum(x2_1, x2_2) - torch.minimum(x1_1, x1_2)
    ch = torch.maximum(y2_1, y2_2) - torch.minimum(y1_1, y1_2)
    c2 = cw.square() + ch.square() + eps

    cx1 = (x1_1 + x2_1) * 0.5
    cy1 = (y1_1 + y2_1) * 0.5
    cx2 = (x1_2 + x2_2) * 0.5
    cy2 = (y1_2 + y2_2) * 0.5
    rho2 = (cx2 - cx1).square() + (cy2 - cy1).square()

    v = (4.0 / (math.pi**2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1.0 + eps))

    return iou - (rho2 / c2 + v * alpha)


def ciou_nwd_loss_xyxy(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    lambda_nwd: float = 0.5,
    c: float = 24.0,
) -> torch.Tensor:
    """Blended CIoU + NWD box regression loss: `(1 - lambda_nwd)*(1 - CIoU) + lambda_nwd*(1 - NWD)`."""
    lam = float(lambda_nwd)
    ciou_loss = 1.0 - ciou_similarity_xyxy(boxes1, boxes2)
    nwd_loss = nwd_loss_xyxy(boxes1, boxes2, c=c)
    return (1.0 - lam) * ciou_loss + lam * nwd_loss

