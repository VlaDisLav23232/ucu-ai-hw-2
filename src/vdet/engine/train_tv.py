"""Torchvision training engine with AMP, warmup+cosine LR, time budget, resumable checkpoints, and U0.5 eval (U2.0)."""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vdet.constants import DEFAULT_SEED, MAX_SUBMISSION_DETS, NUM_CLASSES
from vdet.data.datasets import build_tv_dataloaders
from vdet.data.manifest import load_manifest
from vdet.eval.coco_eval import evaluate
from vdet.infer.predictors import (
    clip_and_filter_boxes_xywh,
    empty_predictions_df,
    normalize_predictions_df,
    save_predictions_parquet,
)
from vdet.models.tv_builders import TorchvisionPredictor, build_tv_model
from vdet.models.tv_patches import compute_rpn_proposal_recall
from vdet.paths import P
from vdet.tracking.aggregate import aggregate_runs
from vdet.tracking.plots import curves
from vdet.tracking.run import RunLogger, make_run_id
from vdet.utils import save_json, seed_everything


@torch.no_grad()
def evaluate_tv_on_loader(
    model: torch.nn.Module,
    val_loader: DataLoader,
    manifest_df: pd.DataFrame,
    device: torch.device,
    conf_thresh: float = 0.001,
    max_det: int = MAX_SUBMISSION_DETS,
    use_amp: bool = True,
    split: str = "val",
    source_tag: str = "tv",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run inference on `val_loader` (cached images), map boxes back to original resolution, and evaluate via U0.5."""
    was_training = model.training
    model.eval()

    meta_map = (
        manifest_df.set_index("image_key")[["width", "height"]]
        .to_dict(orient="index")
    )

    records: list[pd.DataFrame] = []
    eval_keys: list[str] = []

    for images, targets in val_loader:
        imgs_dev = [img.to(device, non_blocking=True) for img in images]
        with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            outputs = model(imgs_dev)

        for img_t, tgt, out in zip(images, targets, outputs):
            key = str(tgt["image_key"])
            eval_keys.append(key)
            boxes_t = out.get("boxes")
            scores_t = out.get("scores")
            labels_t = out.get("labels")
            if boxes_t is None or len(boxes_t) == 0:
                continue

            xyxy = boxes_t.detach().cpu().numpy().astype(np.float32)
            scores = scores_t.detach().cpu().numpy().astype(np.float32)
            cls_ids = (labels_t.detach().cpu().numpy().astype(np.int64) - 1).astype(np.int8)

            keep = (cls_ids >= 0) & (cls_ids < NUM_CLASSES) & (scores >= conf_thresh)
            if not np.any(keep):
                continue
            xyxy = xyxy[keep]
            scores = scores[keep]
            cls_ids = cls_ids[keep]

            cache_h, cache_w = int(img_t.shape[-2]), int(img_t.shape[-1])
            orig_info = meta_map.get(key, {"width": cache_w, "height": cache_h})
            orig_w = float(orig_info["width"])
            orig_h = float(orig_info["height"])

            sx = orig_w / max(float(cache_w), 1.0)
            sy = orig_h / max(float(cache_h), 1.0)
            raw_xywh = np.stack(
                [
                    xyxy[:, 0] * sx,
                    xyxy[:, 1] * sy,
                    (xyxy[:, 2] - xyxy[:, 0]) * sx,
                    (xyxy[:, 3] - xyxy[:, 1]) * sy,
                ],
                axis=1,
            )
            clipped_xywh, valid_box = clip_and_filter_boxes_xywh(raw_xywh, orig_w, orig_h)
            if not np.any(valid_box):
                continue

            records.append(
                pd.DataFrame(
                    {
                        "image_key": key,
                        "cls": cls_ids[valid_box],
                        "score": scores[valid_box],
                        "x": clipped_xywh[:, 0],
                        "y": clipped_xywh[:, 1],
                        "w": clipped_xywh[:, 2],
                        "h": clipped_xywh[:, 3],
                        "source": source_tag,
                    }
                )
            )

    if was_training:
        model.train()

    if records:
        preds_df = normalize_predictions_df(
            pd.concat(records, ignore_index=True),
            conf_thresh=conf_thresh,
            max_det_per_image=max_det,
        )
    else:
        preds_df = empty_predictions_df()

    metrics = evaluate(preds_df, split=split, subset=eval_keys, per_res_bucket=False)
    return metrics, preds_df


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = str(config.get("optimizer", "sgd")).lower()
    lr = float(config.get("lr", 0.005))
    weight_decay = float(config.get("weight_decay", 1e-4))
    momentum = float(config.get("momentum", 0.9))

    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def _compute_lr_scale(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.05,
) -> float:
    """Linear warmup followed by cosine decay down to `min_lr_ratio`."""
    if total_steps <= 1:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return max(0.05, float(step + 1) / float(warmup_steps))
    progress = float(step - warmup_steps) / max(1.0, float(total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def train_torchvision(
    config: dict[str, Any],
    run_id: str | None = None,
    resume_path: str | Path | None = None,
    evaluate_full_val: bool = False,
) -> dict[str, Any]:
    """Train a Faster R-CNN (F1) or RetinaNet (F2) model under Protocol P0 with full local tracking (§7, §9, U2.0)."""
    seed = int(config.get("seed", DEFAULT_SEED))
    seed_everything(seed)

    family = str(config.get("family", "faster_rcnn"))
    exp = str(config.get("exp", "e0"))
    resolved_run_id = run_id or config.get("run_id") or make_run_id(family, exp)

    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    train_cfg = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    inf_cfg = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}

    cache_name = str(config.get("cache", data_cfg.get("cache_name", "k5_drop_L1920")))
    cache_dir = P.cache / cache_name
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    dev_str = str(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(dev_str)
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"

    train_loader, val_loader = build_tv_dataloaders(cache_dir=cache_dir, config=config)
    manifest_df = load_manifest()

    pretrained = bool(config.get("pretrained", True))
    model = build_tv_model(config, pretrained=pretrained)
    model.to(device)

    optimizer = _build_optimizer(model, config)
    base_lr = float(config.get("lr", 0.005))
    epochs = int(config.get("epochs", 10))
    warmup_epochs = float(config.get("warmup_epochs", 1.0))
    grad_clip_norm = float(config.get("grad_clip_norm", 10.0))
    time_budget_h = config.get("time_budget_h", train_cfg.get("time_budget_h", 1.5))
    time_budget_s = float(time_budget_h) * 3600.0 if (time_budget_h is not None and float(time_budget_h) > 0) else None

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    tv_out_dir = P.runs / resolved_run_id
    tv_out_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = tv_out_dir / "last.pth"
    best_ckpt_path = tv_out_dir / "best.pth"

    start_epoch = 1
    best_map = -1.0
    cum_gpu_time_s = 0.0

    res_target = Path(resume_path) if resume_path is not None else (
        Path(config["resume_from"]) if config.get("resume_from") else None
    )
    if res_target is not None and res_target.exists():
        ckpt = torch.load(res_target, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_map = float(ckpt.get("best_map", -1.0))
        cum_gpu_time_s = float(ckpt.get("cum_gpu_time_s", 0.0))
        print(f"Resumed {resolved_run_id} from {res_target} at epoch {start_epoch} (best_map={best_map:.4f})")

    logger = RunLogger(
        run_id=resolved_run_id,
        config=config,
        seed=seed,
        hypothesis=str(config.get("hypothesis", "")),
        parent_run=str(config.get("parent_run", "")),
        unit=str(config.get("unit", "U2.0")),
        family=family,
        exp=exp,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(round(warmup_epochs * steps_per_epoch))
    val_split_name = "train" if bool(config.get("val_on_train", False)) else "val"
    printed_throughput = False
    budget_exhausted = False

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        ep_t0 = time.perf_counter()
        sum_total_loss = 0.0
        sum_box_loss = 0.0
        sum_cls_loss = 0.0
        sum_aux_loss = 0.0
        n_batches = 0
        imgs_processed = 0

        for b_idx, (images, targets) in enumerate(train_loader):
            global_step = (epoch - 1) * steps_per_epoch + b_idx
            lr_scale = _compute_lr_scale(global_step, total_steps, warmup_steps)
            cur_lr = base_lr * lr_scale
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            imgs_dev = [img.to(device, non_blocking=True) for img in images]
            tgts_dev = [
                {
                    "boxes": t["boxes"].to(device, non_blocking=True),
                    "labels": t["labels"].to(device, non_blocking=True),
                }
                for t in targets
            ]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss_dict = model(imgs_dev, tgts_dev)
                if "loss_classifier" in loss_dict:
                    # Faster R-CNN losses: loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
                    cls_l = loss_dict["loss_classifier"]
                    box_l = loss_dict["loss_box_reg"]
                    aux_l = loss_dict.get("loss_objectness", torch.tensor(0.0, device=device)) + loss_dict.get(
                        "loss_rpn_box_reg", torch.tensor(0.0, device=device)
                    )
                else:
                    # RetinaNet losses: classification, bbox_regression
                    cls_l = loss_dict["classification"]
                    box_l = loss_dict["bbox_regression"]
                    aux_l = torch.tensor(0.0, device=device)
                total_loss = cls_l + box_l + aux_l

            if not torch.isfinite(total_loss):
                print(f"[warn] Non-finite loss at epoch {epoch} step {b_idx}: {loss_dict}, skipping batch.")
                continue

            scaler.scale(total_loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            sum_total_loss += float(total_loss.detach().item())
            sum_box_loss += float(box_l.detach().item())
            sum_cls_loss += float(cls_l.detach().item())
            sum_aux_loss += float(aux_l.detach().item())
            n_batches += 1
            imgs_processed += len(images)

            if not printed_throughput and n_batches == 50:
                elapsed_50 = max(1e-3, time.perf_counter() - ep_t0)
                img_per_s = imgs_processed / elapsed_50
                est_epoch_min = (len(train_loader.dataset) / max(img_per_s, 1e-3)) / 60.0
                print(
                    f"[throughput] {resolved_run_id}: {img_per_s:.2f} img/s "
                    f"(est. {est_epoch_min:.1f} min/epoch)"
                )
                printed_throughput = True

            if time_budget_s is not None:
                elapsed_now = cum_gpu_time_s + (time.perf_counter() - ep_t0)
                if elapsed_now >= time_budget_s * 0.94:
                    print(
                        f"[budget] Reached 94% of time budget ({elapsed_now / 3600.0:.3f}h / {time_budget_h}h) "
                        f"at epoch {epoch} batch {b_idx + 1}/{steps_per_epoch}; stopping epoch for validation."
                    )
                    budget_exhausted = True
                    break

        val_metrics, _ = evaluate_tv_on_loader(
            model=model,
            val_loader=val_loader,
            manifest_df=manifest_df,
            device=device,
            conf_thresh=float(inf_cfg.get("conf_thresh", 0.001)),
            max_det=int(inf_cfg.get("max_det", MAX_SUBMISSION_DETS)),
            use_amp=use_amp,
            split=val_split_name,
            source_tag=f"{family}-{exp}",
        )

        ep_time_s = time.perf_counter() - ep_t0
        cum_gpu_time_s += ep_time_s
        cum_gpu_time_h = cum_gpu_time_s / 3600.0

        denom = max(1, n_batches)
        val_ap = float(val_metrics.get("AP", 0.0))
        val_ap50 = float(val_metrics.get("AP50", 0.0))
        val_aps = float(val_metrics.get("APs", 0.0))
        cur_lr = float(optimizer.param_groups[0]["lr"])

        logger.log_epoch(
            epoch=epoch,
            train_loss=round(sum_total_loss / denom, 5),
            train_box_loss=round(sum_box_loss / denom, 5),
            train_cls_loss=round(sum_cls_loss / denom, 5),
            train_aux_loss=round(sum_aux_loss / denom, 5),
            val_map50_95=round(val_ap, 6),
            val_map50=round(val_ap50, 6),
            val_map_small=round(val_aps, 6),
            lr=round(cur_lr, 8),
            epoch_time_s=round(ep_time_s, 2),
            gpu_time_h=round(cum_gpu_time_h, 4),
        )
        print(
            f"[{resolved_run_id}] Epoch {epoch}/{epochs} ({ep_time_s:.1f}s, {cum_gpu_time_h:.3f}h) | "
            f"loss={sum_total_loss / denom:.4f} | proxy AP={val_ap:.4f} | AP50={val_ap50:.4f} | APs={val_aps:.4f}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_map": max(best_map, val_ap),
            "cum_gpu_time_s": cum_gpu_time_s,
            "config": config,
        }
        torch.save(ckpt_payload, last_ckpt_path)
        if val_ap >= best_map:
            best_map = val_ap
            torch.save(ckpt_payload, best_ckpt_path)

        if budget_exhausted or (time_budget_s is not None and cum_gpu_time_s >= time_budget_s):
            break

    # Copy best checkpoint to P.ckpt
    P.ckpt.mkdir(parents=True, exist_ok=True)
    src_ckpt = best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path
    ckpt_dest = P.ckpt / f"{resolved_run_id}_best.pth"
    if src_ckpt.exists():
        shutil.copy2(src_ckpt, ckpt_dest)

    metrics_df = pd.read_csv(logger.metrics_path) if logger.metrics_path.exists() else pd.DataFrame()
    if not metrics_df.empty:
        curves({resolved_run_id: metrics_df}, logger.plots_dir / f"curves_{family}.png")

    # Optional RPN proposal recall diagnostic for F1 (Faster R-CNN)
    if family in {"faster_rcnn", "frcnn", "f1"} and ckpt_dest.exists():
        rpn_stats = compute_rpn_proposal_recall(
            model=model,
            dataloader=val_loader,
            device=device,
            top_k=(300, 1000, 2000),
            max_batches=50,
        )
        save_json(rpn_stats, logger.run_dir / "rpn_recall.json")

    eval_metrics: dict[str, Any] | None = None
    if evaluate_full_val and ckpt_dest.exists():
        val_m = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
        max_val_frames = config.get("max_val_frames", None)
        if max_val_frames is not None:
            val_m = val_m.head(int(max_val_frames))

        predictor = TorchvisionPredictor(
            weights_path=ckpt_dest,
            config=config,
            conf=float(inf_cfg.get("conf_thresh", 0.001)),
            max_det=int(inf_cfg.get("max_det", MAX_SUBMISSION_DETS)),
            batch_size=max(1, int(config.get("batch", 4))),
            device=device,
            source_tag=f"{family}-{exp}",
        )
        preds_val_df = predictor.predict(val_m)

        save_predictions_parquet(preds_val_df, tv_out_dir / "preds_val.parquet", conf_thresh=0.001)
        save_predictions_parquet(
            preds_val_df,
            logger.run_dir / "preds_val_slim.parquet",
            conf_thresh=float(inf_cfg.get("slim_conf_thresh", 0.05)),
        )

        subset_keys = val_m["image_key"].tolist() if max_val_frames is not None else None
        eval_metrics = evaluate(
            preds_val_df,
            split="val",
            subset=subset_keys,
            per_res_bucket=True,
        )

    summary = logger.finalize(
        eval_metrics=eval_metrics,
        checkpoint_uri=ckpt_dest.as_posix() if ckpt_dest.exists() else "",
    )
    aggregate_runs()
    return {
        "run_id": resolved_run_id,
        "run_dir": logger.run_dir,
        "checkpoint": ckpt_dest,
        "metrics_df": metrics_df,
        "eval_metrics": eval_metrics,
        "summary": summary,
    }

