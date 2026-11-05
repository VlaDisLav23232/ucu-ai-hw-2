"""Ultralytics (YOLO11 / RT-DETR) training wrapper, custom trainers (NWD / RT-DETR), and Predictor (§7, F3, F4)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from vdet.constants import DEFAULT_SEED, MAX_SUBMISSION_DETS
from vdet.data.export_ultra import write_ultralytics_data_yaml
from vdet.data.manifest import load_manifest
from vdet.eval.coco_eval import evaluate
from vdet.infer.predictors import (
    Predictor,
    clip_and_filter_boxes_xywh,
    empty_predictions_df,
    normalize_predictions_df,
    save_predictions_parquet,
)
from vdet.models.losses import nwd_similarity_xyxy
from vdet.paths import P
from vdet.tracking.aggregate import aggregate_runs
from vdet.tracking.plots import curves
from vdet.tracking.run import RunLogger, make_run_id
from vdet.tracking.ultra_adapter import adapt_ultralytics_run
from vdet.utils import seed_everything


def _prepare_runtime_data_yaml(
    cache_dir: Path,
    train_stride: int = 1,
    max_train_images: int | None = None,
    val_on_train: bool = False,
) -> Path:
    """Create a runtime Ultralytics data YAML pointing to `cache_dir` (and optional subset txt)."""
    cache_dir = cache_dir.resolve()
    if train_stride <= 1 and max_train_images is None and not val_on_train:
        return write_ultralytics_data_yaml(cache_dir, out_filename="data.yaml")

    train_imgs = sorted((cache_dir / "train" / "images").glob("*.jpg"))
    if train_stride > 1:
        train_imgs = train_imgs[::train_stride]
    if max_train_images is not None:
        train_imgs = train_imgs[:max_train_images]

    subset_tag = f"s{train_stride}_n{max_train_images or 'all'}"
    subset_txt = cache_dir / f"train_subset_{subset_tag}.txt"
    subset_txt.write_text(
        "\n".join(p.resolve().as_posix() for p in train_imgs) + "\n",
        encoding="utf-8",
    )
    val_target = subset_txt.name if val_on_train else "val/images"
    return write_ultralytics_data_yaml(
        cache_dir,
        train_rel=subset_txt.name,
        val_rel=val_target,
        out_filename=f"data_{subset_tag}.yaml",
    )


def _load_ultralytics_model(model_spec: str | Path, family: str | None = None):
    """Instantiate YOLO or RTDETR from Ultralytics based on model name or family."""
    from ultralytics import RTDETR, YOLO

    spec_str = str(model_spec)
    is_rtdetr = "rtdetr" in spec_str.lower() or (family is not None and "rtdetr" in family.lower())
    if is_rtdetr:
        return RTDETR(spec_str)
    return YOLO(spec_str)


from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


class NWDBboxLoss(BboxLoss):
    """Ultralytics `BboxLoss` subclass blending CIoU and NWD (`F3-E2`)."""

    def __init__(self, reg_max: int, lambda_nwd: float = 0.5, nwd_c: float = 24.0) -> None:
        super().__init__(reg_max)
        self.lambda_nwd = float(lambda_nwd)
        self.nwd_c = float(nwd_c)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fg_idx = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[fg_idx].sum(-1, keepdim=True)

        pred_fg = pred_bboxes[fg_idx]
        target_fg = target_bboxes[fg_idx]
        ciou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)

        # Scale from stride units back to image pixels for NWD constant C
        stride_b = stride.expand(pred_bboxes.shape[0], -1, 1)[fg_idx]
        pred_px = pred_fg * stride_b
        target_px = target_fg * stride_b
        nwd = nwd_similarity_xyxy(pred_px, target_px, c=self.nwd_c).unsqueeze(-1)

        blended_sim = (1.0 - self.lambda_nwd) * ciou + self.lambda_nwd * nwd
        loss_iou = ((1.0 - blended_sim) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_idx].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_idx])
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_s = pred_dist * stride
            pred_dist_s[..., 0::2] /= imgsz[1]
            pred_dist_s[..., 0::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist_s[fg_idx], target_ltrb[fg_idx], reduction="none").mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class NWDDetectionLoss(v8DetectionLoss):
    """`v8DetectionLoss` using `NWDBboxLoss` (`CIoU + NWD` blend)."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        lambda_nwd: float = 0.5,
        nwd_c: float = 24.0,
    ) -> None:
        super().__init__(model, tal_topk=tal_topk)
        m = model.model[-1]
        self.bbox_loss = NWDBboxLoss(m.reg_max, lambda_nwd=lambda_nwd, nwd_c=nwd_c).to(self.device)


class NWDDetectionModel(DetectionModel):
    """Top-level `DetectionModel` subclass initializing `NWDDetectionLoss`."""

    _vdet_lambda_nwd: float = 0.5
    _vdet_nwd_c: float = 24.0

    def init_criterion(self):
        return NWDDetectionLoss(
            self,
            lambda_nwd=getattr(self, "_vdet_lambda_nwd", 0.5),
            nwd_c=getattr(self, "_vdet_nwd_c", 24.0),
        )


class NWDDetectionTrainer(DetectionTrainer):
    """Top-level `DetectionTrainer` that serializes standard `DetectionModel` checkpoints cleanly."""

    _vdet_lambda_nwd: float = 0.5
    _vdet_nwd_c: float = 24.0

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = NWDDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        model._vdet_lambda_nwd = float(self._vdet_lambda_nwd)
        model._vdet_nwd_c = float(self._vdet_nwd_c)
        if weights:
            model.load(weights)
        return model

    def save_model(self):
        ema_obj = getattr(self, "ema", None)
        ema_model = getattr(ema_obj, "ema", None)
        orig_cls = ema_model.__class__ if ema_model is not None else None
        try:
            if ema_model is not None:
                ema_model.__class__ = DetectionModel
            return super().save_model()
        finally:
            if ema_model is not None and orig_cls is not None:
                ema_model.__class__ = orig_cls


class CustomRTDETRModel(RTDETRDetectionModel):
    """Top-level `RTDETRDetectionModel` subclass with customizable matcher cost gains."""

    _vdet_cost_gain: dict[str, float] | None = None
    _vdet_loss_gain: dict[str, float] | None = None

    def init_criterion(self):
        loss_gain = getattr(self, "_vdet_loss_gain", None)
        cost_gain = getattr(self, "_vdet_cost_gain", None)
        criterion = RTDETRDetectionLoss(
            nc=self.nc,
            loss_gain=dict(loss_gain) if loss_gain is not None else None,
            use_vfl=True,
        )
        if cost_gain is not None:
            criterion.matcher.cost_gain = {k: float(v) for k, v in cost_gain.items()}
        return criterion


class CustomRTDETRTrainer(RTDETRTrainer):
    """Top-level `RTDETRTrainer` that serializes standard `RTDETRDetectionModel` checkpoints cleanly."""

    _vdet_cost_gain: dict[str, float] | None = None
    _vdet_loss_gain: dict[str, float] | None = None
    _vdet_num_denoising: int | None = None

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = CustomRTDETRModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        model._vdet_cost_gain = self._vdet_cost_gain
        model._vdet_loss_gain = self._vdet_loss_gain
        if weights:
            model.load(weights)
        if self._vdet_num_denoising is not None:
            head = model.model[-1]
            if hasattr(head, "nd"):
                head.nd = int(self._vdet_num_denoising)
            if hasattr(head, "num_denoising"):
                head.num_denoising = int(self._vdet_num_denoising)
        return model

    def save_model(self):
        ema_obj = getattr(self, "ema", None)
        ema_model = getattr(ema_obj, "ema", None)
        orig_cls = ema_model.__class__ if ema_model is not None else None
        try:
            if ema_model is not None:
                ema_model.__class__ = RTDETRDetectionModel
            return super().save_model()
        finally:
            if ema_model is not None and orig_cls is not None:
                ema_model.__class__ = orig_cls


def make_nwd_bbox_loss(reg_max: int, lambda_nwd: float = 0.5, nwd_c: float = 24.0) -> NWDBboxLoss:
    """Build an Ultralytics `NWDBboxLoss` instance that blends CIoU and NWD (`F3-E2`)."""
    return NWDBboxLoss(reg_max, lambda_nwd=lambda_nwd, nwd_c=nwd_c)


def make_nwd_detection_trainer(lambda_nwd: float = 0.5, nwd_c: float = 24.0):
    """Configure and return the top-level `NWDDetectionTrainer` class."""
    NWDDetectionTrainer._vdet_lambda_nwd = float(lambda_nwd)
    NWDDetectionTrainer._vdet_nwd_c = float(nwd_c)
    return NWDDetectionTrainer


def make_rtdetr_custom_trainer(
    cost_gain: dict[str, float] | None = None,
    loss_gain: dict[str, float] | None = None,
    num_denoising: int | None = None,
):
    """Configure and return the top-level `CustomRTDETRTrainer` class."""
    CustomRTDETRTrainer._vdet_cost_gain = dict(cost_gain) if cost_gain is not None else None
    CustomRTDETRTrainer._vdet_loss_gain = dict(loss_gain) if loss_gain is not None else None
    CustomRTDETRTrainer._vdet_num_denoising = int(num_denoising) if num_denoising is not None else None
    return CustomRTDETRTrainer


class UltralyticsPredictor(Predictor):
    """Uniform `Predictor` wrapping an Ultralytics checkpoint on original-resolution images (§6.3)."""

    def __init__(
        self,
        weights_path: str | Path,
        imgsz: int = 1280,
        conf: float = 0.001,
        iou: float = 0.6,
        max_det: int = MAX_SUBMISSION_DETS,
        batch_size: int = 8,
        device: str | int | None = None,
        half: bool = True,
        source_tag: str = "ultra",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.model = _load_ultralytics_model(self.weights_path)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.batch_size = int(batch_size)
        self.device = device
        self.half = bool(half)
        self.source_tag = source_tag

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

        img_paths = [(d_root / str(r.path)).as_posix() for r in m_sub.itertuples(index=False)]
        keys = m_sub["image_key"].astype(str).tolist()
        widths = m_sub["width"].to_numpy(dtype=np.float64)
        heights = m_sub["height"].to_numpy(dtype=np.float64)

        records: list[pd.DataFrame] = []
        n = len(img_paths)
        for start in range(0, n, self.batch_size):
            end = min(n, start + self.batch_size)
            batch_paths = img_paths[start:end]
            pred_kwargs: dict[str, Any] = {
                "source": batch_paths,
                "imgsz": self.imgsz,
                "conf": self.conf,
                "iou": self.iou,
                "max_det": self.max_det,
                "verbose": False,
                "stream": False,
            }
            if self.device is not None:
                pred_kwargs["device"] = self.device

            results = self.model.predict(**pred_kwargs)
            for idx_in_batch, res in enumerate(results):
                g_idx = start + idx_in_batch
                key = keys[g_idx]
                orig_w = widths[g_idx]
                orig_h = heights[g_idx]

                boxes_obj = res.boxes
                if boxes_obj is None or len(boxes_obj) == 0:
                    continue

                xyxy = boxes_obj.xyxy.detach().cpu().numpy()
                scores = boxes_obj.conf.detach().cpu().numpy().astype(np.float32)
                cls_ids = boxes_obj.cls.detach().cpu().numpy().astype(np.int8)

                raw_xywh = np.stack(
                    [
                        xyxy[:, 0],
                        xyxy[:, 1],
                        xyxy[:, 2] - xyxy[:, 0],
                        xyxy[:, 3] - xyxy[:, 1],
                    ],
                    axis=1,
                )
                clipped_xywh, valid = clip_and_filter_boxes_xywh(raw_xywh, orig_w, orig_h)
                if not np.any(valid):
                    continue

                v_cls = cls_ids[valid]
                v_scores = scores[valid]
                df_frame = pd.DataFrame(
                    {
                        "image_key": key,
                        "cls": v_cls,
                        "score": v_scores,
                        "x": clipped_xywh[:, 0],
                        "y": clipped_xywh[:, 1],
                        "w": clipped_xywh[:, 2],
                        "h": clipped_xywh[:, 3],
                        "source": self.source_tag,
                    }
                )
                records.append(df_frame)

        if not records:
            return empty_predictions_df()
        combined = pd.concat(records, ignore_index=True)
        return normalize_predictions_df(combined, conf_thresh=self.conf, max_det_per_image=self.max_det)


def train_ultralytics(
    config: dict[str, Any],
    run_id: str | None = None,
    evaluate_full_val: bool = False,
) -> dict[str, Any]:
    """Train a YOLO11 or RT-DETR model under Protocol P0 and log all artifacts locally (§7, §9, F3, F4)."""
    seed = int(config.get("seed", DEFAULT_SEED))
    seed_everything(seed)

    family = str(config.get("family", "yolo"))
    exp = str(config.get("exp", "e0"))
    resolved_run_id = run_id or config.get("run_id") or make_run_id(family, exp)

    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    input_cfg = config.get("input", {}) if isinstance(config.get("input"), dict) else {}
    train_cfg = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    aug_cfg = config.get("augmentation", {}) if isinstance(config.get("augmentation"), dict) else {}
    inf_cfg = config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}

    cache_name = str(config.get("cache", data_cfg.get("cache_name", "k5_drop_L1920")))
    cache_dir = P.cache / cache_name
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache directory not found: {cache_dir}. Run `python scripts/build_cache.py` first."
        )

    train_stride = int(config.get("train_subset_stride", 1))
    max_train_images = config.get("max_train_images", None)
    val_on_train = bool(config.get("val_on_train", False))
    runtime_data_yaml = _prepare_runtime_data_yaml(
        cache_dir=cache_dir,
        train_stride=train_stride,
        max_train_images=int(max_train_images) if max_train_images is not None else None,
        val_on_train=val_on_train,
    )

    logger = RunLogger(
        run_id=resolved_run_id,
        config=config,
        seed=seed,
        hypothesis=str(config.get("hypothesis", "")),
        parent_run=str(config.get("parent_run", "")),
        unit=str(config.get("unit", "U1.1")),
        family=family,
        exp=exp,
    )

    model_weights = str(config.get("model", "yolo11s.pt"))
    pretrained_weights = config.get("pretrained_weights", None)
    imgsz = int(config.get("imgsz", input_cfg.get("imgsz", 1280)))
    batch = int(config.get("batch", 8))
    workers = int(config.get("workers", 4))
    epochs = int(config.get("epochs", 100))
    time_budget_h = config.get("time_budget_h", train_cfg.get("time_budget_h", None))
    close_mosaic = int(config.get("close_mosaic", 2))
    device = config.get("device", None)

    model = _load_ultralytics_model(model_weights, family=family)
    if pretrained_weights:
        model = model.load(str(pretrained_weights))

    ultra_kwargs: dict[str, Any] = {
        "data": runtime_data_yaml.as_posix(),
        "project": P.runs.as_posix(),
        "name": resolved_run_id,
        "exist_ok": True,
        "imgsz": imgsz,
        "batch": batch,
        "epochs": epochs,
        "workers": workers,
        "seed": seed,
        "deterministic": True,
        "max_det": int(inf_cfg.get("max_det", MAX_SUBMISSION_DETS)),
        "conf": float(inf_cfg.get("conf_thresh", 0.001)),
        "iou": float(inf_cfg.get("nms_iou", 0.6)),
        "fliplr": float(aug_cfg.get("hflip", 0.5)),
        "flipud": float(aug_cfg.get("vflip", 0.0)),
        "degrees": float(aug_cfg.get("degrees", 0.0)),
        "scale": float(aug_cfg.get("scale_min", 0.5)),
        "hsv_h": float(aug_cfg.get("hsv_h", 0.015)),
        "hsv_s": float(aug_cfg.get("hsv_s", 0.4)),
        "hsv_v": float(aug_cfg.get("hsv_v", 0.3)),
        "close_mosaic": close_mosaic,
        "amp": True,
        "verbose": False,
        "plots": True,
    }
    if time_budget_h is not None and float(time_budget_h) > 0:
        ultra_kwargs["time"] = float(time_budget_h)
    if device is not None:
        ultra_kwargs["device"] = device
    if "mosaic" in config:
        ultra_kwargs["mosaic"] = float(config["mosaic"])
    if "lr0" in config:
        ultra_kwargs["lr0"] = float(config["lr0"])
    if "optimizer" in config:
        ultra_kwargs["optimizer"] = str(config["optimizer"])

    # Custom trainer hooks for F3-E2 (CIoU + NWD) and F4-E2 (RT-DETR matching / denoising)
    loss_type = str(config.get("loss_type", "ciou")).lower()
    lambda_nwd = float(config.get("lambda_nwd", 0.0))
    if loss_type == "ciou_nwd" or lambda_nwd > 0.0:
        lam = lambda_nwd if lambda_nwd > 0.0 else 0.5
        nwd_c = float(config.get("nwd_c", 24.0))
        ultra_kwargs["trainer"] = make_nwd_detection_trainer(lambda_nwd=lam, nwd_c=nwd_c)
    elif (
        config.get("rtdetr_cost_gain") is not None
        or config.get("rtdetr_loss_gain") is not None
        or config.get("rtdetr_num_denoising") is not None
    ):
        ultra_kwargs["trainer"] = make_rtdetr_custom_trainer(
            cost_gain=config.get("rtdetr_cost_gain"),
            loss_gain=config.get("rtdetr_loss_gain"),
            num_denoising=config.get("rtdetr_num_denoising"),
        )

    model.train(**ultra_kwargs)

    ultra_out_dir = P.runs / resolved_run_id
    metrics_df = adapt_ultralytics_run(ultra_out_dir, logger.run_dir)
    if not metrics_df.empty:
        curves({resolved_run_id: metrics_df}, logger.plots_dir / f"curves_{family}.png")

    # Copy best checkpoint to P.ckpt
    P.ckpt.mkdir(parents=True, exist_ok=True)
    best_pt = ultra_out_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = ultra_out_dir / "weights" / "last.pt"
    ckpt_dest = P.ckpt / f"{resolved_run_id}_best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, ckpt_dest)

    eval_metrics: dict[str, Any] | None = None
    if evaluate_full_val and ckpt_dest.exists():
        manifest_df = load_manifest()
        val_m = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
        max_val_frames = config.get("max_val_frames", None)
        if max_val_frames is not None:
            val_m = val_m.head(int(max_val_frames))

        predictor = UltralyticsPredictor(
            weights_path=ckpt_dest,
            imgsz=imgsz,
            conf=float(inf_cfg.get("conf_thresh", 0.001)),
            iou=float(inf_cfg.get("nms_iou", 0.6)),
            max_det=int(inf_cfg.get("max_det", MAX_SUBMISSION_DETS)),
            batch_size=max(1, batch),
            device=device,
            source_tag=f"{family}-{exp}",
        )
        preds_val_df = predictor.predict(val_m)

        # Save full predictions in runs/ and slim predictions (score >= 0.05) in experiments/runs/<run_id>/
        save_predictions_parquet(preds_val_df, ultra_out_dir / "preds_val.parquet", conf_thresh=0.001)
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
