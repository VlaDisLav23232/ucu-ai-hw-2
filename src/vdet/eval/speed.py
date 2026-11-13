"""FPS benchmark protocol for Ultralytics (.pt) and Torchvision (.pth) models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from vdet.data.manifest import load_manifest
from vdet.models.tv_builders import build_tv_model
from vdet.models.ultra import _load_ultralytics_model
from vdet.paths import P
from vdet.tracking.aggregate import aggregate_runs
from vdet.utils import get_gpu_info, load_json, load_yaml, save_json


def _preload_resized_frames(
    n_frames: int = 220,
    long_side: int = 1280,
) -> list[np.ndarray]:
    """Pre-decode and resize `n_frames` val images to `long_side` in RAM so JPEG decode is excluded from timing."""
    manifest_df = load_manifest()
    val_m = manifest_df[manifest_df["split"] == "val"].head(n_frames).reset_index(drop=True)
    frames_rgb: list[np.ndarray] = []
    for r in val_m.itertuples(index=False):
        img_path = P.data / str(r.path)
        bgr = cv2.imread(img_path.as_posix(), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        scale = float(long_side) / float(max(h, w))
        new_w = max(32, int(round(w * scale)))
        new_h = max(32, int(round(h * scale)))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        frames_rgb.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    return frames_rgb


def benchmark_checkpoint_speed(
    weights_path: str | Path,
    run_id: str | None = None,
    config_path: str | Path | None = None,
    imgsz: int = 1280,
    warmup: int = 20,
    timed: int = 200,
    fp16: bool = True,
    device: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure median latency and FPS over `timed` pre-decoded val frames after `warmup` frames.

    Torchvision checkpoints need their model config: pass it as a dict (`config`) or a YAML path.
    """
    w_path = Path(weights_path)
    dev_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)
    use_half = bool(fp16) and dev.type == "cuda"

    total_needed = warmup + timed
    frames_rgb = _preload_resized_frames(n_frames=total_needed, long_side=imgsz)
    if len(frames_rgb) < total_needed and len(frames_rgb) > 0:
        while len(frames_rgb) < total_needed:
            frames_rgb.append(frames_rgb[len(frames_rgb) % len(frames_rgb)])

    is_tv = w_path.suffix.lower() in {".pth", ".ckpt"}
    latencies_ms: list[float] = []

    if is_tv:
        ckpt = torch.load(w_path, map_location="cpu", weights_only=False)
        cfg = (
            config
            if config is not None
            else load_yaml(Path(config_path))
            if config_path is not None
            else (ckpt.get("config", {"family": "faster_rcnn"}) if isinstance(ckpt, dict) else {"family": "faster_rcnn"})
        )
        model = build_tv_model(cfg, pretrained=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        model.to(dev).eval()

        tensors = [
            torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).to(device=dev, dtype=torch.float32) / 255.0
            for f in frames_rgb
        ]
        with torch.no_grad():
            for idx, t in enumerate(tensors):
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                with torch.amp.autocast(device_type=dev.type, enabled=use_half):
                    _ = model([t])
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if idx >= warmup:
                    latencies_ms.append(dt_ms)
    else:
        model = _load_ultralytics_model(w_path)
        for idx, frame in enumerate(frames_rgb):
            # Ultralytics predict on numpy BGR image
            bgr = frame[:, :, ::-1]
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            _ = model.predict(
                source=bgr,
                imgsz=imgsz,
                conf=0.001,
                iou=0.6,
                max_det=500,
                device=dev_str,
                verbose=False,
            )
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if idx >= warmup:
                latencies_ms.append(dt_ms)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    median_ms = float(np.median(arr))
    mean_ms = float(np.mean(arr))
    p95_ms = float(np.percentile(arr, 95))
    fps = round(1000.0 / max(median_ms, 1e-6), 2)

    gpu_info = get_gpu_info()
    result = {
        "weights": w_path.as_posix(),
        "run_id": run_id or w_path.stem.replace("_best", ""),
        "imgsz": int(imgsz),
        "batch": 1,
        "fp16": use_half,
        "warmup_frames": int(warmup),
        "timed_frames": int(len(arr)),
        "median_latency_ms": round(median_ms, 3),
        "mean_latency_ms": round(mean_ms, 3),
        "p95_latency_ms": round(p95_ms, 3),
        "fps": fps,
        "gpu_type": gpu_info.get("device_name", "CPU"),
    }

    resolved_run_id = result["run_id"]
    run_dir = P.exp_runs_dir / resolved_run_id
    if run_dir.exists():
        save_json(result, run_dir / "speed.json")
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            summary["fps"] = fps
            save_json(summary, summary_path)
            aggregate_runs()

    return result
