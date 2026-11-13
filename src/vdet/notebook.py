"""Helpers shared by the notebooks: training runs, inference, evaluation, submissions, display.

Training helpers are resumable: a run whose checkpoint and logged summary already exist is
not trained again; its logged training history and validation metrics are loaded instead
(delete ``checkpoints/<run_id>_best.*`` to retrain from scratch).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vdet.config import resolve_config
from vdet.constants import MAX_SUBMISSION_DETS
from vdet.paths import P
from vdet.utils import load_json, seed_everything, timer

DEFAULT_COLS = ["run_id", "family", "exp", "imgsz", "epochs_done", "gpu_h",
                "AP", "AP50", "AP75", "APs", "APm", "APl", "AR500", "fps"]


# --------------------------------------------------------------------------------------
# Setup & small utilities
# --------------------------------------------------------------------------------------
def setup_notebook(seed: int = 42) -> None:
    """Seed everything, create output dirs and print where the project paths resolve to."""
    import subprocess

    seed_everything(seed)
    P.ensure_writable_dirs()
    try:   # nvidia-smi: does not initialise CUDA (and its ~2.8 GB host memory) inside the notebook process
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, check=True).stdout.strip().splitlines()[0]
    except Exception:
        gpu = "CPU only"
    print(f"Project root : {P.root}")
    print(f"Data         : {P.data} (exists={P.data.exists()})")
    print(f"Cache        : {P.cache} (exists={P.cache.exists()})")
    print(f"Checkpoints  : {P.ckpt}")
    print(f"Device       : {gpu}")


def run_summary(run_id: str) -> dict[str, Any] | None:
    """Return experiments/runs/<run_id>/summary.json (or None if the run was never logged)."""
    p = P.exp_runs_dir / run_id / "summary.json"
    return load_json(p) if p.exists() else None


def results_table(run_ids: list[str] | None = None, families: list[str] | None = None,
                  cols: list[str] | None = None, refresh: bool = True) -> pd.DataFrame:
    """Rows of experiments/results.csv for the given run_ids / families, in the given order."""
    if refresh:
        from vdet.tracking.aggregate import aggregate_runs

        aggregate_runs()
    df = pd.read_csv(P.results_csv)
    if run_ids is not None:
        df = df[df["run_id"].isin(run_ids)].set_index("run_id").reindex(
            [r for r in run_ids if r in set(df["run_id"])]).reset_index()
    if families is not None:
        df = df[df["family"].isin(families)]
    cols = [c for c in (cols or DEFAULT_COLS) if c in df.columns]
    return df[cols].reset_index(drop=True)


def show_image(name: str | Path, width: int = 900) -> None:
    """Display an image from images/ (or any path) inline."""
    from IPython.display import Image, display

    p = Path(name)
    if not p.is_absolute() and not p.exists():
        p = P.images / p
    if p.exists():
        display(Image(filename=p.as_posix(), width=width))
    else:
        print(f"[missing figure] {p}")


def figure_or_generate(name: str, generate_fn, regenerate: bool = False, width: int = 900) -> None:
    """Show images/<name>; call `generate_fn()` first if the file is missing (or `regenerate`)."""
    p = P.images / name
    if regenerate or not p.exists():
        generate_fn()
    show_image(p, width=width)


# --------------------------------------------------------------------------------------
# Training experiments
# --------------------------------------------------------------------------------------
def _report(run_id: str) -> dict[str, Any] | None:
    s = run_summary(run_id)
    if s and s.get("AP") not in (None, ""):
        fps = s.get("fps", "")
        print(f"[{run_id}] AP={float(s['AP']):.4f} | AP50={float(s['AP50']):.4f} | "
              f"APs={float(s['APs']):.4f} | AR500={float(s['AR500']):.4f} | "
              f"epochs={s.get('epochs_done')} | gpu_h={s.get('gpu_h')} | fps={fps}")
    elif s:
        print(f"[{run_id}] logged without validation metrics (smoke test).")
    return s


def training_log(run_id: str, cols: list[str] | None = None) -> pd.DataFrame:
    """Per-epoch training log of a run (experiments/runs/<run_id>/metrics.csv)."""
    p = P.exp_runs_dir / run_id / "metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    keep = [c for c in (cols or ["epoch", "gpu_time_h", "lr", "train_loss", "loss_box", "loss_cls",
                                 "val_AP", "val_AP50", "val_APs"]) if c in df.columns]
    return df[keep].dropna(axis=1, how="all")


def _show_log(run_id: str) -> None:
    from IPython.display import display

    log = training_log(run_id)
    if not log.empty:
        display(log.round(5))


def _run_isolated(target: str, kwargs: dict[str, Any], expandable: bool = False) -> None:
    """Run `module.function(**kwargs)` in a fresh Python process and stream its output into the notebook.

    GPU-heavy phases (training, prediction, benchmarking) run in their own process so that all of their host and
    GPU memory is released when they finish. The machine has 7.6 GB RAM: keeping a finished training process alive
    while the validation predictions are evaluated exhausts memory.

    `expandable=True` enables PyTorch's `expandable_segments` allocator: torchvision detection training otherwise
    fragments the allocator until it reserves more than the 6 GB card has and crashes with "CUDA error: out of memory".
    Ultralytics runs must NOT use it (its out-of-memory fallback in the TaskAlignedAssigner trips a PyTorch assert).
    """
    import os
    import pickle
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
        pickle.dump(kwargs, fh)
        path = fh.name
    code = ("import importlib, pickle, sys; m, f = sys.argv[1].rsplit('.', 1); "
            "kw = pickle.load(open(sys.argv[2], 'rb')); getattr(importlib.import_module(m), f)(**kw)")
    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if expandable:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    proc = subprocess.Popen([sys.executable, "-u", "-c", code, target, path], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=str(P.root), env=env)
    try:
        for raw in proc.stdout:                       # progress bars update with '\r': keep the last state of each line
            print(raw.decode("utf-8", errors="replace").rstrip("\n").rsplit("\r", 1)[-1], flush=True)
    finally:
        rc = proc.wait()
        os.unlink(path)
    if rc != 0:
        raise RuntimeError(f"{target} failed with exit code {rc}")


def _is_done(run_id: str, cfg: dict[str, Any]) -> bool:
    """A run is finished when its summary exists and contains validation metrics (smoke tests have none)."""
    s = run_summary(run_id)
    return s is not None and (cfg.get("exp") == "smoke" or s.get("AP") not in (None, ""))


def _finish_interrupted_run(run_id: str, resolved_cfg: dict[str, Any], ckpt: Path) -> None:
    """Evaluate the saved `preds_val.parquet` of a trained run and write eval_val.json / summary.json."""
    from vdet.eval.coco_eval import evaluate
    from vdet.infer.predictors import save_predictions_parquet
    from vdet.tracking.aggregate import aggregate_runs
    from vdet.tracking.run import RunLogger

    preds = pd.read_parquet(P.runs / run_id / "preds_val.parquet")
    subset = preds["image_key"].astype(str).unique().tolist() if resolved_cfg.get("max_val_frames") else None
    metrics = evaluate(preds, split="val", subset=subset, per_res_bucket=True)
    logger = RunLogger(run_id=run_id, config=resolved_cfg, seed=int(resolved_cfg.get("seed", 42)),
                       hypothesis=str(resolved_cfg.get("hypothesis", "")), parent_run=str(resolved_cfg.get("parent_run", "")),
                       family=str(resolved_cfg.get("family", "yolo")), exp=str(resolved_cfg.get("exp", "e0")))
    slim = logger.run_dir / "preds_val_slim.parquet"
    if not slim.exists():
        save_predictions_parquet(preds, slim, conf_thresh=float(resolved_cfg.get("inference", {}).get("slim_conf_thresh", 0.05)))
    logger.finalize(eval_metrics=metrics, checkpoint_uri=ckpt.as_posix())
    aggregate_runs()


def _evaluate_run(run_id: str, resolved: dict[str, Any], ckpt: Path, batch: int, expandable: bool = False) -> None:
    """Val predictions (own process, skipped if saved) -> official evaluation -> run summary."""
    preds_path = P.runs / run_id / "preds_val.parquet"
    if not preds_path.exists():
        is_tv = ckpt.suffix.lower() in {".pth", ".ckpt"}
        _run_isolated("vdet.notebook.predict_split", dict(
            weights=str(ckpt), split="val", out=str(preds_path), imgsz=int(resolved.get("imgsz", 1280)), batch=batch,
            tv_config=resolved if is_tv else None, max_frames=resolved.get("max_val_frames"),
            source_tag=f"{resolved.get('family', 'run')}-{resolved.get('exp', '')}"), expandable=expandable)
    _finish_interrupted_run(run_id, resolved, ckpt)


def _benchmark(ckpt: Path, run_id: str, imgsz: int, tv_config: dict[str, Any] | None = None,
               expandable: bool = False) -> None:
    if not (P.exp_runs_dir / run_id / "speed.json").exists() and ckpt.exists():
        _run_isolated("vdet.eval.speed.benchmark_checkpoint_speed",
                      dict(weights_path=str(ckpt), run_id=run_id, imgsz=imgsz, config=tv_config), expandable=expandable)


def train_ultralytics_run(cfg: dict[str, Any],
                          overrides: dict[str, Any] | None = None,
                          benchmark: bool = True, benchmark_imgsz: int | None = None,
                          predict_test: bool = False, test_batch: int = 2) -> dict[str, Any] | None:
    """Train (if missing) + evaluate + benchmark a YOLO11 / RT-DETR experiment defined by `cfg`.

    Resumable: a finished run is not trained again; a run that was trained but not evaluated is only evaluated.
    """
    run_id = cfg["run_id"]
    ckpt = P.ckpt / f"{run_id}_best.pt"
    resolved = resolve_config(cfg, overrides)
    if _is_done(run_id, resolved):
        print(f"[{run_id}] training completed earlier ({ckpt.name}); loading its training log.")
    else:
        if not ckpt.exists():
            with timer(f"train {run_id}"):
                _run_isolated("vdet.models.ultra.train_ultralytics",
                              dict(config=resolved, run_id=run_id, evaluate_full_val=False))
        else:
            print(f"[{run_id}] training finished earlier; evaluating.")
        if bool(resolved.get("auto_eval_val", True)):
            _evaluate_run(run_id, resolved, ckpt, batch=max(1, int(resolved.get("batch", 8))))
    _show_log(run_id)

    if benchmark:
        _benchmark(ckpt, run_id, benchmark_imgsz or int(cfg.get("imgsz", 1280)))
    if predict_test:
        test_submission(run_id, weights=ckpt, imgsz=int(cfg.get("imgsz", 1280)), batch=test_batch)
    return _report(run_id)


def train_torchvision_run(cfg: dict[str, Any], benchmark: bool = True) -> dict[str, Any] | None:
    """Train (if missing) + evaluate + benchmark a Faster R-CNN / RetinaNet experiment.

    Torchvision configs are used as-is (not merged with the protocol). Resumable: an interrupted run continues from
    its last finished epoch; a trained but not evaluated run is only evaluated.
    """
    run_id = cfg["run_id"]
    ckpt = P.ckpt / f"{run_id}_best.pth"
    if _is_done(run_id, cfg):
        print(f"[{run_id}] training completed earlier ({ckpt.name}); loading its training log.")
    else:
        if not ckpt.exists():
            last = P.runs / run_id / "last.pth"
            run_cfg = cfg
            if last.exists() and not cfg.get("resume_from"):
                run_cfg = dict(cfg, resume_from=last.as_posix())
                print(f"[{run_id}] interrupted earlier; resuming from {last.name}.")
            with timer(f"train {run_id}"):
                _run_isolated("vdet.engine.train_tv.train_torchvision",
                              dict(config=run_cfg, run_id=run_id, evaluate_full_val=False), expandable=True)
        else:
            print(f"[{run_id}] training finished earlier; evaluating.")
        if bool(cfg.get("evaluate_full_val", cfg.get("exp") != "smoke")):
            _evaluate_run(run_id, cfg, ckpt, batch=max(1, int(cfg.get("batch", 4))), expandable=True)
    _show_log(run_id)

    if benchmark and cfg.get("exp") != "smoke":
        _benchmark(ckpt, run_id, int(cfg.get("imgsz", 1280)), tv_config=cfg, expandable=True)
    return _report(run_id)


# --------------------------------------------------------------------------------------
# Inference, evaluation & submissions
# --------------------------------------------------------------------------------------
def predict_split(weights: str | Path, split: str, out: str | Path, imgsz: int = 1280,
                  batch: int = 4, tv_config: dict[str, Any] | None = None,
                  tta: bool = False, tta_scales=(1.0, 1.25), tiling: bool = False,
                  slice_size: int = 800, overlap: float = 0.25, max_frames: int | None = None,
                  source_tag: str = "ultra") -> pd.DataFrame:
    """Run a checkpoint (.pt Ultralytics or .pth torchvision) on original val/test frames."""
    from vdet.data.manifest import load_manifest
    from vdet.infer.predictors import save_predictions_parquet
    from vdet.infer.tiling import TiledPredictor
    from vdet.infer.tta import TTAPredictor

    sub = load_manifest()
    sub = sub[sub["split"] == split].reset_index(drop=True)
    if max_frames is not None:
        sub = sub.head(max_frames).reset_index(drop=True)

    weights = Path(weights)
    if weights.suffix.lower() in {".pth", ".ckpt"}:
        from vdet.models.tv_builders import TorchvisionPredictor

        predictor = TorchvisionPredictor(weights_path=weights, config=tv_config, conf=0.001,
                                         max_det=MAX_SUBMISSION_DETS, batch_size=batch, source_tag=source_tag)
    else:
        from vdet.models.ultra import UltralyticsPredictor

        predictor = UltralyticsPredictor(weights_path=weights, imgsz=imgsz, conf=0.001, iou=0.6,
                                         max_det=MAX_SUBMISSION_DETS, batch_size=batch, source_tag=source_tag)
    if tta:
        predictor = TTAPredictor(base_predictor=predictor, scales=tta_scales, hflip=True, conf_thresh=0.001,
                                 max_det=MAX_SUBMISSION_DETS, source_tag=source_tag)
    if tiling:
        predictor = TiledPredictor(base_predictor=predictor, slice_size=slice_size, overlap_ratio=overlap,
                                   conf_thresh=0.001, max_det=MAX_SUBMISSION_DETS, source_tag=source_tag)
    with timer(f"predict {source_tag} ({split}, n={len(sub)})"):
        df = predictor.predict(sub, data_root=P.data)
    save_predictions_parquet(df, out, conf_thresh=0.001, max_det_per_image=MAX_SUBMISSION_DETS)
    return df


def make_submission(preds: str | Path | pd.DataFrame, out_csv: str | Path,
                    overwrite: bool = False) -> dict[str, Any]:
    """Format test predictions into a Kaggle CSV (if missing) and validate it."""
    from vdet.submission import to_csv, validate

    out_csv = Path(out_csv)
    if overwrite or not out_csv.exists():
        df = pd.read_parquet(preds) if not isinstance(preds, pd.DataFrame) else preds
        to_csv(df, out_csv, P.test_images_csv)
    return validate(out_csv, P.test_images_csv)


def test_submission(run_id: str, weights: str | Path, imgsz: int = 1280, batch: int = 2,
                    tv_config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Predict the test split (if not done yet), write submissions/<run_id>.csv and validate it."""
    preds = P.runs / run_id / "preds_test.parquet"
    if not preds.exists():
        _run_isolated("vdet.notebook.predict_split", dict(weights=str(weights), split="test", out=str(preds), imgsz=imgsz,
                                                           batch=batch, tv_config=tv_config, source_tag=run_id),
                      expandable=Path(weights).suffix.lower() in {".pth", ".ckpt"})
    rep = make_submission(preds, P.subs / f"{run_id}.csv")
    print(f"[{run_id}] submission {P.subs / f'{run_id}.csv'}: {rep}")
    return rep


def evaluate_preds(preds: str | Path | pd.DataFrame, split: str = "val",
                   label: str | None = None, per_res_bucket: bool = False) -> dict[str, Any]:
    """Official local evaluation (COCO AP@[.5:.95], maxDets 500, ignore regions as crowd)."""
    from vdet.eval.coco_eval import evaluate

    df = pd.read_parquet(preds) if not isinstance(preds, pd.DataFrame) else preds
    m = evaluate(df, split=split, per_res_bucket=per_res_bucket)
    if label:
        print(f"[{label}] AP={m['AP']:.4f} | AP50={m['AP50']:.4f} | AP75={m['AP75']:.4f} | "
              f"APs={m['APs']:.4f} | APm={m['APm']:.4f} | APl={m['APl']:.4f} | AR500={m['AR500']:.4f}")
    return m
