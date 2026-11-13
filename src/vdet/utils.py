"""Seeding, timing, git metadata, hardware info, and YAML/JSON I/O utilities."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from vdet.constants import DEFAULT_SEED
from vdet.paths import P, detect_env, find_repo_root


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch (if installed) RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


@contextmanager
def timer(label: str | None = None) -> Iterator[dict[str, float]]:
    """Context manager measuring elapsed wall-clock seconds."""
    state: dict[str, float] = {"elapsed_s": 0.0}
    t0 = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed_s"] = time.perf_counter() - t0
        if label:
            print(f"[timer] {label}: {state['elapsed_s']:.2f}s")


def get_git_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Return current git commit hash, short hash, branch, and dirty flag."""
    root = repo_root or find_repo_root()
    info: dict[str, Any] = {
        "commit": "unknown",
        "commit_short": "unknown",
        "branch": "unknown",
        "dirty": False,
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        short = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info.update(
            {
                "commit": commit,
                "commit_short": short,
                "branch": branch,
                "dirty": bool(status),
            }
        )
    except Exception:
        pass
    return info


def get_gpu_info() -> dict[str, Any]:
    """Return GPU availability, device name, count, and VRAM in GB."""
    gpu: dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "cpu",
        "vram_gb": 0.0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu.update(
                {
                    "cuda_available": True,
                    "device_count": int(torch.cuda.device_count()),
                    "device_name": str(props.name),
                    "vram_gb": round(props.total_memory / (1024**3), 2),
                }
            )
    except ImportError:
        pass
    return gpu


def get_env_metadata(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Collect full environment metadata required for env.json."""
    git_info = get_git_info()
    gpu_info = get_gpu_info()

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg_name, mod_name in [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("pyarrow", "pyarrow"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("faster_coco_eval", "faster_coco_eval"),
        ("opencv", "cv2"),
    ]:
        try:
            mod = __import__(mod_name)
            versions[pkg_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg_name] = "not_installed"

    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host_env": detect_env(),
        "platform": platform.platform(),
        "seed": seed,
        "git": git_info,
        "gpu": gpu_info,
        "versions": versions,
    }


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Write a dictionary to a YAML file, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    return p


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write JSON serializable data to a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)
    return p


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `overrides` into a copy of `base`."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_resolved_config(
    config_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve layered configuration: protocol <- config_path <- cli_overrides."""
    proto_file = Path(protocol_path) if protocol_path else (P.configs / "protocol.yaml")
    if proto_file.exists():
        resolved: dict[str, Any] = load_yaml(proto_file)
    else:
        from vdet.config import PROTOCOL  # protocol embedded in vdet.config

        resolved = copy.deepcopy(PROTOCOL)
    if config_path is not None:
        cfg_file = Path(config_path)
        user_cfg = load_yaml(cfg_file)
        resolved = deep_merge(resolved, user_cfg)
        resolved["_config_path"] = cfg_file.as_posix()
    if cli_overrides:
        resolved = deep_merge(resolved, cli_overrides)
    return resolved


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

