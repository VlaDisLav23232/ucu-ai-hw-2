"""Environment auto-detection and POSIX pathlib root resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root directory (the one containing pyproject.toml)."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


def detect_env() -> str:
    """Detect runtime environment: 'local_wsl', 'kaggle', or 'colab'."""
    explicit = os.environ.get("VDET_ENV", "").strip().lower()
    if explicit in {"local_wsl", "kaggle", "colab"}:
        return explicit
    if Path("/kaggle/input").exists():
        return "kaggle"
    if Path("/content").exists():
        return "colab"
    return "local_wsl"


def _resolve_path_entry(raw: str | Path, repo_root: Path) -> Path:
    """Convert a config string or env var into an absolute POSIX Path."""
    expanded = os.path.expanduser(os.path.expandvars(str(raw)))
    p = Path(expanded)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


@dataclass(frozen=True)
class Paths:
    """Resolved project directory paths across local WSL, Kaggle, and Colab."""

    env: str
    root: Path
    configs: Path
    data: Path
    cache: Path
    runs: Path
    ckpt: Path
    subs: Path
    experiments: Path
    images: Path

    @property
    def manifest_parquet(self) -> Path:
        return self.cache / "manifest.parquet"

    @property
    def boxes_parquet(self) -> Path:
        return self.cache / "boxes.parquet"

    @property
    def ignore_parquet(self) -> Path:
        return self.cache / "ignore.parquet"

    @property
    def test_images_csv(self) -> Path:
        return self.data / "test_images.csv"

    @property
    def results_csv(self) -> Path:
        return self.experiments / "results.csv"

    @property
    def lb_csv(self) -> Path:
        return self.experiments / "lb.csv"

    @property
    def exp_runs_dir(self) -> Path:
        return self.experiments / "runs"

    def ensure_writable_dirs(self) -> None:
        """Create writable output directories if they do not exist."""
        for d in (self.cache, self.runs, self.ckpt, self.subs, self.exp_runs_dir, self.images):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError:
                # Read-only mounts (e.g. /kaggle/input/...) should not fail
                pass

    def to_dict(self) -> dict[str, str]:
        return {
            "env": self.env,
            "root": self.root.as_posix(),
            "configs": self.configs.as_posix(),
            "data": self.data.as_posix(),
            "cache": self.cache.as_posix(),
            "runs": self.runs.as_posix(),
            "ckpt": self.ckpt.as_posix(),
            "subs": self.subs.as_posix(),
            "experiments": self.experiments.as_posix(),
            "images": self.images.as_posix(),
        }

    def __repr__(self) -> str:
        items = ", ".join(f"{k}='{v}'" for k, v in self.to_dict().items())
        return f"Paths({items})"


_LOCAL_DEFAULTS: dict[str, str] = {
    "data": "data", "cache": "cache", "runs": "runs", "ckpt": "checkpoints",
    "subs": "submissions", "experiments": "experiments", "images": "images",
}

# Per-environment defaults; environment variables VDET_DATA, VDET_CACHE, VDET_RUNS, VDET_CKPT,
# VDET_SUBS, VDET_EXPERIMENTS and VDET_IMAGES always take precedence.
ENV_PATH_DEFAULTS: dict[str, dict[str, str]] = {
    "local_wsl": _LOCAL_DEFAULTS,
    "kaggle": {
        "data": "/kaggle/input/multiple-object-video-detection-hw-2-ucu-ai-2026",
        "cache": "/kaggle/working/cache", "runs": "/kaggle/working/runs",
        "ckpt": "/kaggle/working/checkpoints", "subs": "/kaggle/working/submissions",
        "experiments": "experiments", "images": "images",
    },
    "colab": {
        "data": "/content/data", "cache": "/content/cache", "runs": "/content/runs",
        "ckpt": "/content/drive/MyDrive/vdet/checkpoints", "subs": "/content/submissions",
        "experiments": "experiments", "images": "images",
    },
}


def resolve_paths(env: str | None = None, repo_root: Path | None = None) -> Paths:
    """Resolve all project paths from per-environment defaults, an optional configs/paths.yaml override and env vars."""
    root = find_repo_root(repo_root)
    active_env = env or detect_env()
    configs_dir = root / "configs"
    paths_yaml = configs_dir / "paths.yaml"

    cfg: dict[str, Any] = dict(ENV_PATH_DEFAULTS.get(active_env, _LOCAL_DEFAULTS))
    if paths_yaml.exists():
        with paths_yaml.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            cfg.update(loaded.get(active_env, {}))

    data_raw = os.environ.get("VDET_DATA") or cfg.get("data", "data")
    cache_raw = os.environ.get("VDET_CACHE") or cfg.get("cache", "cache")
    runs_raw = os.environ.get("VDET_RUNS") or cfg.get("runs", "runs")
    ckpt_raw = os.environ.get("VDET_CKPT") or cfg.get("ckpt", "checkpoints")
    subs_raw = os.environ.get("VDET_SUBS") or cfg.get("subs", "submissions")
    exp_raw = os.environ.get("VDET_EXPERIMENTS") or cfg.get("experiments", "experiments")
    img_raw = os.environ.get("VDET_IMAGES") or cfg.get("images", "images")

    return Paths(
        env=active_env,
        root=root,
        configs=configs_dir,
        data=_resolve_path_entry(data_raw, root),
        cache=_resolve_path_entry(cache_raw, root),
        runs=_resolve_path_entry(runs_raw, root),
        ckpt=_resolve_path_entry(ckpt_raw, root),
        subs=_resolve_path_entry(subs_raw, root),
        experiments=_resolve_path_entry(exp_raw, root),
        images=_resolve_path_entry(img_raw, root),
    )


P: Paths = resolve_paths()

