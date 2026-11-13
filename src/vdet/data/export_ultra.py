"""Export Ultralytics dataset YAML for an offline cache directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vdet.constants import CLASS_NAMES, NUM_CLASSES
from vdet.utils import save_yaml


def write_ultralytics_data_yaml(
    cache_dir: str | Path,
    train_rel: str = "train/images",
    val_rel: str = "val/images",
    out_filename: str = "data.yaml",
) -> Path:
    """Write Ultralytics `data.yaml` pointing to `cache_dir` with absolute POSIX path."""
    c_dir = Path(cache_dir).resolve()
    payload: dict[str, Any] = {
        "path": c_dir.as_posix(),
        "train": train_rel,
        "val": val_rel,
        "nc": NUM_CLASSES,
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    out_path = c_dir / out_filename
    return save_yaml(payload, out_path)

