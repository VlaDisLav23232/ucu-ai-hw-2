"""vdet: Multiple-Object Video Detection (VisDrone-VID) — UCU AI 2026 HW2."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

# On WSL2, ensure the paravirtualized host driver (/usr/lib/wsl/lib/libcuda.so.1)
# is preloaded before any native Linux stub in /usr/lib/x86_64-linux-gnu/.
_WSL_LIBCUDA = Path("/usr/lib/wsl/lib/libcuda.so.1")
if _WSL_LIBCUDA.exists():
    try:
        ctypes.CDLL(str(_WSL_LIBCUDA), mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    except OSError:
        pass

from vdet.constants import (
    CLASS_NAMES,
    COCO_TO_YOLO_ID,
    NUM_CLASSES,
    VISDRONE_ORIG_TO_YOLO_ID,
    YOLO_TO_COCO_ID,
)
from vdet.paths import P, Paths, resolve_paths

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "YOLO_TO_COCO_ID",
    "COCO_TO_YOLO_ID",
    "VISDRONE_ORIG_TO_YOLO_ID",
    "P",
    "Paths",
    "resolve_paths",
]

