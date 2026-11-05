"""Class ontology, ID mappings, and competition constants (§2.3, §2.4)."""

from __future__ import annotations

# 0-based class names used by YOLO and the Kaggle submission format
CLASS_NAMES: tuple[str, ...] = (
    "person",    # 0 -> COCO 1 (VisDrone 1 pedestrian + 2 people)
    "bicycle",   # 1 -> COCO 2 (VisDrone 3 bicycle)
    "car",       # 2 -> COCO 3 (VisDrone 4 car)
    "van",       # 3 -> COCO 4 (VisDrone 5 van)
    "truck",     # 4 -> COCO 5 (VisDrone 6 truck)
    "tricycle",  # 5 -> COCO 6 (VisDrone 7 tricycle + 8 awning-tricycle)
    "bus",       # 6 -> COCO 7 (VisDrone 9 bus)
    "motor",     # 7 -> COCO 8 (VisDrone 10 motor)
)

NUM_CLASSES: int = len(CLASS_NAMES)

CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# YOLO (0..7) <-> COCO category_id (1..8)
YOLO_TO_COCO_ID: dict[int, int] = {idx: idx + 1 for idx in range(NUM_CLASSES)}
COCO_TO_YOLO_ID: dict[int, int] = {idx + 1: idx for idx in range(NUM_CLASSES)}

# Original 10+2 VisDrone category IDs -> 0-based 8-class ID (None for ignore / others)
VISDRONE_ORIG_TO_YOLO_ID: dict[int, int | None] = {
    0: None,   # ignored regions
    1: 0,      # pedestrian -> person
    2: 0,      # people -> person
    3: 1,      # bicycle -> bicycle
    4: 2,      # car -> car
    5: 3,      # van -> van
    6: 4,      # truck -> truck
    7: 5,      # tricycle -> tricycle
    8: 5,      # awning-tricycle -> tricycle
    9: 6,      # bus -> bus
    10: 7,     # motor -> motor
    11: None,  # others -> ignored
}

# COCO category list for building ground-truth JSONs
COCO_CATEGORIES: list[dict[str, int | str]] = [
    {"id": idx + 1, "name": name} for idx, name in enumerate(CLASS_NAMES)
]

# Distinct class palette (RGB tuples in [0, 255]) for visual debugging & plots
CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    0: (230, 57, 70),    # person - red
    1: (244, 162, 97),   # bicycle - orange
    2: (42, 157, 143),   # car - teal
    3: (69, 123, 157),   # van - steel blue
    4: (155, 93, 229),   # truck - purple
    5: (233, 196, 106),  # tricycle - gold
    6: (241, 91, 181),   # bus - magenta
    7: (0, 180, 216),    # motor - cyan
}

# Official split counts verified on local dataset (§2.1)
EXPECTED_SPLIT_FRAMES: dict[str, int] = {
    "train": 24_201,
    "val": 2_846,
    "test": 3_811,
}

EXPECTED_SPLIT_CLIPS: dict[str, int] = {
    "train": 56,
    "val": 7,
    "test": 48,
}

EXPECTED_SPLIT_BOXES: dict[str, int] = {
    "train": 1_105_516,
    "val": 114_132,
}

# COCO evaluation defaults (§3.1)
DEFAULT_MAX_DETS: list[int] = [1, 10, 500]
MAX_SUBMISSION_DETS: int = 500
DEFAULT_SEED: int = 42

