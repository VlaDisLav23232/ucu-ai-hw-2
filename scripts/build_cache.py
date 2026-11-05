#!/usr/bin/env python3
"""CLI to build offline subsampled and resized training caches (U0.4)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vdet.data.cache import build_cache_from_recipe
from vdet.paths import P
from vdet.utils import load_yaml, timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline training/val cache (§6.2, U0.4).")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to cache recipe YAML (e.g. configs/data/cache_k5_blur.yaml).",
    )
    parser.add_argument("--data-root", type=Path, default=P.data)
    parser.add_argument("--cache-root", type=Path, default=P.cache)
    parser.add_argument("--workers", type=int, default=None, help="Override num_workers.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames per split (for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = load_yaml(args.config)
    if args.workers is not None:
        recipe["num_workers"] = args.workers

    cache_name = recipe.get("cache_name", args.config.stem)
    with timer(f"build_cache ({cache_name})"):
        out_dir = build_cache_from_recipe(
            recipe=recipe,
            data_root=args.data_root,
            cache_root=args.cache_root,
            max_frames_per_split=args.max_frames,
        )

    idx_df = pd.read_parquet(out_dir / "index.parquet")
    print(f"\n=== Cache Built: {out_dir.as_posix()} ===")
    for split, group in idx_df.groupby("split"):
        print(f"  {split:5s}: {len(group):5d} frames")
    print(f"  recipe.yaml: {(out_dir / 'recipe.yaml').exists()}")
    print(f"  data.yaml:   {(out_dir / 'data.yaml').exists()}")


if __name__ == "__main__":
    main()

