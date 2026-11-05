#!/usr/bin/env python3
"""CLI to build manifest.parquet, boxes.parquet, and deduplicated ignore.parquet (U0.2)."""

from __future__ import annotations

import argparse
from pathlib import Path

from vdet.data.manifest import build_manifests, verify_manifest_counts
from vdet.paths import P
from vdet.utils import timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build parquet manifests for train/val/test.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=P.data,
        help="Path to raw competition data root.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=P.cache,
        help="Path to cache directory where parquet files are written.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Skip strict official split count assertions (useful for synthetic mini tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with timer("build_manifests"):
        manifest_df, boxes_df, ignore_df = build_manifests(
            data_root=args.data_root,
            cache_root=args.cache_root,
            strict_verify=not args.no_strict,
        )

    print("\n=== Manifest Summary ===")
    for split in ("train", "val", "test"):
        sub = manifest_df[manifest_df["split"] == split]
        n_frames = len(sub)
        n_clips = sub["clip"].nunique()
        n_boxes = int(sub["n_boxes"].sum()) if split != "test" else 0
        n_ig_frames = int((sub["n_ignore"] > 0).sum())
        n_ig_regions = int(sub["n_ignore"].sum())
        print(
            f"  {split:5s}: clips={n_clips:2d}, frames={n_frames:6d}, "
            f"boxes={n_boxes:9d}, ignore_frames={n_ig_frames:6d}, ignore_regions={n_ig_regions:6d}"
        )

    if not args.no_strict:
        stats = verify_manifest_counts(manifest_df, boxes_df, ignore_df)
        print("Verified all contract counts and metadata join rates:", stats)

    for fname in ("manifest.parquet", "boxes.parquet", "ignore.parquet"):
        fpath = args.cache_root / fname
        size_mb = fpath.stat().st_size / (1024**2)
        print(f"  Wrote {fpath.as_posix()} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

