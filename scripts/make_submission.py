#!/usr/bin/env python3
"""CLI to format a predictions parquet into a validated Kaggle submission CSV (U0.6)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vdet.constants import MAX_SUBMISSION_DETS
from vdet.paths import P
from vdet.submission import to_csv, validate
from vdet.utils import timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and validate a Kaggle submission CSV.")
    parser.add_argument("--preds", type=Path, required=True, help="Input predictions .parquet file.")
    parser.add_argument("--out", type=Path, required=True, help="Output submission .csv path.")
    parser.add_argument(
        "--test-images-csv",
        type=Path,
        default=P.test_images_csv,
        help="Path to test_images.csv.",
    )
    parser.add_argument("--max-dets", type=int, default=MAX_SUBMISSION_DETS)
    parser.add_argument("--decimals", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds_df = pd.read_parquet(args.preds)
    with timer("to_csv"):
        out_path = to_csv(
            preds_df=preds_df,
            out_csv=args.out,
            test_images_csv=args.test_images_csv,
            max_dets=args.max_dets,
            decimals=args.decimals,
        )
    with timer("validate"):
        stats = validate(out_path, test_images_csv=args.test_images_csv, max_dets=args.max_dets)
    size_mb = out_path.stat().st_size / (1024**2)
    print(f"Submission validated and saved to {out_path.as_posix()} ({size_mb:.2f} MB):", stats)


if __name__ == "__main__":
    main()

