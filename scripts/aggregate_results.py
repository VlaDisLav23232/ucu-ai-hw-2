#!/usr/bin/env python3
"""CLI to regenerate experiments/results.csv from experiments/runs/*/summary.json (U0.7)."""

from __future__ import annotations

import argparse
from pathlib import Path

from vdet.paths import P
from vdet.tracking.aggregate import aggregate_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate experiments/results.csv.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=P.experiments,
        help="Path to experiments directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional override path for results.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = aggregate_runs(experiments_dir=args.experiments_dir, out_csv=args.out)
    out_path = args.out or (args.experiments_dir / "results.csv")
    print(f"Aggregated {len(df)} run(s) into {out_path.as_posix()}")


if __name__ == "__main__":
    main()

