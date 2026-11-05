#!/usr/bin/env python3
"""CLI to run U0.8 EDA and frame-step redundancy analysis, exporting figures to images/."""

from __future__ import annotations

import argparse
from pathlib import Path

from vdet.data.redundancy import generate_all_eda_figures
from vdet.paths import P
from vdet.utils import save_json, timer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate U0.8 EDA figures and redundancy table.")
    parser.add_argument("--cache-root", type=Path, default=P.cache)
    parser.add_argument("--images-dir", type=Path, default=P.images)
    args = parser.parse_args()

    with timer("generate_all_eda_figures"):
        saved, lag_df = generate_all_eda_figures(
            cache_root=args.cache_root,
            images_dir=args.images_dir,
        )

    print("\n=== Frame-Step Redundancy (k = 1..20) ===")
    print(lag_df.to_string(index=False))

    lag_df.to_csv(args.cache_root / "redundancy_lag_k.csv", index=False)
    save_json(
        {k: p.name for k, p in saved.items()},
        args.cache_root / "eda_figures_index.json",
    )

    print("\n=== Saved U0.8 Figures ===")
    for name, p in saved.items():
        kb = p.stat().st_size / 1024.0
        print(f"  {name:24s}: {p.as_posix()} ({kb:.1f} KB)")


if __name__ == "__main__":
    main()

