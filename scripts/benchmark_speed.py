#!/usr/bin/env python3
"""CLI to benchmark inference FPS under Protocol P0 (§7) and update experiments/runs/<run_id>/speed.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from vdet.eval.speed import benchmark_checkpoint_speed
from vdet.utils import timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference FPS under Protocol P0 (§7).")
    parser.add_argument("--weights", type=Path, required=True, help="Path to checkpoint (.pt or .pth).")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run_id to update in experiments/runs/.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config for Torchvision models.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with timer(f"benchmark_speed ({args.weights.name})"):
        res = benchmark_checkpoint_speed(
            weights_path=args.weights,
            run_id=args.run_id,
            config_path=args.config,
            imgsz=args.imgsz,
            warmup=args.warmup,
            timed=args.timed,
            device=args.device,
        )
    print(
        f"=== FPS Benchmark: {res['run_id']} ===\n"
        f"  GPU           : {res['gpu_type']}\n"
        f"  Median latency: {res['median_latency_ms']:.2f} ms (p95={res['p95_latency_ms']:.2f} ms)\n"
        f"  Throughput    : {res['fps']:.2f} FPS"
    )


if __name__ == "__main__":
    main()

