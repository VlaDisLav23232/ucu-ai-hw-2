#!/usr/bin/env bash
# Overnight execution script for Local WSL (RTX 4050 6 GB)
# Automatically skips any run whose checkpoint + summary already completed!
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "============================================================"
echo "[1/6] Checking U2.0 Smoke Tests..."
echo "============================================================"
if [ ! -f "checkpoints/frcnn-smoke-local_best.pth" ]; then
    python scripts/train_tv.py --config configs/faster_rcnn/smoke_local.yaml
else
    echo "  -> Skipping frcnn-smoke-local (already completed)."
fi

if [ ! -f "checkpoints/retina-smoke-local_best.pth" ]; then
    python scripts/train_tv.py --config configs/retinanet/smoke_local.yaml
else
    echo "  -> Skipping retina-smoke-local (already completed)."
fi

echo "============================================================"
echo "[2/6] F3 Exp A: YOLO11s High-Res imgsz=1600 (yolo-e1)..."
echo "============================================================"
if [ ! -f "submissions/yolo-e1.csv" ]; then
    python scripts/train_ultra.py --config configs/yolo/e1_imgsz1600.yaml --batch 2 --eval-val
    python scripts/benchmark_speed.py --weights checkpoints/yolo-e1_best.pt --run-id yolo-e1 --imgsz 1600
    python scripts/predict.py --weights checkpoints/yolo-e1_best.pt --split test --batch 2 --imgsz 1600 --out runs/yolo-e1/preds_test.parquet
    python scripts/make_submission.py --preds runs/yolo-e1/preds_test.parquet --out submissions/yolo-e1.csv
else
    echo "  -> Skipping yolo-e1 (already completed: AP=0.2184, submissions/yolo-e1.csv saved)."
fi

echo "============================================================"
echo "[3/6] F3 Exp B: YOLO11s CIoU + NWD Loss Blend (yolo-e2)..."
echo "============================================================"
if [ ! -f "submissions/yolo-e2.csv" ]; then
    python scripts/train_ultra.py --config configs/yolo/e2_nwd.yaml --batch 2 --eval-val
    python scripts/benchmark_speed.py --weights checkpoints/yolo-e2_best.pt --run-id yolo-e2
    python scripts/predict.py --weights checkpoints/yolo-e2_best.pt --split test --batch 4 --out runs/yolo-e2/preds_test.parquet
    python scripts/make_submission.py --preds runs/yolo-e2/preds_test.parquet --out submissions/yolo-e2.csv
else
    echo "  -> Skipping yolo-e2 (already completed)."
fi

echo "============================================================"
echo "[4/6] F4 Baseline: RT-DETR-L nq=300 (rtdetr-e0)..."
echo "============================================================"
if [ ! -f "experiments/runs/rtdetr-e0/speed.json" ]; then
    python scripts/train_ultra.py --config configs/rtdetr/e0_base.yaml --batch 2 --eval-val
    python scripts/benchmark_speed.py --weights checkpoints/rtdetr-e0_best.pt --run-id rtdetr-e0
else
    echo "  -> Skipping rtdetr-e0 (already completed)."
fi

echo "============================================================"
echo "[5/6] F4 Exp A: RT-DETR-L nq=600 Dense Queries (rtdetr-e1)..."
echo "============================================================"
if [ ! -f "submissions/rtdetr-e1.csv" ]; then
    python scripts/train_ultra.py --config configs/rtdetr/e1_queries.yaml --batch 2 --eval-val
    python scripts/benchmark_speed.py --weights checkpoints/rtdetr-e1_best.pt --run-id rtdetr-e1
    python scripts/predict.py --weights checkpoints/rtdetr-e1_best.pt --split test --batch 2 --out runs/rtdetr-e1/preds_test.parquet
    python scripts/make_submission.py --preds runs/rtdetr-e1/preds_test.parquet --out submissions/rtdetr-e1.csv
else
    echo "  -> Skipping rtdetr-e1 (already completed)."
fi

echo "============================================================"
echo "[6/6] F4 Exp B: RT-DETR-L Matching + Denoising (rtdetr-e2)..."
echo "============================================================"
if [ ! -f "experiments/runs/rtdetr-e2/speed.json" ]; then
    python scripts/train_ultra.py --config configs/rtdetr/e2_matching_dn.yaml --batch 2 --eval-val
    python scripts/benchmark_speed.py --weights checkpoints/rtdetr-e2_best.pt --run-id rtdetr-e2
else
    echo "  -> Skipping rtdetr-e2 (already completed)."
fi

echo "============================================================"
echo "Aggregating all experiment results into experiments/results.csv..."
echo "============================================================"
python scripts/aggregate_results.py
echo "ALL OVERNIGHT RUNS COMPLETED SUCCESSFULLY!"
