#!/usr/bin/env bash
# Executes Milestone M4 (Multi-Model WBF Ensemble + Temporal Tracklet Rescoring & Gap Propagation + Final Kaggle Submissions)
# and Milestone M5 (TIDE Error Breakdown, Summary Figures, and Visualisation Bonus: EigenCAM, RT-DETR Attention, UMAP).
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "============================================================"
echo "[1/3] Running M4 Pipeline (WBF Ensemble + Temporal Bonus + Test Submissions)..."
echo "============================================================"
python scripts/run_pipeline.py --config configs/pipeline/final.yaml --predict-test

echo "============================================================"
echo "[2/3] Running M5 Diagnostics, TIDE Error Analysis & Visualisation Bonus..."
echo "============================================================"
python scripts/run_diagnostics.py

echo "============================================================"
echo "[3/3] Regenerating Master Results Table (experiments/results.csv)..."
echo "============================================================"
python scripts/aggregate_results.py

echo "============================================================"
echo "ALL M4 & M5 PIPELINE STEPS AND SUBMISSIONS COMPLETED!"
echo "New Kaggle submissions ready in submissions/:"
ls -lh submissions/*.csv
echo "============================================================"

