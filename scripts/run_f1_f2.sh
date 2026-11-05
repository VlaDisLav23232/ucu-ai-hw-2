#!/usr/bin/env bash
# Executes Family 1 (Faster R-CNN R50-FPN v2: e0, e1, e2) and Family 2 (RetinaNet R50-FPN v2: e0, e1, e2)
# Automatically skips any run that has already completed.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

for exp_cfg in "frcnn-e0:configs/faster_rcnn/e0_base.yaml" \
               "frcnn-e1:configs/faster_rcnn/e1_rpn_proposals.yaml" \
               "frcnn-e2:configs/faster_rcnn/e2_giou.yaml" \
               "retina-e0:configs/retinanet/e0_base.yaml" \
               "retina-e1:configs/retinanet/e1_kmeans_anchors.yaml" \
               "retina-e2:configs/retinanet/e2_focal_assign.yaml"; do
    rid="${exp_cfg%%:*}"
    cfg="${exp_cfg#*:}"
    echo "============================================================"
    echo "Running ${rid} (${cfg})..."
    echo "============================================================"
    if [ ! -f "experiments/runs/${rid}/speed.json" ]; then
        python scripts/train_tv.py --config "${cfg}" --batch 2 --eval-val
        python scripts/benchmark_speed.py --weights "checkpoints/${rid}_best.pth" --run-id "${rid}" --config "${cfg}"
    else
        echo "  -> Skipping ${rid} (already completed)."
    fi
done

python scripts/aggregate_results.py
echo "ALL F1 (Faster R-CNN) & F2 (RetinaNet) RUNS COMPLETED!"

