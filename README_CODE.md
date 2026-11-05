# VisDrone-VID Detection Codebase (`vdet`) — Reproducibility & Execution Guide (`U5.6`)

Competition: **`multiple-object-video-detection-hw-2-ucu-ai-2026`** (8-class per-frame aerial video object detection).

---

## 1. Environment Setup (`U0.1`)

### Local WSL 2 (Ubuntu 24.04, NVIDIA RTX 4050 6 GB)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
pytest -q
```

### Kaggle Notebooks / Google Colab (T4 16 GB)
See [00_env_bootstrap.ipynb](file:///home/marko/ucu-ai-hw-2/notebooks/00_env_bootstrap.ipynb). `vdet.paths.P` auto-detects `local_wsl`, `kaggle`, or `colab`.

---

## 2. Data Preparation & Offline Cache Building (`U0.2`–`U0.4`)

```bash
# 1. Build Parquet manifests (manifest.parquet, boxes.parquet, ignore.parquet)
python scripts/prepare_manifest.py

# 2. Verify coordinate conversions, deduplicated crowd ignore regions, and generate U0.3 figures
python scripts/verify_data.py

# 3. Generate U0.8 EDA figures and frame-step redundancy table (lag k=1..20)
python scripts/run_eda.py

# 4. Build k=5 offline caches (winning policy from U1.2 = 'drop')
python scripts/build_cache.py --config configs/data/cache_k5_drop.yaml --workers 4
```

---

## 3. Model Families (`M1`–`M3`: `F1`, `F2`, `F3`, `F4`)

All four model families train under **Fixed Protocol P0** ([protocol.yaml](file:///home/marko/ucu-ai-hw-2/configs/protocol.yaml): `cache=k5_drop_L1920`, `imgsz=1280`, `1.5 GPU-h` budget, `seed=42`, evaluated on all `2,846` original `val` frames with `maxDets=[1, 10, 500]` and `iscrowd=1` ignore regions).

### Family 3: One-Stage Anchor-Free (`YOLO11s`)
```bash
python scripts/train_ultra.py --config configs/yolo/e0_base.yaml --batch 4 --eval-val
python scripts/train_ultra.py --config configs/yolo/e1_imgsz1600.yaml --batch 2 --eval-val
python scripts/train_ultra.py --config configs/yolo/e2_nwd.yaml --batch 2 --eval-val
```

### Family 4: Query-Based Transformer (`RT-DETR-L`)
```bash
python scripts/train_ultra.py --config configs/rtdetr/e0_base.yaml --batch 2 --eval-val
python scripts/train_ultra.py --config configs/rtdetr/e1_queries.yaml --batch 2 --eval-val
python scripts/train_ultra.py --config configs/rtdetr/e2_matching_dn.yaml --batch 2 --eval-val
```

### Family 1 (`Faster R-CNN R50-FPN v2`) & Family 2 (`RetinaNet R50-FPN v2`)
```bash
# Run all 6 Torchvision experiments sequentially:
bash scripts/run_f1_f2.sh
```

---

## 4. Milestone M4 ("Your Pipeline") & M5 Diagnostics (`U4.1`–`U4.8`, `U5.1`–`U5.5`)

```bash
# Run Cross-Family WBF Ensembling + Temporal Tracklet Rescoring & Gap Propagation + Diagnostics + Kaggle Submissions:
bash scripts/run_m4_m5_and_submissions.sh
```

Individual CLI tools:
- **Multi-Model WBF / Soft-NMS / NMS Fusion (`U4.3`):**
  ```bash
  python scripts/fuse.py \
      --preds runs/yolo-e1/preds_val.parquet runs/rtdetr-e0/preds_val.parquet runs/rtdetr-e1/preds_val.parquet \
      --weights 1.5 2.5 2.2 --method wbf --iou-thr 0.55 \
      --out runs/ensemble-wbf/preds_val.parquet --run-id ensemble-wbf --eval-val
  ```
- **Temporal Tracklet Rescoring & Linear Gap Propagation (`U4.4`–`U4.6`):**
  ```bash
  python scripts/temporal.py \
      --preds runs/ensemble-wbf/preds_val.parquet \
      --out runs/pipeline-final/preds_val_final.parquet \
      --method rescore --iou-link 0.45 --alpha 0.70 --run-id pipeline-final --eval-val
  ```
- **SAHI Tiling (`U4.1`) & TTA (`U4.2`) via `predict.py`:**
  ```bash
  python scripts/predict.py --weights checkpoints/yolo-e1_best.pt --split val --tiling --slice-size 800 --out runs/yolo-e1/preds_val_tiled.parquet
  python scripts/predict.py --weights checkpoints/yolo-e1_best.pt --split val --tta --out runs/yolo-e1/preds_val_tta.parquet
  ```
- **TIDE Error Breakdown, Summary Figures & Visualisation Bonus (`U5.1`–`U5.5`):**
  ```bash
  python scripts/run_diagnostics.py
  ```

---

## 5. Credits & Citations
- **SAHI Tiling (`U4.1`):** Akyon et al., *"Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection"*, ICIP 2022.
- **Weighted Box Fusion (`U4.3`):** Solovyev et al., *"Weighted boxes fusion: Ensembling boxes from different object detection models"*, Image and Vision Computing, 2021 (`ensemble-boxes`).
- **Seq-NMS (`U4.5`):** Han et al., *"Seq-NMS for Video Object Detection"*, arXiv:1602.08465, 2016.
- **Normalized Wasserstein Distance (`F3-E2`):** Wang et al., *"A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"*, arXiv:2110.13389, 2021.
- **TIDE Error Breakdown (`U5.1`):** Bolya et al., *"TIDE: A General Toolbox for Identifying Object Detection Errors"*, ECCV 2020.
