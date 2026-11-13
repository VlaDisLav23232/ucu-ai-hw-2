# Run `pipeline-final`
- **Hypothesis:** Cross-family Weighted Box Fusion (YOLO11s @1600/1280 + RT-DETR-L nq=300/600) followed by temporal tracklet rescoring and linear gap propagation yields cumulative holdout AP gains over every single model.
- **Change vs parent (`temporal-rescore`):** Parent: temporal-rescore
- **Outcome:** Completed: val AP=0.2996, AP50=0.5638, APs=0.1574, AR500=0.5287.
