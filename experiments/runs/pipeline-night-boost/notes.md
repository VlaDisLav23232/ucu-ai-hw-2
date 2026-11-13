# Run `pipeline-night-boost`
- **Hypothesis:** Combining fine-tuned RT-DETR-L (k=2), YOLO11m@1600, and YOLO11s-P2 (stride-4 tiny-object head) with multi-scale/flip TTA, SAHI native-resolution tiling, class-aware WBF, and velocity-projected tracklet rescoring pushes AP significantly beyond 0.30.
- **Change vs parent (`temporal-rescore-boost`):** Parent: temporal-rescore-boost
- **Outcome:** Completed: val AP=0.3129, AP50=0.5891, APs=0.1751, AR500=0.5335.
