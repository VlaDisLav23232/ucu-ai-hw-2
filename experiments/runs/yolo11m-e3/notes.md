# Run `yolo11m-e3`
- **Hypothesis:** Scaling from YOLO11s (9.4M) to YOLO11m (20.1M) at imgsz=1600 on k=2 frames with copy_paste=0.25 and mixup=0.15 increases multi-scale feature capacity and rare-class recall.
- **Change vs parent (`yolo-e1`):** Parent: yolo-e1
- **Outcome:** Completed: val AP=0.2118, AP50=0.4314, APs=0.1393, AR500=0.3704, epochs=1, gpu_h=2.0191, fps=15.63.
