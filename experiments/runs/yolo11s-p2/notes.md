# Run `yolo11s-p2`
- **Hypothesis:** Adding a stride-4 P2/4 detection head (Detect P2,P3,P4,P5) initialized from yolo-e1_best.pt provides 4x denser spatial cells for <32x32 px objects.
- **Change vs parent (`yolo-e1`):** Parent: yolo-e1
- **Outcome:** Completed: val AP=0.1962, AP50=0.3940, APs=0.1012, AR500=0.3410, epochs=2, gpu_h=0.9903, fps=38.48.
