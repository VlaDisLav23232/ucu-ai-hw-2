# Run `yolo-e1`
- **Hypothesis:** Increasing input resolution from 1280 to 1600 (search range {1280, 1600}) increases small-object feature map density and boosts AP_small and overall AP.
- **Change vs parent (`yolo-e0`):** Parent: yolo-e0
- **Outcome:** Completed: val AP=0.2184, AP50=0.4346, APs=0.1312, AR500=0.3716, epochs=11, gpu_h=1.5039, fps=32.52.
