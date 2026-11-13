# Run `yolo-e2`
- **Hypothesis:** Blending CIoU with Normalized Wasserstein Distance (lambda_nwd=0.5, C=24.0 px fitted to mean 1280-scale box size) stabilizes regression gradients for tiny objects (<32^2 px).
- **Change vs parent (`yolo-e0`):** Parent: yolo-e0
- **Outcome:** Completed: val AP=0.2021, AP50=0.3959, APs=0.0872, AR500=0.3081, epochs=17, gpu_h=1.5028, fps=55.01.
