# Run `rtdetr-e3-ft`
- **Hypothesis:** Fine-tuning rtdetr-e0_best.pt (AP=0.2654 @ 3 ep) on 2.5x denser video frames (k=2) with cosine annealing (lr0=1.5e-4) and copy-paste augmentation unlocks full query convergence.
- **Change vs parent (`rtdetr-e0`):** Parent: rtdetr-e0
- **Outcome:** Completed: val AP=0.2596, AP50=0.5077, APs=0.1264, AR500=0.4200, epochs=2, gpu_h=2.3227, fps=9.88.
