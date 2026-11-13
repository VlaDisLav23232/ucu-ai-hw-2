# Run `rtdetr-e2`
- **Hypothesis:** Re-weighting Hungarian bipartite matching toward spatial localization (bbox=7.5, giou=3.0, class=1.5 vs 5/2/2) and doubling denoising groups (100 -> 200) accelerates convergence on small aerial objects.
- **Change vs parent (`rtdetr-e1`):** Parent: rtdetr-e1
- **Outcome:** Completed: val AP=0.2522, AP50=0.4851, APs=0.1406, AR500=0.5109, epochs=2, gpu_h=1.536, fps=8.74.
