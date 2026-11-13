# Run `rtdetr-e1`
- **Hypothesis:** Increasing decoder object queries from nq=300 to nq=600 relieves top-k query bottleneck in dense aerial scenes (up to 180 objects/frame) and improves AR@500 and AP.
- **Change vs parent (`rtdetr-e0`):** Parent: rtdetr-e0
- **Outcome:** Completed: val AP=0.2606, AP50=0.5012, APs=0.1307, AR500=0.5006, epochs=2, gpu_h=1.5306, fps=8.82.
