# Run `retina-e2`
- **Hypothesis:** Relaxing anchor positive/negative IoU thresholds from 0.5/0.4 to 0.4/0.3 with focal gamma=1.5, alpha=0.25 and GIoU box regression prevents tiny aerial objects from being starved of positive anchors.
- **Change vs parent (`retina-e1`):** Parent: retina-e1
- **Outcome:** Completed: val AP=0.0694, AP50=0.2054, APs=0.0448, AR500=0.2411, epochs=7, gpu_h=1.4208.
