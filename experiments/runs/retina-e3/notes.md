# Run `retina-e3`
- **Hypothesis:** Exp B alone: relaxed matcher IoU thresholds 0.4/0.3, focal gamma=1.5, alpha=0.25 and GIoU box regression with the default COCO anchors. Added after retina-e1 showed that the k-means anchors hurt, to separate the loss/matching effect from the anchor change.
- **Change vs parent (`retina-e0`):** Parent: retina-e0
- **Outcome:** Completed: val AP=0.1424, AP50=0.2664, APs=0.0473, AR500=0.2543, epochs=7, gpu_h=1.4205.
