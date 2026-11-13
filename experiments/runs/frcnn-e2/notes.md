# Run `frcnn-e2`
- **Hypothesis:** Replacing Smooth-L1 RoI regression loss with scale-invariant GIoU loss on decoded boxes directly optimizes IoU across multi-scale aerial targets and improves AP75 and overall AP.
- **Change vs parent (`frcnn-e1`):** Parent: frcnn-e1
- **Outcome:** Completed: val AP=0.2237, AP50=0.4654, APs=0.1050, AR500=0.3932, epochs=5, gpu_h=1.4285.
