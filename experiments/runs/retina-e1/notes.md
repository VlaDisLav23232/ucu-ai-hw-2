# Run `retina-e1`
- **Hypothesis:** Replacing COCO default anchors (mean best IoU 0.648) with 1280-scale k-means anchors (10..109 px, mean best IoU 0.819, 9 anchors/location) increases positive anchor matches on small VisDrone objects.
- **Change vs parent (`retina-e0`):** Parent: retina-e0
- **Outcome:** Completed: val AP=0.0686, AP50=0.1993, APs=0.0383, AR500=0.2548, epochs=7, gpu_h=1.4207.
