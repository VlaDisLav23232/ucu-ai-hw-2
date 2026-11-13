# Run `frcnn-e1`
- **Hypothesis:** Doubling RPN pre/post-NMS proposals and expanding RoI batch size to 1024 (positive_fraction=0.33) with halved base anchor sizes (16..256 px) prevents RPN proposal starvation on crowded aerial frames.
- **Change vs parent (`frcnn-e0`):** Parent: frcnn-e0
- **Outcome:** Completed: val AP=0.2154, AP50=0.4609, APs=0.1050, AR500=0.4376, epochs=5, gpu_h=1.4282.
