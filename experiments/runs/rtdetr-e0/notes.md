# Run `rtdetr-e0`
- **Hypothesis:** COCO-pretrained RT-DETR-L baseline (300 queries, imgsz=1280, AdamW, 1.5 GPU-h budget) removes heuristic NMS via bipartite Hungarian matching.
- **Change vs parent (`none`):** Parent: none
- **Outcome:** Completed: val AP=0.2654, AP50=0.5129, APs=0.1370, AR500=0.4329, epochs=3, gpu_h=1.5184, fps=9.04.
