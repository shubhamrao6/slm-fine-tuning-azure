# Task 3: Granulometry Benchmarking (Base Model)

Benchmark Qwen2.5-VL-3B (base, un-fine-tuned) on the granulometry test set.
This establishes the baseline that Task 4 (fine-tuning) should improve upon.

## Dataset
- Source: `datasets/granulometry/`
- Test: 108 images (12 per class, 9 classes)
- Train: 791 images (for Task 4)
- Ground truth: max particle size (8/16/32mm) and grading (coarse/medium/fine)
- Pixel-to-mm: 8 px/mm

## What the model is asked
For each test image, the model receives:
- The image (resized to 500x500)
- Prompt asking for max particle size and grading classification

Expected JSON output:
```json
{"max_particle_size_mm": 32, "grading": "coarse"}
```

## Metrics
- Max particle size accuracy: % correct (8, 16, or 32)
- Grading accuracy: % correct (coarse, medium, fine)
- JSON validity: % of responses that are parseable JSON
- Average inference time per image

## Run
Upload `benchmark_granulometry.py` and the test images to the compute instance, then:
```bash
python benchmark_granulometry.py
```
