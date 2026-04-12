# Task 3: Granulometry Benchmarking (Base Model)

Benchmark Qwen2.5-VL-3B (base, un-fine-tuned) on the granulometry test set.
This establishes the baseline that Task 4 (fine-tuning) should improve upon.

## Status: Done

## Dataset
- Source: `datasets/granulometry/`
- Test: 108 images (12 per class, 9 classes)
- Train: 791 images (for Task 4)
- Ground truth: max particle size (8/16/32mm) and grading (coarse/medium/fine)
- Original GSD: 8 px/mm (2200x3000 images)

## Classification Task
9 classes across two independent axes:
- Max particle size: 8mm, 16mm, 32mm
- Grading (DIN 1045 standard): coarse (A), medium (B), fine (C)

Grading describes the particle size distribution, not absolute sizes:
- Coarse (A): most particles are similar size, close to max. Uniform texture.
- Medium (B): moderate mix of large and small.
- Fine (C): wide range of sizes, small particles fill gaps. Dense, packed texture.

## Two Benchmark Modes
- Zero-shot (1500px, GSD=4.0 px/mm): no reference image
- Few-shot (ref@800 + test@1400, GSD=3.7 px/mm): reference classification chart included

## Results

| Metric | Zero-Shot | Few-Shot |
|--------|-----------|----------|
| JSON validity | 100% | 100% |
| Size accuracy | 36.1% | 36.1% |
| Grading accuracy | 34.3% | 24.1% |
| Both correct | 12.0% | 8.3% |
| Avg inference time | 8.9s | 9.4s |

Random chance = 33% (3 values per axis). The base model performs at or near random.

### Key Observations
- Zero-shot heavily biases toward 16mm/medium (~73% of predictions)
- Few-shot heavily biases toward 32mm/fine (~79% of predictions)
- 100% JSON validity — structured output format works, content is wrong
- In natural language diagnostics, the model reasons correctly about coarse vs fine but cannot reliably compress reasoning into correct JSON classification
- The model cannot accurately estimate pixel sizes in images despite being given GSD

### Conclusion
The base Qwen2.5-VL-3B cannot do this task without fine-tuning. LoRA fine-tuning in Task 4 should teach the model the correct visual-to-classification mapping.

## Run
```bash
# Zero-shot
python benchmark_granulometry.py --mode zero-shot

# Few-shot
python benchmark_granulometry.py --mode few-shot

# Quick test
python benchmark_granulometry.py --mode zero-shot --limit 10
```

## Files
| File | Description |
|------|-------------|
| `benchmark_granulometry.py` | CLI benchmark script |
| `benchmark_granulometry.ipynb` | Notebook with diagnostics + visualization |
| `examples_classification_data.png` | Reference chart for few-shot mode |
| `benchmark_results_zero-shot.json` | Zero-shot results (108 images) |
| `benchmark_results_few-shot.json` | Few-shot results (108 images) |
