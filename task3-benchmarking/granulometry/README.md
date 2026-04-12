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

### Qwen2.5-VL-3B (base model)

| Metric | Zero-Shot | Few-Shot |
|--------|-----------|----------|
| JSON validity | 100% | 100% |
| Size accuracy | 36.1% | 36.1% |
| Grading accuracy | 34.3% | 24.1% |
| Both correct | 12.0% | 8.3% |
| Avg inference time | 8.9s | 9.4s |

### Frontier Models

| Model | Mode | Size | Grading | Both | Time |
|-------|------|------|---------|------|------|
| GPT-5 | Zero-shot | 59.3% | 33.3% | 18.5% | 11.9s |
| GPT-5 | Few-shot | 66.7% | 33.3% | 22.2% | 15.8s |
| GPT-4.1 | Zero-shot (t=0.7) | 53.7% | 49.1% | 31.5% | 4.3s |
| GPT-4.1 | Few-shot (t=0.7) | 62.0% | 59.3% | 29.6% | 4.7s |

Random chance = 33% (3 values per axis).

### Key Observations
- Qwen2.5-VL-3B performs at random chance on both axes
- GPT-5 gets size right (~60-67%) but predicts "fine" for ALL 108 images (no temperature control)
- GPT-4.1 with temperature=0.7 and improved DIN 1045 prompts is the best overall
- GPT-4.1 few-shot: 62% size, 59.3% grading — the only model to break past random on grading
- All models achieve 100% JSON validity
- Grading (coarse/medium/fine) is the hard problem — domain-specific DIN 1045 convention
- The "gaps between stones" visual test in prompts significantly improved grading accuracy
- All models struggle with 32mm size detection (consistently underestimate largest particles)

### Conclusion
The base Qwen2.5-VL-3B cannot do this task without fine-tuning. GPT-4.1 few-shot is the strongest baseline and is selected as the SEAL teacher for Task 4 data augmentation.

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
| `benchmark_granulometry.py` | CLI benchmark script (Qwen2.5-VL-3B) |
| `benchmark_granulometry.ipynb` | Notebook: Qwen benchmark with diagnostics + visualization |
| `benchmark_frontier.ipynb` | Notebook: GPT-5 and GPT-4.1 benchmarks + comparison |
| `examples_classification_data.png` | Reference chart for few-shot mode |
| `benchmark_results_zero-shot.json` | Qwen zero-shot results (108 images) |
| `benchmark_results_few-shot.json` | Qwen few-shot results (108 images) |
| `benchmark_results_frontier-zero-shot.json` | GPT-5 zero-shot results |
| `benchmark_results_frontier-few-shot.json` | GPT-5 few-shot results |
| `benchmark_results_gpt41-zero-shot.json` | GPT-4.1 zero-shot results |
| `benchmark_results_gpt41-few-shot.json` | GPT-4.1 few-shot results |
