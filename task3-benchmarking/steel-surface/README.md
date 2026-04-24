# Task 3: Steel Surface Defect Benchmarking

Benchmark Qwen2.5-VL-3B and frontier models (GPT-4.1, GPT-5) on the NEU Steel Surface Defect dataset.
Same methodology as granulometry: zero-shot, few-shot, and frontier model comparison.

## Dataset: NEU-CLS

| Property | Value |
|----------|-------|
| Source | Northeastern University, China |
| Classes | 6: crazing (Cr), inclusion (In), patches (Pa), pitted surface (PS), rolled-in scale (RS), scratches (Sc) |
| Images | 1,800 total (300 per class) |
| Size | 200×200 grayscale |
| Split | 240 train / 60 test per class (1,440 / 360) |
| Download | [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |

## Benchmark Results (360 test images)

| Method | Accuracy | JSON Valid | Time/img |
|--------|----------|-----------|----------|
| Qwen2.5-VL-3B (ZS) | 21.7% | 100% | 1.9s |
| Qwen2.5-VL-3B (FS) | 22.8% | 100% | 2.9s |
| GPT-4.1 (ZS, t=0.7) | 46.9% | 98% | 1.6s |
| GPT-4.1 (FS, t=0.7) | 91.1% | 99% | 2.5s |
| GPT-5 (ZS, t=1) | 45.8% | 100% | 7.6s |
| GPT-5 (FS, t=1) | 86.4% | 100% | 10.7s |
| Random chance | 16.7% | — | — |

### Per-Class Accuracy (best model: GPT-4.1 FS = 91.1%)

| Class | Qwen ZS | Qwen FS | GPT-4.1 ZS | GPT-4.1 FS | GPT-5 ZS | GPT-5 FS |
|-------|---------|---------|------------|------------|----------|----------|
| crazing | 18% | 13% | 95% | 98% | 62% | 97% |
| inclusion | 35% | 23% | 2% | 77% | 0% | 67% |
| patches | 22% | 7% | 80% | 98% | 70% | 98% |
| pitted_surface | 3% | 2% | 45% | 95% | 75% | 87% |
| rolled-in_scale | 2% | 0% | 0% | 93% | 3% | 85% |
| scratches | 50% | 92% | 60% | 85% | 65% | 85% |

### Key Findings

- Qwen2.5-VL-3B base model is near random chance (~22%) — strong candidate for CoT distillation improvement
- Few-shot reference image is critical: GPT-4.1 jumps from 46.9% → 91.1% with the reference grid
- GPT-4.1 FS (91.1%) outperforms GPT-5 FS (86.4%) — GPT-4.1 is the better SEAL teacher for this task
- Inclusion and rolled-in_scale are the hardest classes (visually similar elongated dark marks)
- Qwen FS has a strong scratches bias (92%) — predicts scratches for almost everything

## Fine-Tuning Results (Task 4)

| Method | Accuracy | Training Data |
|--------|----------|--------------|
| Direct LoRA | 63.1% | 30 images (5/class) |
| SEAL LoRA | 66.7% | 30 images → 120 augmented examples |

SEAL LoRA achieves 3× improvement over base (22% → 67%) and beats GPT-4.1 zero-shot (47%).

## Defect Definitions

| Class | Code | Visual Description |
|-------|------|-------------------|
| Crazing | Cr | Network of fine surface cracks, web-like pattern |
| Inclusion | In | Dark spots or streaks embedded in the surface, foreign material |
| Patches | Pa | Irregular lighter/darker areas on the surface, uneven texture |
| Pitted Surface | PS | Small holes or depressions scattered across the surface |
| Rolled-in Scale | RS | Oxide scale pressed into the surface during rolling, elongated marks |
| Scratches | Sc | Linear marks/grooves on the surface, directional damage |

## Reference Image

`Sample-images-in-the-NEU-CLS-dataset.png` — shows examples of all 6 classes for few-shot prompting.

## Files

| File | Description |
|------|-------------|
| `benchmark_steel.ipynb` | Main benchmark notebook (Qwen zero-shot + few-shot) |
| `benchmark_frontier.ipynb` | Frontier model benchmark (GPT-4.1 + GPT-5) |
| `config.py` | Shared config and prompts |
| `Sample-images-in-the-NEU-CLS-dataset.png` | Reference image for few-shot |
