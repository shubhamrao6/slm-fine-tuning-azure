# Task 3: RIAWELC Weld Defect Benchmarking

Benchmark Qwen2.5-VL-3B and frontier models (GPT-4.1, GPT-5) on 4-class weld defect classification from X-ray radiographs.

## Dataset: RIAWELC

| Property | Value |
|----------|-------|
| Source | University of Valparaíso, Chile |
| Classes | 4: lack of penetration, porosity, cracks, no defect |
| Images | 24,407 total (227×227 grayscale PNG) |
| Split | 15,863 train / 6,101 validation / 2,443 test |
| Benchmark subset | 240 test images (60 per class, stratified sample) |
| Download | [GitHub](https://github.com/RIAWELC/RIAWELC) |

## Benchmark Results (240 test images)

| Method | Accuracy | JSON Valid | Time/img |
|--------|----------|-----------|----------|
| Qwen2.5-VL-3B (ZS) | 30.8% | 100% | 2.0s |
| Qwen2.5-VL-3B (FS) | 51.2% | 100% | 2.9s |
| GPT-4.1 (ZS, t=0.7) | 57.5% | 97% | 2.0s |
| GPT-4.1 (FS, t=0.7) | 65.0% | 100% | 3.0s |
| GPT-5 (ZS, t=1) | 62.5% | 100% | 5.6s |
| GPT-5 (FS, t=1) | 62.5% | 100% | 8.0s |
| Random chance | 25.0% | — | — |

### Per-Class Accuracy

| Class | Qwen ZS | Qwen FS | GPT-4.1 ZS | GPT-4.1 FS | GPT-5 ZS | GPT-5 FS |
|-------|---------|---------|------------|------------|----------|----------|
| lack_of_penetration | 15% | 42% | 62% | 38% | 72% | 60% |
| porosity | 13% | 77% | 72% | 92% | 78% | 90% |
| cracks | 2% | 3% | 0% | 30% | 0% | 0% |
| no_defect | 93% | 83% | 97% | 100% | 100% | 100% |

### Key Findings

- Cracks is catastrophically hard — 0% for all models zero-shot, only GPT-4.1 FS reaches 30%
- No_defect is trivially easy (83-100%) — uniform radiographs are distinctive
- Porosity responds well to few-shot (Qwen: 13% → 77%, GPT-4.1: 72% → 92%)
- GPT-4.1 FS (65.0%) is the best practical choice — GPT-5 FS ties at 62.5% but is 2.7× slower and scores 0% on cracks
- X-ray radiographs are a fundamentally different modality — all models struggle more than on surface photos
- GPT-4.1 will be used as SEAL teacher (faster, only model that can detect cracks at all)

## Classes

| Class | Train | Test (full) | Test (sampled) | Description |
|-------|-------|-------------|----------------|-------------|
| lack_of_penetration | 4,962 | 765 | 60 | Dark line/band along weld centerline — incomplete root fusion |
| porosity | 4,108 | 632 | 60 | Scattered dark circular spots — trapped gas pores |
| cracks | 2,893 | 446 | 60 | Dark sharp jagged lines — fractures in weld |
| no_defect | 3,900 | 600 | 60 | Uniform gray — sound weld |

## Fine-Tuning Results (Task 4)

| Method | Accuracy | Training Data |
|--------|----------|--------------|
| Direct LoRA | 73.3% | 24 images (6/class) |
| SEAL LoRA | 75.8% | 24 images → 96 augmented examples |

Both beat GPT-4.1 FS (65.0%) — the fine-tuned 3B model outperforms the frontier model on X-ray radiographs.

## Files

| File | Description |
|------|-------------|
| `benchmark_weld.ipynb` | Qwen2.5-VL-3B benchmark (zero-shot + few-shot) |
| `benchmark_frontier.ipynb` | GPT-4.1 + GPT-5 benchmark |
| `config.py` | Shared config, prompts, class definitions |
| `riawelc_reference_grid.png` | 4×1 reference grid for few-shot |
