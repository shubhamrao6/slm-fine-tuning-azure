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

## Classes

| Class | Train | Test (full) | Test (sampled) | Description |
|-------|-------|-------------|----------------|-------------|
| lack_of_penetration | 4,962 | 765 | 60 | Dark line/band along weld centerline — incomplete root fusion |
| porosity | 4,108 | 632 | 60 | Scattered dark circular spots — trapped gas pores |
| cracks | 2,893 | 446 | 60 | Dark sharp jagged lines — fractures in weld |
| no_defect | 3,900 | 600 | 60 | Uniform gray — sound weld |

## Files

| File | Description |
|------|-------------|
| `benchmark_weld.ipynb` | Qwen2.5-VL-3B benchmark (zero-shot + few-shot) |
| `benchmark_frontier.ipynb` | GPT-4.1 + GPT-5 benchmark |
| `config.py` | Shared config, prompts, class definitions |
| `riawelc_reference_grid.png` | 4×1 reference grid for few-shot |
