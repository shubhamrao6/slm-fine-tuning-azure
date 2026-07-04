# Ablation 5: Multiple Seeds (Statistical Significance)

## Goal

Run granulometry and weld defect experiments with 5 different random seeds to report mean ± std accuracy and establish statistical significance of the CoT-augmented improvement over Direct LoRA.

## Setup

- **Tasks**: Granulometry (9 classes, 18 images) + Weld Defects (4 classes, 24 images)
- **Seeds**: [42, 123, 456, 789, 1024]
- **What varies per seed**: Training image selection from the train split
- **What stays fixed**: Hyperparameters, test set, evaluation protocol, cached CoT descriptions (regenerated per seed since images change)
- **Runs**: 5 seeds × 2 tasks × 2 approaches (Direct + SEAL) = 20 training runs

## Expected Output

```
Granulometry (both correct):
  Direct LoRA: 71.3 ± X.X%
  CoT-Aug:     79.6 ± X.X%
  
Weld Defects (overall):
  Direct LoRA: 73.3 ± X.X%
  CoT-Aug:     75.8 ± X.X%
```

## Estimated Cost

- GPU time: ~16h on 1× L4 24GB (~$16)
- GPT-4.1 API: ~$4 (new CoT generations for 4 new seeds × 2 tasks)
- Total: ~$20

## Hardware

- GCP Vertex AI Workbench: `slm-workbench-l4`
- GPU: 1× NVIDIA L4 24GB
- Region: asia-southeast1-b

## Files

| File | Description |
|------|-------------|
| `ablation_multi_seed.ipynb` | Main notebook — runs all 20 experiments |
| `results/` | Output JSON files per seed |
| `README.md` | This file |
