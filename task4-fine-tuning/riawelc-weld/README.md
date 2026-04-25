# Task 4: LoRA Fine-Tuning — RIAWELC Weld Defects

Fine-tune Qwen2.5-VL-3B on 24 training images (6 per class) to classify weld defects from X-ray radiographs into 4 classes. Same methodology as granulometry, steel surface, and UHCS.

## Baseline (Task 3)

| Method | Accuracy | Notes |
|--------|----------|-------|
| Qwen2.5-VL-3B (ZS) | 30.8% | Near random (25%) |
| Qwen2.5-VL-3B (FS) | 51.2% | |
| GPT-4.1 (ZS, t=0.7) | 57.5% | |
| GPT-4.1 (FS, t=0.7) | 65.0% | SEAL teacher |

## Results (240 test images)

| Method | Accuracy | Training Images | Training Examples | Epochs | LR |
|--------|----------|----------------|-------------------|--------|-----|
| Qwen base (ZS) | 30.8% | 0 | 0 | — | — |
| Direct LoRA | 73.3% | 24 | 24 | 40 | 2e-5 |
| **SEAL LoRA** | **75.8%** | 24 | 96 | 40 | 2e-5 |
| GPT-4.1 (FS) | 65.0% | — | — | — | — |

### Per-Class Accuracy

| Class | Base ZS | Direct | SEAL | GPT-4.1 FS |
|-------|---------|--------|------|------------|
| lack_of_penetration | 15% | 65% | 72% | 38% |
| porosity | 13% | 83% | 83% | 92% |
| cracks | 2% | 58% | 50% | 30% |
| no_defect | 93% | 87% | 98% | 100% |

### Key Findings

- SEAL LoRA (75.8%) beats Direct (73.3%) and surpasses GPT-4.1 FS (65.0%) by 10.8pp
- Both methods beat the frontier model — a 3B model trained on 24 images outperforms GPT-4.1 few-shot
- Cracks: from 0% (all models zero-shot) to 58% (Direct) / 50% (SEAL) with just 6 training images
- The 3B model detects cracks better than GPT-4.1 FS (30%) — strongest result across all use cases
- Main confusion: cracks ↔ lack_of_penetration (both are dark lines in radiographs)
- 2.5× improvement over base (30.8% → 75.8%)

## Files

| File | Description |
|------|-------------|
| `train_direct.ipynb` | Approach A: direct LoRA training + eval |
| `train_augmented.ipynb` | Approach B: SEAL CoT generation + training + eval |
| `config.py` | Shared config, prompts (identical to task3 benchmarking definitions), LoRA params |
