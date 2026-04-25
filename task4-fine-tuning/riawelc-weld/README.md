# Task 4: LoRA Fine-Tuning — RIAWELC Weld Defects

Fine-tune Qwen2.5-VL-3B on 24 training images (6 per class) to classify weld defects from X-ray radiographs into 4 classes. Same methodology as granulometry, steel surface, and UHCS.

## Baseline (Task 3)

| Method | Accuracy | Notes |
|--------|----------|-------|
| Qwen2.5-VL-3B (ZS) | 30.8% | Near random (25%) |
| Qwen2.5-VL-3B (FS) | 51.2% | |
| GPT-4.1 (ZS, t=0.7) | 57.5% | |
| GPT-4.1 (FS, t=0.7) | 65.0% | SEAL teacher (only model detecting cracks) |
| GPT-5 (FS, t=1) | 62.5% | 0% on cracks even with few-shot |

## Approach A: Direct LoRA

- 24 images (6 per class) × 1 example = 24 training pairs
- Image + prompt → `{"defect_class": "..."}`
- 40 epochs, lr=2e-5

## Approach B: SEAL-Augmented LoRA

- 24 images × 4 examples = 96 training pairs
- GPT-4.1 generates 3 CoT descriptions per image (answer-conditioned, t=0.7)
- Plus 1 direct JSON per image
- CoT examples use prompt WITHOUT "Respond with JSON" instruction (last 2 lines stripped)
- Code appends correct JSON to GPT-4.1's description
- 40 epochs, lr=2e-5

## Files

| File | Description |
|------|-------------|
| `train_direct.ipynb` | Approach A: direct LoRA training + eval |
| `train_augmented.ipynb` | Approach B: SEAL CoT generation + training + eval |
| `config.py` | Shared config, prompts (identical to task3 benchmarking definitions), LoRA params |
