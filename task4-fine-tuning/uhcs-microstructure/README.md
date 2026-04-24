# Task 4: LoRA Fine-Tuning — UHCS Microstructure

Fine-tune Qwen2.5-VL-3B on ~30 training images (5 per class, fewer for rare classes) to classify UHCS microconstituents into 6 classes. Same methodology as granulometry and steel surface.

## Baseline (Task 3)

| Method | Accuracy | Notes |
|--------|----------|-------|
| Qwen2.5-VL-3B (ZS) | 60.8% | Biased toward spheroidite majority class |
| GPT-4.1 (ZS, t=0.7) | 46.7% | |
| GPT-4.1 (FS, t=0.7) | 71.7% | SEAL teacher |
| GPT-5 (FS, t=1) | 80.0% | Best frontier |

## Dataset Notes

- Heavily imbalanced: spheroidite (372) vs pearlite+widmanstatten (5)
- Training uses train_manifest.json (created by benchmarking notebook, ~478 images)
- Rare classes (pearlite: 12 train, pearlite+widmanstatten: 2 train) will have fewer than 5 training images
- Magnification varies (49x–19641x) and is included in the prompt

## Approach A: Direct LoRA

- ~30 images × 1 example = ~30 training pairs
- Image + prompt (with magnification) → `{"primary_microconstituent": "..."}`
- 40 epochs, lr=2e-5

## Approach B: SEAL-Augmented LoRA

- ~30 images × 4 examples = ~120 training pairs
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
