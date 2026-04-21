# Task 4: LoRA Fine-Tuning — Steel Surface Defects (NEU-CLS)

Fine-tune Qwen2.5-VL-3B on 18 training images (3 per class) to classify steel surface defects into 6 classes. Same methodology as granulometry: Direct LoRA vs SEAL-augmented CoT distillation.

## Baseline (Task 3)

| Method | Accuracy | Notes |
|--------|----------|-------|
| Qwen2.5-VL-3B (ZS) | 21.7% | Near random (16.7%) |
| Qwen2.5-VL-3B (FS) | 22.8% | Reference grid doesn't help |
| GPT-4.1 (FS, t=0.7) | 91.1% | SEAL teacher |
| GPT-5 (FS, t=1) | 86.4% | Slower, slightly worse |

## Approach A: Direct LoRA

- 18 images × 1 example = 18 training pairs
- Image + prompt → `{"defect_class": "..."}`
- 40 epochs, lr=2e-5

## Approach B: SEAL-Augmented LoRA

- 18 images × 4 examples = 72 training pairs
- GPT-4.1 generates 3 CoT descriptions per image (answer-conditioned, t=0.7)
- Plus 1 direct JSON per image
- 15 epochs, lr=1e-5

## Files

| File | Description |
|------|-------------|
| `train_direct.ipynb` | Approach A: direct LoRA training + eval |
| `train_augmented.ipynb` | Approach B: SEAL CoT generation + training + eval |
| `config.py` | Shared config, prompts, LoRA params |
| `training_data_direct.jsonl` | Generated: 18 direct examples |
| `training_data_augmented.jsonl` | Generated: 72 augmented examples |
| `lora_direct/` | Saved: direct LoRA adapter |
| `lora_augmented/` | Saved: SEAL LoRA adapter |
| `results_direct.json` | Eval results: direct approach |
| `results_augmented.json` | Eval results: SEAL approach |
