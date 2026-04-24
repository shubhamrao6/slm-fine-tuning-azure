# Task 4: LoRA Fine-Tuning — Steel Surface Defects (NEU-CLS)

Fine-tune Qwen2.5-VL-3B on 30 training images (5 per class) to classify steel surface defects into 6 classes. Same methodology as granulometry: Direct LoRA vs SEAL-augmented CoT distillation.

## Baseline (Task 3)

| Method | Accuracy | Notes |
|--------|----------|-------|
| Qwen2.5-VL-3B (ZS) | 21.7% | Near random (16.7%) |
| Qwen2.5-VL-3B (FS) | 22.8% | Reference grid doesn't help |
| GPT-4.1 (ZS, t=0.7) | 46.9% | Zero-shot frontier |
| GPT-4.1 (FS, t=0.7) | 91.1% | SEAL teacher (few-shot with reference grid) |

## Results (360 test images)

| Method | Accuracy | Training Images | Training Examples | Epochs | LR |
|--------|----------|----------------|-------------------|--------|-----|
| Qwen base (ZS) | 21.7% | 0 | 0 | — | — |
| Direct LoRA | 63.1% | 30 | 30 | 40 | 2e-5 |
| **SEAL LoRA** | **66.7%** | 30 | 120 | 40 | 2e-5 |
| GPT-4.1 (FS) | 91.1% | — | — | — | — |

### Per-Class Accuracy

| Class | Base ZS | Direct LoRA | SEAL LoRA | GPT-4.1 FS |
|-------|---------|-------------|-----------|------------|
| crazing | 18% | 83% | 68% | 98% |
| inclusion | 35% | 38% | 32% | 77% |
| patches | 22% | 52% | 75% | 98% |
| pitted_surface | 3% | 83% | 85% | 95% |
| rolled-in_scale | 2% | 58% | 55% | 93% |
| scratches | 50% | 63% | 85% | 85% |

### Key Findings

- SEAL LoRA (66.7%) beats Direct LoRA (63.1%) — CoT distillation adds +3.6pp overall
- Both methods achieve 3× improvement over base model (22% → 63-67%)
- Both beat GPT-4.1 zero-shot (46.9%) — the fine-tuned 3B model outperforms the frontier model without reference images
- SEAL is more balanced: no class below 32%, strong on patches (75%) and scratches (85%)
- Direct is better on crazing (83% vs 68%) and inclusion (38% vs 32%) — memorization helps for some classes
- Inclusion remains the hardest class (32-38%) — 65% of errors are confusion with scratches
- Increasing training images from 18 → 30 improved SEAL from 55.8% → 66.7% (+11pp)

## Approach A: Direct LoRA

- 30 images (5 per class) × 1 example = 30 training pairs
- Image + prompt → `{"defect_class": "..."}`
- 40 epochs, lr=2e-5, grad_accum=4, cosine schedule with 10% warmup

## Approach B: SEAL-Augmented LoRA (Winner)

- 30 images (5 per class) × 4 examples = 120 training pairs
- GPT-4.1 generates 3 CoT descriptions per image (answer-conditioned, t=0.7)
- Plus 1 direct JSON per image
- CoT examples use prompt WITHOUT "Respond with JSON" instruction
- Code appends correct JSON to GPT-4.1's description
- 40 epochs, lr=2e-5, grad_accum=4, cosine schedule with 10% warmup

## Files

| File | Description |
|------|-------------|
| `train_direct.ipynb` | Approach A: direct LoRA training + eval |
| `train_augmented.ipynb` | Approach B: SEAL CoT generation + training + eval |
| `config.py` | Shared config, prompts, LoRA params |
| `training_data_direct.jsonl` | Generated: 30 direct examples |
| `training_data_augmented.jsonl` | Generated: 120 augmented examples |
| `lora_direct/` | Saved: direct LoRA adapter |
| `lora_augmented/` | Saved: SEAL LoRA adapter |
| `results_direct.json` | Eval results: direct approach |
| `results_augmented.json` | Eval results: SEAL approach |
