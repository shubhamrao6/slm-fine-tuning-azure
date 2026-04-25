# Task 4: LoRA Fine-Tuning — UHCS Microstructure

Fine-tune Qwen2.5-VL-3B on 30 training images (6 per class) to classify UHCS microconstituents into 5 classes. pearlite+widmanstatten dropped (only 5 total images in dataset).

## Baseline (Task 3)

| Method | Accuracy | Notes |
|--------|----------|-------|
| Qwen2.5-VL-3B (ZS) | 60.8% | Biased toward spheroidite majority class |
| GPT-4.1 (ZS, t=0.7) | 46.7% | |
| GPT-4.1 (FS, t=0.7) | 71.7% | SEAL teacher |
| GPT-5 (FS, t=1) | 80.0% | Best frontier |

## Results (117 test images, 5 classes)

| Method | Accuracy | Training Images | Training Examples | Epochs | LR |
|--------|----------|----------------|-------------------|--------|-----|
| Qwen base (ZS) | 60.8% | 0 | 0 | — | — |
| Direct LoRA | 67.5% | 30 | 30 | 40 | 2e-5 |
| **SEAL LoRA** | **68.4%** | 30 | 120 | 40 | 2e-5 |
| GPT-4.1 (FS) | 71.7% | — | — | — | — |

### Per-Class Accuracy

| Class | N | Base ZS | Direct | SEAL | GPT-4.1 FS |
|-------|---|---------|--------|------|------------|
| spheroidite | 74 | 73% | 74% | 72% | 62% |
| network | 20 | 95% | 75% | 80% | 80% |
| spheroidite+widmanstatten | 15 | 0% | 13% | 27% | 93% |
| pearlite+spheroidite | 5 | 0% | 100% | 80% | 100% |
| pearlite | 3 | 0% | 67% | 100% | 67% |

### Key Findings

- SEAL LoRA (68.4%) beats Direct (67.5%) and approaches GPT-4.1 FS (71.7%)
- Both beat GPT-4.1 zero-shot (46.7%) with just 30 training images
- Compound classes improved from 0% base to 27-100% after fine-tuning
- spheroidite+widmanstatten remains hardest (27%) — requires detecting two co-existing features
- pearlite+widmanstatten dropped due to insufficient data (5 total, 2 in train)

## Files

| File | Description |
|------|-------------|
| `train_direct.ipynb` | Approach A: direct LoRA training + eval |
| `train_augmented.ipynb` | Approach B: SEAL CoT generation + training + eval |
| `config.py` | Shared config, prompts (identical to task3 benchmarking definitions), LoRA params |
