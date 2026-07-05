# Ablation: Unconditioned CoT Baseline

## Goal

Prove that **answer-conditioning** is essential for CoT distillation quality. Show that unconditioned reasoning (where GPT-4.1 classifies freely without knowing the answer) produces worse training data that degrades model performance.

## Design

| Method | Teacher knows answer? | Training data quality | Expected accuracy |
|--------|----------------------|----------------------|-------------------|
| Direct LoRA | N/A (no teacher) | 100% correct labels | ~75% |
| **Unconditioned CoT** | No | ~30% correct reasoning | Much worse |
| CoT-Aug (conditioned) | Yes | 100% correct reasoning | ~78% |

## How it works

1. Send each training image to GPT-4.1 with the classification prompt (same as evaluation)
2. GPT-4.1 classifies AND explains freely — WITHOUT being told the correct answer
3. GPT-4.1's own prediction becomes the training label (even if WRONG)
4. Train the student model on this data

Since GPT-4.1 only gets ~29.6% correct on granulometry, ~70% of the training data will teach the model to reason toward WRONG answers.

## Tasks

- Granulometry (GPT-4.1 accuracy: 29.6% → ~70% wrong reasoning in training data)
- Weld Defects (GPT-4.1 accuracy: 65.0% → ~35% wrong reasoning in training data)

## How to Run

```bash
# Step 1: Generate unconditioned CoT from GPT-4.1
python generate_unconditioned_cot.py

# Step 2: Train and evaluate
python run_unconditioned_cot.py
```

## Files

| File | Description |
|------|-------------|
| `generate_unconditioned_cot.py` | Calls GPT-4.1 to freely classify training images |
| `run_unconditioned_cot.py` | Trains with unconditioned data and evaluates |
| `results/` | Output JSON files |
| `README.md` | This file |

## Expected Outcome

Unconditioned CoT should perform significantly **worse** than both Direct and Conditioned CoT — possibly even worse than Direct LoRA — because it teaches the model to reason toward wrong answers. This directly proves answer-conditioning is essential.

## Note on API Access

Requires Azure OpenAI (GPT-4.1) API access. If API is unavailable, a simulated version can be constructed by randomly assigning ~70% of existing CoT descriptions to wrong classes (matching GPT-4.1's known error rate).
