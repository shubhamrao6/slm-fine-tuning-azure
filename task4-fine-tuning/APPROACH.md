# SEAL-Inspired Chain-of-Thought Distillation for Industrial VLMs

## Overview

Fine-tune Qwen2.5-VL-3B (a small VLM) across 4 industrial vision tasks using LoRA, with two approaches:

- **Approach A (Direct)**: Image + Prompt → JSON
- **Approach B (SEAL-augmented)**: Image + Prompt → Description + JSON

The key innovation: a frontier model (GPT-4.1) generates rich, image-specific reasoning that justifies the correct classification. This reasoning is distilled into the small model during training.

---

## The Problem

Qwen2.5-VL-3B cannot classify industrial images out of the box:

| Task | Base ZS Accuracy | Random Chance |
|------|-----------------|---------------|
| Granulometry (9 classes) | 12.0% | 11.1% |
| Steel Surface (6 classes) | 21.7% | 16.7% |
| UHCS Microstructure (5 classes) | 60.8%* | 20.0% |
| Weld Defects (4 classes) | 30.8% | 25.0% |

*UHCS base accuracy is inflated by spheroidite majority class bias (62% of test set).

The model can see the images but lacks domain knowledge. It doesn't know that empty gaps between stones = coarse grading, or that dark circular spots in a radiograph = porosity.

---

## The Training Data Flow

Both approaches use the exact same input format. The only difference is the response.

### Input (same for both approaches, same for training and evaluation)

```
[IMAGE] + [PROMPT with detailed class definitions]
```

The prompt includes:
- Image context (size, modality, magnification where applicable)
- All class definitions with detailed visual descriptions
- Output format instruction (JSON)

**Critical**: the training prompt is IDENTICAL to the benchmarking prompt that produced the frontier model results. Using weaker definitions degrades performance.

### Response — Approach A (Direct)

Just the JSON:
```json
{"defect_class": "porosity"}
```

### Response — Approach B (SEAL-augmented)

Description + JSON:
```
The image shows several scattered small dark circular spots distributed across the weld area. These spots are characteristically round, consistent with gas pores trapped during solidification. This is not lack_of_penetration, which would show a continuous dark line along the weld centerline, nor cracks, which would appear as sharp jagged lines.
{"defect_class": "porosity"}
```

---

## How The Description Is Generated

The frontier model (GPT-4.1) receives:
1. **The actual image** — so the description is grounded in real visual content
2. **The correct answer** — so the reasoning is guaranteed to lead to the right classification
3. **Domain definitions** — so it uses correct terminology
4. **Contrastive pairs** — so it explains why it's NOT the similar classes

The frontier model is NOT classifying. It already knows the answer. It is explaining WHY the answer is correct based on what it sees.

### Why answer-conditioning is essential

Without the answer, GPT-4.1's accuracy varies by task:
- Granulometry: 29.6% (both correct) — would produce 70% wrong reasoning
- Steel surface: 91.1% FS — would produce 9% wrong reasoning
- UHCS: 71.7% FS — would produce 28% wrong reasoning
- Weld defects: 65.0% FS — would produce 35% wrong reasoning

By providing the answer, we get 100% correct reasoning in ALL training data.

### Prompt structure for CoT examples

For CoT training examples, the user prompt STRIPS the last 2 lines ("Respond with ONLY a JSON object:" and the JSON template). This way:
- Direct examples: user says "respond with JSON" → assistant outputs JSON
- CoT examples: user doesn't mention JSON → assistant outputs description + JSON

The code appends the correct JSON to GPT-4.1's description. GPT-4.1 is instructed to output ONLY the justification text, no JSON.

### Multiple descriptions per image

For each training image, we generate 3 descriptions (temperature=0.7 for variation) + 1 direct JSON = 4 training examples per image.

---

## Results Across 4 Tasks

### Granulometry (9 classes, 18 training images)

| Method | Size | Grading | Both |
|--------|------|---------|------|
| Qwen base ZS | 36.1% | 34.3% | 12.0% |
| GPT-4.1 FS | 62.0% | 59.3% | 29.6% |
| Direct LoRA | 89.8% | 78.7% | 71.3% |
| **SEAL LoRA** | **91.7%** | **86.1%** | **79.6%** |

SEAL adds +8.3pp on combined accuracy. The 3B model outperforms GPT-4.1 by 50pp.

### Steel Surface Defects (6 classes, 30 training images)

| Method | Accuracy |
|--------|----------|
| Qwen base ZS | 21.7% |
| GPT-4.1 FS | 91.1% |
| Direct LoRA | 63.1% |
| **SEAL LoRA** | **66.7%** |

SEAL adds +3.6pp. Both beat GPT-4.1 ZS (46.9%). Gap to GPT-4.1 FS reflects the difficulty of 6 visually similar grayscale classes.

### UHCS Microstructure (5 classes, 30 training images)

| Method | Accuracy |
|--------|----------|
| Qwen base ZS | 60.8% |
| GPT-4.1 FS | 71.7% |
| Direct LoRA | 67.5% |
| **SEAL LoRA** | **68.4%** |

SEAL adds +0.9pp. Approaches GPT-4.1 FS. Compound classes (spheroidite+widmanstatten) improved from 0% to 27%.

### Weld Defects (4 classes, 24 training images)

| Method | Accuracy |
|--------|----------|
| Qwen base ZS | 30.8% |
| GPT-4.1 FS | 65.0% |
| Direct LoRA | 73.3% |
| **SEAL LoRA** | **75.8%** |

SEAL adds +2.5pp. **Both beat GPT-4.1 FS** — the 3B model outperforms the frontier model by 10.8pp. Cracks went from 0% (all models ZS) to 58% with 6 training images.

---

## Why SEAL Works Better Than Direct

| What Direct learns | What SEAL learns |
|---|---|
| "This specific image = porosity" | "Circular dark spots = porosity, jagged lines = cracks" |
| Memorizes 24-30 images | Learns transferable visual reasoning patterns |
| No decision boundary knowledge | Explicitly taught contrastive features |
| 24-30 training examples | 96-120 training examples from same images |

With only 24-30 training images, memorization is a real risk. The CoT descriptions force the model to learn generalizable features instead of memorizing specific images.

---

## The Complete Pipeline

```
STEP 1: Select training images (5-6 per class from train split)

STEP 2A: Create Direct training data
  └── For each image: [image + full prompt] → [JSON]

STEP 2B: Create SEAL-augmented training data
  └── For each image:
      ├── Send image + correct answer + definitions to GPT-4.1
      ├── GPT-4.1 generates 3 justified descriptions (t=0.7)
      ├── Code appends correct JSON to each description
      ├── Also create 1 direct JSON-only pair
      └── 4 training pairs per image

STEP 3: LoRA fine-tune Qwen2.5-VL-3B
  ├── Approach A: train on direct data → lora_direct/
  └── Approach B: train on augmented data → lora_augmented/

STEP 4: Evaluate both on held-out test set
  ├── Same prompt as training
  ├── Parse JSON from response (extract from description if needed)
  └── Compare: Approach A vs Approach B vs baselines
```

---

## Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank (r) | 16 | Standard for VLM fine-tuning |
| LoRA alpha | 32 | α/r = 2.0, standard scaling |
| LoRA dropout | 0.05 | Light regularization |
| Target modules | q,k,v,o,gate,up,down_proj | All projections for maximum capacity |
| Learning rate | 2e-5 | Standard LoRA LR, proven across all 4 tasks |
| Epochs | 40 | Sufficient for convergence on small datasets |
| Batch size | 1 | VRAM constraint |
| Gradient accumulation | 4 | Effective batch = 4 |
| Precision | BF16 | Full precision, no quantization during training |
| Gradient checkpointing | Enabled | Saves ~2-3 GB VRAM |
| Scheduler | Cosine with 10% warmup | Smooth convergence |
| Eval temperature | 0.1 | Low variance for consistent evaluation |
| Eval max_new_tokens | 256 | Sufficient for description + JSON |

---

## What Makes This Novel

1. **Answer-conditioned CoT distillation from frontier VLM to small VLM** — the frontier model's visual reasoning is baked into training data with guaranteed correctness
2. **Applied across 4 industrial domains** — construction, steel manufacturing, metallurgy, NDT welding
3. **3 different image modalities** — macro photography, optical/SEM microscopy, X-ray radiography
4. **Extremely low-data regime** — 18-30 images per task, where every training example matters
5. **Small model beats frontier model** — on weld defects, the 3B model outperforms GPT-4.1 FS
6. **Consistent prompt design** — same detailed prompt used in training and evaluation, matching the benchmarking definitions

---

## References

- [SEAL: Self-Adapting Language Models (MIT, 2025)](https://arxiv.org/abs/2506.10943)
- [Reproducing SEAL (Zhang, 2025)](https://wtzhang99.github.io/blog/reproducing-seal/)
- [LoRA: Low-Rank Adaptation (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Distilling Step-by-Step (Hsieh et al., 2023)](https://arxiv.org/abs/2305.02301)
- [Coenen et al. — Learning to Sieve (2022)](https://arxiv.org/abs/2204.03333)
- [NEU Surface Defect Database — Song & Yan (2013)](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- [UHCS Microstructure — NIST](https://materialsdata.nist.gov/handle/11256/940)
- [RIAWELC Weld Defects — Totino et al. (2022)](https://www.researchgate.net/publication/369294451)
