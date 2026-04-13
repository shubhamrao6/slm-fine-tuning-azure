# Task 4: SEAL-Inspired Chain-of-Thought Distillation for Granulometry

## Overview

Fine-tune Qwen2.5-VL-3B (a small VLM) to classify concrete aggregate images into 9 classes using LoRA, with two approaches:

- **Approach A (Direct)**: Image + Prompt → JSON
- **Approach B (SEAL-augmented)**: Image + Prompt → Description + JSON

The key innovation in Approach B: a frontier model (GPT-4.1) generates rich, image-specific reasoning that justifies the correct classification. This reasoning is distilled into the small model during training.

---

## The Problem

Qwen2.5-VL-3B cannot classify concrete aggregate out of the box:
- Size accuracy: 36.1% (random chance = 33.3%)
- Grading accuracy: 34.3% (random chance = 33.3%)

The model can see the image but lacks domain knowledge. It doesn't know that:
- "Empty gaps between stones" = coarse grading (DIN 1045 curve A)
- "Completely filled gaps" = fine grading (DIN 1045 curve C)
- "8 pixels per mm" means a 128px stone is 16mm

Even frontier models struggle: GPT-4.1 achieves 62% size / 59.3% grading — better than random but far from reliable.

---

## The 9 Classes

Two independent axes, 3 values each:

| | 8mm max | 16mm max | 32mm max |
|---|---------|----------|----------|
| Coarse (A) | A8 | A16 | A32 |
| Medium (B) | B8 | B16 | B32 |
| Fine (C) | C8 | C16 | C32 |

**Max particle size** (8, 16, 32mm): the diameter of the largest stone in the image.

**Grading** (DIN 1045 standard): the shape of the particle size distribution curve.
- Coarse (curve A): most particles are concentrated near the max size. Gaps between stones are EMPTY. Surface looks uniform, single-layer.
- Medium (curve B): balanced mix of sizes. Gaps PARTIALLY filled by smaller particles.
- Fine (curve C): wide range of sizes. Gaps COMPLETELY filled with small particles. Surface looks dense, packed, heterogeneous.

**Key visual test**: look at the spaces between the largest stones.
- Gaps EMPTY → coarse
- Gaps PARTIALLY filled → medium
- Gaps COMPLETELY filled → fine

---

## The Training Data Flow

Both approaches use the exact same input format. The only difference is the response.

### Input (same for both approaches, same for training and evaluation)

```
[IMAGE] + [PROMPT]
```

**Image**: the actual aggregate photograph, resized to MAX_DIM=800px (processor handles tokenization via max_pixels setting).

**Prompt**: a detailed, consistent prompt used everywhere — training, evaluation, both approaches:

```
Classify this concrete aggregate photograph.
Ground sampling distance (GSD) = {gsd} px/mm.
At this GSD: 8mm stone ≈ {8*gsd}px, 16mm ≈ {16*gsd}px, 32mm ≈ {32*gsd}px.

Classification axes:
1. MAX PARTICLE SIZE: estimate the largest stone's width in pixels, divide by GSD,
   round to 8, 16, or 32 mm.
2. GRADING (DIN 1045 standard — describes size DISTRIBUTION, not absolute size):
   - COARSE (A): particles concentrated near max size. Gaps between stones are EMPTY.
     Uniform, single-layer texture.
   - MEDIUM (B): balanced mix. Gaps PARTIALLY filled by smaller particles.
   - FINE (C): wide size range. Gaps COMPLETELY filled with small particles.
     Dense, packed texture.

Respond with JSON: {"max_particle_size_mm": <8|16|32>, "grading": "<coarse|medium|fine>"}
```

The `{gsd}` value is computed dynamically based on the actual image resize:
- Original images: 2200×3000 at GSD = 8.0 px/mm
- After resize to 800px max: scale = 800/3000 = 0.267, GSD = 8.0 × 0.267 = 2.1 px/mm
- The prompt always reflects the actual GSD of the image the model sees

This prompt is identical in training and evaluation. The model always sees the same instruction format.

### Response — Approach A (Direct)

Just the JSON:
```json
{"max_particle_size_mm": 32, "grading": "coarse"}
```

The model learns pure pattern matching: this visual pattern → this label.

### Response — Approach B (SEAL-augmented)

Description + JSON:
```
The largest stones in this image are approximately 30mm across, visible as the dominant
particles spanning roughly 63 pixels at the current GSD. Looking at the gaps between
these large stones, they are clearly empty — no smaller particles fill the spaces between
them. The surface has a uniform, single-layer appearance with stones of similar size,
which is characteristic of DIN 1045 curve A (coarse grading). The lack of fine material
filling the voids confirms this is not medium or fine grading.
{"max_particle_size_mm": 32, "grading": "coarse"}
```

The model learns reasoning + classification: this visual pattern → this is why → this label.

---

## How The Description Is Generated

The frontier model (GPT-4.1, Azure OpenAI) generates the description. Critically, it receives:

1. **The actual image** — so the description is grounded in real visual content
2. **The correct answer** — so the reasoning is guaranteed to lead to the right classification
3. **The DIN 1045 definitions** — so it uses the correct domain terminology
4. **Specific instructions** — to describe gaps, particle sizes, and texture

The frontier model is NOT classifying the image. It already knows the answer. It is explaining WHY the answer is correct based on what it sees. This is the key distinction.

### Why the frontier model needs the correct answer

Without the answer, GPT-4.1 gets grading wrong ~40% of the time (proven in Task 3 benchmarking). If we let it classify and describe simultaneously, 40% of our training data would contain wrong reasoning leading to wrong labels. The small model would learn incorrect patterns.

By providing the answer, we get:
- 100% correct labels (we supply them from ground truth)
- Correct reasoning grounded in the actual image (GPT-4.1 can see it)
- Domain-specific language (we provide DIN 1045 definitions)

The frontier model acts as an expert teacher who knows the answer and explains it while looking at the actual material.

### Multiple descriptions per image

For each training image, we generate 3 descriptions (using temperature=0.7 for variation). Each description highlights slightly different aspects:
- One might focus on gap patterns
- Another might emphasize particle size estimation
- A third might use contrastive reasoning ("this is NOT fine because...")

Plus 1 direct JSON-only response (no reasoning) per image.

Total per image: 3 justified + 1 direct = 4 training examples.

---

## Why This Should Work

### The small model's limitation

Qwen2.5-VL-3B can see the image but can't reason about it in the context of DIN 1045. It doesn't know what "empty gaps" means for grading classification. When forced to output JSON directly, it defaults to the most common pattern (16mm/medium for everything).

### What the descriptions teach

The descriptions bridge the gap between visual perception and domain knowledge:

| What the model sees | What the description teaches | What the model learns |
|---|---|---|
| Stones with spaces between them | "The gaps are empty, no small particles fill them" | Empty gaps = coarse |
| Dense packed surface | "Small particles completely fill all gaps between larger stones" | Filled gaps = fine |
| Large stones ~256px wide | "At GSD 2.1, these stones are approximately 32mm" | Big stones = 32mm class |

After training on enough of these description→JSON pairs, the model internalizes the reasoning patterns. When it sees a new image with empty gaps, it has learned (from the descriptions) that empty gaps → coarse, even though it couldn't figure this out on its own.

### Why descriptions are better than just labels

| Training approach | What the model learns | Generalization |
|---|---|---|
| Direct (JSON only) | "This specific image = coarse 32mm" | Memorizes 9-18 images |
| SEAL (description + JSON) | "Empty gaps = coarse, filled gaps = fine, big stones = 32mm" | Learns transferable reasoning patterns |

With only 9-18 training images, memorization is a real risk. The descriptions force the model to learn generalizable features instead of memorizing specific images.

---

## The Complete Pipeline

```
STEP 1: Select training images
  └── 9 images (1 per class) or 18 images (2 per class) from train split

STEP 2A: Create Direct training data (Approach A)
  └── For each image: [image + prompt] → [JSON]
  └── 9 or 18 training pairs

STEP 2B: Create SEAL-augmented training data (Approach B)
  └── For each image:
      ├── Send image + correct answer + DIN definitions to GPT-4.1
      ├── GPT-4.1 generates 3 justified descriptions (temperature=0.7)
      ├── Each description ends with the correct JSON
      ├── Also create 1 direct JSON-only pair
      └── 4 training pairs per image = 36 or 72 total

STEP 3: LoRA fine-tune Qwen2.5-VL-3B
  ├── Approach A: train on direct data → save adapter to lora_direct/
  ├── Free memory
  └── Approach B: train on augmented data → save adapter to lora_augmented/

STEP 4: Evaluate both on 108 test images
  ├── Same prompt as training (with correct GSD for eval resolution)
  ├── Parse JSON from response (ignore any reasoning text, extract JSON)
  └── Compare: Approach A vs Approach B vs baselines

STEP 5: (Task 5) Winner gets merged + quantized for edge deployment
```

---

## Evaluation Strategy

At evaluation time, the model receives the same prompt as training. It may respond with:
- Just JSON (learned from direct examples)
- Description + JSON (learned from augmented examples)

The parser extracts the JSON regardless — it searches for `{"max_particle_size_mm": ..., "grading": "..."}` anywhere in the response.

### What we measure

| Metric | What it tells us |
|---|---|
| Size accuracy | Can the model estimate the largest particle size? |
| Grading accuracy | Can the model assess the distribution pattern? |
| Both correct | Can it get both right simultaneously? |
| JSON validity | Does the model output parseable JSON? |

### What we compare

| Method | Training data | Expected result |
|---|---|---|
| Qwen base (no training) | None | ~33% (random) |
| GPT-4.1 (no training, API) | None | ~60% size, ~59% grading |
| Direct LoRA (18 images) | 18 × [image→JSON] | ~82% size, ~71% grading (proven) |
| Direct LoRA (9 images) | 9 × [image→JSON] | Lower than 18 (less data) |
| SEAL LoRA (9 images) | 9 × 4 = 36 [image→description+JSON] | Should beat Direct-9 |
| SEAL LoRA (18 images) | 18 × 4 = 72 [image→description+JSON] | Should beat Direct-18 |

The key comparison: Direct-9 vs SEAL-9. If SEAL wins with the same 9 images, it proves that the frontier model's reasoning descriptions add value beyond just having more training examples.

---

## Training Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| LoRA rank (r) | 16 | Standard for VLM fine-tuning |
| LoRA alpha | 32 | α/r = 2.0, standard scaling |
| LoRA dropout | 0.05 | Light regularization |
| Target modules | q,k,v,o,gate,up,down_proj | All projections for maximum capacity |
| Learning rate | 1e-5 (SEAL) / 2e-5 (Direct) | Lower for SEAL (more data, avoid overfitting) |
| Epochs | 15 (SEAL) / 40 (Direct) | Fewer for SEAL (more examples per epoch) |
| Batch size | 1 | VRAM constraint |
| Gradient accumulation | 4 | Effective batch = 4 |
| Precision | BF16 | Full precision, no quantization during training |
| Gradient checkpointing | Enabled | Saves ~2-3 GB VRAM |
| Image resolution | 800px max (processor max_pixels) | Fits in VRAM with gradients |

---

## What Makes This Novel

1. **Chain-of-thought distillation from frontier VLM to small VLM** — the frontier model's visual reasoning is baked into training data, not just its labels

2. **Answer-conditioned justification** — the frontier model is given the correct answer and asked to explain why, ensuring 100% correct reasoning in training data

3. **Applied to industrial domain-specific vision** — DIN 1045 aggregate grading, a task no model was trained on

4. **Extremely low-data regime** — 9-18 images, where every training example matters

5. **Image-grounded descriptions** — the frontier model sees the actual image, so descriptions reference real visual features, not generic text

6. **Consistent prompt design** — the same detailed prompt (with DIN definitions, GSD, output format) is used in training and evaluation, eliminating train/eval mismatch

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Model outputs long descriptions at eval instead of JSON | Include direct JSON-only examples in training mix (1 per image) |
| Frontier model generates wrong reasoning despite correct answer | Review a sample of generated descriptions before training |
| Overfitting on small dataset | Use fewer epochs (15), lower LR (1e-5), LoRA dropout (0.05) |
| Description quality varies | Generate 3 per image (temperature=0.7), all grounded in the actual image |
| Parser can't find JSON in long response | Parser searches for `{...}` pattern anywhere in response |

---

## References

- [SEAL: Self-Adapting Language Models (MIT, 2025)](https://arxiv.org/abs/2506.10943) — framework for models generating their own training data
- [Reproducing SEAL (Zhang, 2025)](https://wtzhang99.github.io/blog/reproducing-seal/) — external editors match RL-based self-editing
- [LoRA: Low-Rank Adaptation (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) — parameter-efficient fine-tuning
- [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903) — reasoning improves LLM performance
- [Distilling Step-by-Step (Hsieh et al., 2023)](https://arxiv.org/abs/2305.02301) — distilling reasoning from large to small models
- [Coenen et al. — Learning to Sieve (2022)](https://arxiv.org/abs/2204.03333) — the granulometry dataset
- [DIN 1045 — Concrete structures standard](https://www.beuth.de/en/standard/din-1045-2/147411977) — grading curve definitions
