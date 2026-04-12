# Task 4: LoRA Fine-Tuning — Granulometry Classification

Fine-tune Qwen2.5-VL-3B on 18 training images (2 per class) to classify concrete aggregate by max particle size (8/16/32mm) and grading (coarse/medium/fine). Compare two approaches:

1. Standard LoRA: direct fine-tuning on 18 labeled image-JSON pairs
2. SEAL-inspired LoRA: use GPT-4.1 (Azure OpenAI) as a frontier teacher to generate augmented training data from the same 18 images, then fine-tune on the enriched dataset

## Baseline (Task 3)

| Model | Mode | Size | Grading | Both |
|-------|------|------|---------|------|
| Qwen2.5-VL-3B | Zero-shot | 36.1% | 34.3% | 12.0% |
| Qwen2.5-VL-3B | Few-shot | 36.1% | 24.1% | 8.3% |
| GPT-5 | Few-shot | 66.7% | 33.3% | 22.2% |
| GPT-4.1 | Few-shot (t=0.7) | 62.0% | 59.3% | 29.6% |

Random chance = 33%. Qwen base model cannot do this task.

## Frontier Model Selection: GPT-4.1

GPT-4.1 (Azure OpenAI) was selected as the SEAL teacher over GPT-5 because:
- Best grading accuracy: 59.3% vs GPT-5's 33.3% (GPT-5 predicted "fine" for all 108 images)
- Temperature control: GPT-4.1 supports temperature tuning; GPT-5 is a reasoning model locked to t=1
- Faster: 4.7s/image vs 15.8s/image
- Cheaper: standard model pricing vs reasoning model pricing
- Predicts all 3 gradings: coarse (10), medium (38), fine (60) — not stuck on one answer

Endpoint: `ether-openai` (East US 2), deployment: `gpt-4.1`

---

## Grading Definitions (DIN 1045 Standard)

The dataset uses the German DIN 1045 standard grading curves A/B/C for concrete aggregate.
Grading describes the SHAPE of the particle size distribution curve — it is independent of the max particle size.

The 9 classes are formed by combining 3 max particle sizes (8, 16, 32mm) with 3 grading curves (A, B, C):

| | 8mm max | 16mm max | 32mm max |
|---|---------|----------|----------|
| A (coarse) | A8 | A16 | A32 |
| B (medium) | B8 | B16 | B32 |
| C (fine) | C8 | C16 | C32 |

### Curve A — Coarse (uniformly graded)
- Most particles are concentrated near the maximum size
- Very few small particles present
- Large visible gaps/voids between stones (not filled by smaller material)
- Surface appears as a single layer of similarly-sized stones
- Low packing density — you can see the background between particles
- Example: A32 = mostly 16-32mm stones with empty gaps between them

### Curve B — Medium (well-graded)
- Balanced mix of particle sizes from small to large
- Some smaller particles fill gaps between larger ones, but not completely
- Moderate packing density
- Example: B16 = mix of sizes up to 16mm, gaps partially filled

### Curve C — Fine (continuously graded)
- Wide range of particle sizes present simultaneously
- Many small particles densely fill ALL gaps between larger stones
- Very high packing density — almost no visible voids or background
- Surface appears tightly packed, dense, and heterogeneous
- Example: C32 = some 32mm stones but lots of 4-16mm particles filling every gap

### Key Visual Test
Look at the spaces between the largest stones:
- Gaps are EMPTY → coarse (A)
- Gaps are PARTIALLY filled → medium (B)
- Gaps are COMPLETELY filled with smaller particles → fine (C)

### Ground Sampling Distance (GSD)
Original images: 2200x3000 pixels at GSD = 8.0 pixels per mm.
At this resolution: 8mm stone = ~64px, 16mm = ~128px, 32mm = ~256px.

---

## Why LoRA (not QLoRA)

| Approach | Training precision | Quality | VRAM needed |
|----------|-------------------|---------|-------------|
| QLoRA | LoRA on 4-bit quantized model | Good (slight loss from quantized weights) | ~4-5 GB |
| LoRA (our choice) | LoRA on full BF16 model | Better (training sees full-precision weights) | ~7-8 GB |

Since we have 2x V100 (32 GB total), there is no VRAM constraint. We use full-precision LoRA for maximum quality. Quantization happens later in Task 5 for edge deployment.

---

## Approach A: Standard LoRA (18 examples)

Direct LoRA fine-tuning on 18 images with ground truth JSON labels.

Training format (JSONL):
```json
{"messages": [
  {"role": "user", "content": [
    {"type": "image", "image": "path/to/S1_A32_IMG_5952.JPG"},
    {"type": "text", "text": "Classify this concrete aggregate photo. GSD = 8.0 px/mm.\nRespond with JSON: {\"max_particle_size_mm\": <8|16|32>, \"grading\": \"<coarse|medium|fine>\"}"}
  ]},
  {"role": "assistant", "content": "{\"max_particle_size_mm\": 32, \"grading\": \"coarse\"}"}
]}
```

LoRA config:
- Rank: 16, Alpha: 32, Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Learning rate: 2e-5, Epochs: 20, Batch size: 1, Gradient accumulation: 4
- Model loaded in BF16 (full precision)

---

## Approach B: SEAL-Inspired LoRA (augmented data)

Adapts the SEAL (Self-Adapting LLMs) framework for vision-language models using GPT-4.1 as the external editor.

### What is SEAL?

SEAL (Self-Adapting LLMs) is a 2025 framework from MIT that enables models to generate their own fine-tuning data. It has two steps:
- Step A (self-editing): the model generates training Q&A pairs from new information, then fine-tunes on them via LoRA
- Step B (RL optimization): train the model to become a better self-editor (optional, expensive, minimal benefit for instruction-tuned models)

Key finding from the reproduction study: using a strong external editor (e.g. GPT-5) achieves comparable results to RL-based self-editing at a fraction of the cost. For instruction-tuned models, almost all improvement comes from Step A alone.

- Paper: [Self-Adapting Language Models (arxiv 2506.10943)](https://arxiv.org/abs/2506.10943)
- Reproduction: [Reproducing SEAL](https://wtzhang99.github.io/blog/reproducing-seal/)

### Our Adaptation (Novel)

SEAL was designed for text-only LLMs internalizing factual knowledge. We adapt it for a VLM doing visual classification of industrial images — this is a novel application.

Our approach:
1. Send each of the 18 training images to GPT-4.1 (Azure OpenAI) with the correct label
2. For each image, GPT-4.1 generates ~8 training variations:
   - Direct JSON classification (short prompt, JSON response)
   - Chain-of-thought reasoning leading to the correct classification
   - Visual feature description (what makes this class distinct)
   - Contrastive explanation (why this grading, not the others — using the gaps visual test)
   - Size-focused analysis (pixel estimation using GSD)
   - Grading-focused analysis (distribution pattern description)
   - Multiple prompt phrasings (with/without GSD, with/without definitions)
3. This produces ~144 training pairs from just 18 images
4. LoRA fine-tune Qwen2.5-VL-3B on the augmented dataset

### Why This Should Work Better

- GPT-4.1 can articulate visual reasoning the 3B model can't discover on its own
- Multiple phrasings teach the model to generalize, not memorize
- Chain-of-thought examples teach intermediate reasoning steps
- Contrastive examples teach decision boundaries between classes (the "gaps" test)
- ~8x more training data from the same 18 images

### What Makes This Novel

1. First application of SEAL-style data augmentation to VLMs (original paper is text-only LLMs)
2. Using a frontier vision model as the "external editor" for a domain-specific industrial vision task
3. Cross-modal knowledge distillation: frontier model's visual reasoning → small model's classification ability
4. Extremely low-data regime (18 images) where data augmentation matters most
5. Domain-specific adaptation: DIN 1045 grading conventions taught through generated training data

---

## Pipeline

```
Step 1: Prepare training data
  ├── Select 18 images (2 per class from train split)
  ├── Create direct JSONL (Approach A: 18 examples)
  └── Generate augmented JSONL via GPT-4.1 (Approach B: ~144 examples)

Step 2: Fine-tune (LoRA on BF16 model)
  ├── Approach A: LoRA on 18 direct examples → lora_direct/
  └── Approach B: LoRA on ~144 augmented examples → lora_augmented/

Step 3: Evaluate on 108 test images
  ├── Run both fine-tuned models
  ├── Compare against Task 3 baselines (Qwen, GPT-5, GPT-4.1)
  └── Compare Approach A vs Approach B

Step 4: (Task 5) Merge LoRA → Quantize to INT4 → Edge deployment
```

## Expected Results

| Metric | Qwen Base (Task 3) | After LoRA (expected) |
|--------|--------------------|-----------------------|
| JSON validity | 100% | ~100% |
| Size accuracy | 36% | ~70-80% |
| Grading accuracy | 34% | ~60-80% |
| Both correct | 12% | ~50-70% |

## Files

| File | Description |
|------|-------------|
| `prepare_training_data.py` | Select 18 training images, create JSONL for Approach A |
| `generate_augmented_data.py` | SEAL-inspired augmentation via GPT-4.1 (Approach B) |
| `fine_tune.py` | LoRA fine-tuning script for both approaches |
| `fine_tune_granulometry.ipynb` | Notebook: full pipeline with visualization |
| `evaluate.py` | Run fine-tuned model on test set, compare with baseline |
| `training_data_direct.jsonl` | Approach A training data (18 examples) |
| `training_data_augmented.jsonl` | Approach B training data (~144 examples) |
| `.env.example` | API key template for GPT-4.1 |
| `requirements.txt` | Python dependencies |

## Hardware

- Training: `slm-workbench` — 2x V100 16GB ($6.12/hr)
- LoRA on BF16 model fits in single V100 (~7-8 GB)
- Estimated training time: ~15-30 min per approach
- Estimated cost: ~$2-5 total (compute) + ~$1-2 (GPT-4.1 API for augmentation)

## References

- [SEAL: Self-Adapting Language Models (MIT, 2025)](https://arxiv.org/abs/2506.10943)
- [Reproducing SEAL — external editors match RL](https://wtzhang99.github.io/blog/reproducing-seal/)
- [Coenen et al. — Learning to Sieve: Prediction of Grading Curves from Images of Concrete Aggregate (2022)](https://arxiv.org/abs/2204.03333)
- [DIN 1045 — Concrete, reinforced and prestressed concrete structures](https://www.beuth.de/en/standard/din-1045-2/147411977)
- [Qwen2.5-VL Fine-Tuning Guide](https://connectaman.hashnode.dev/fine-tuning-the-qwen25-7b-vl-instruct-model-a-comprehensive-guide)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [Qwen2.5-VL LoRA with ai-toolkit](https://www.kombitz.com/2025/09/15/how-to-train-a-qwen-image-lora-with-ai-toolkit-with-ai-toolkit/)
