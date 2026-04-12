# Task 4: LoRA Fine-Tuning — Granulometry Classification

Fine-tune Qwen2.5-VL-3B on 18 training images (2 per class) to classify concrete aggregate by max particle size (8/16/32mm) and grading (coarse/medium/fine). Compare two approaches:

1. Standard LoRA: direct fine-tuning on 18 labeled image-JSON pairs
2. SEAL-inspired LoRA: use a frontier model (Claude Opus 4.6 / GPT-5) to generate augmented training data from the same 18 images, then fine-tune on the enriched dataset

## Baseline (Task 3)

| Metric | Zero-Shot | Few-Shot |
|--------|-----------|----------|
| Size accuracy | 36.1% | 36.1% |
| Grading accuracy | 34.3% | 24.1% |
| Both correct | 12.0% | 8.3% |

Random chance = 33%. The base model cannot do this task.

## Approach A: Standard LoRA (18 examples)

Direct QLoRA fine-tuning on 18 images with ground truth JSON labels.

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

QLoRA config:
- Rank: 16, Alpha: 32, Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Learning rate: 2e-5, Epochs: 20, Batch size: 1, Gradient accumulation: 4
- 4-bit quantization (NF4) for memory efficiency

## Approach B: SEAL-Inspired LoRA (augmented data)

Adapts the SEAL (Self-Adapting LLMs) framework for vision-language models.

### What is SEAL?

SEAL (Self-Adapting LLMs) is a 2025 framework from MIT that enables models to generate their own fine-tuning data. The key insight: instead of training directly on raw examples, have a strong model transform the data into a format optimized for learning.

- Paper: [Self-Adapting Language Models (arxiv 2506.10943)](https://arxiv.org/abs/2506.10943)
- Reproduction study: [Reproducing SEAL](https://wtzhang99.github.io/blog/reproducing-seal/) — found that using a strong external editor (GPT-5) matches RL-based self-editing at a fraction of the cost

### Our Adaptation (Novel)

SEAL was designed for text-only LLMs internalizing factual knowledge. We adapt it for a VLM doing visual classification — this is a novel application. Our approach:

1. Send each of the 18 training images to a frontier vision model (Claude Opus 4.6 or GPT-5)
2. For each image, the frontier model generates multiple training variations:
   - Chain-of-thought reasoning that leads to the correct classification
   - Multiple prompt phrasings (with/without GSD, with/without grading definitions)
   - Descriptions of visual features that distinguish this class from others
   - Contrastive examples ("this is coarse because X, unlike fine which would show Y")
3. This produces ~8-10 training pairs per image = ~144-180 total from just 18 images
4. QLoRA fine-tune Qwen2.5-VL-3B on the augmented dataset

### Why This Should Work Better

- The frontier model can articulate visual reasoning the small model can't discover on its own
- Multiple phrasings teach the model to generalize, not memorize
- Chain-of-thought examples teach intermediate reasoning steps
- Contrastive examples teach decision boundaries between classes
- 10x more training data from the same 18 images

### What Makes This Novel

1. First application of SEAL-style data augmentation to VLMs (original paper is text-only LLMs)
2. Using a frontier vision model as the "external editor" for a domain-specific industrial vision task
3. Cross-modal knowledge distillation: frontier model's visual reasoning → small model's classification ability
4. Extremely low-data regime (18 images) where data augmentation matters most

## Grading Definitions (DIN 1045)

Grading describes the particle size distribution, not absolute sizes:
- Coarse (A): most particles are similar size, close to max. Uniform texture. Few small particles.
- Medium (B): moderate mix of large and small particles.
- Fine (C): wide range of sizes. Many small particles fill gaps between larger ones. Dense, packed texture.

## Pipeline

```
Step 1: Prepare training data
  ├── 18 images (2 per class from train split)
  ├── Ground truth labels from manifest
  └── Generate augmented data via frontier model (Approach B only)

Step 2: Fine-tune
  ├── Approach A: QLoRA on 18 direct examples
  └── Approach B: QLoRA on ~150 augmented examples

Step 3: Evaluate
  ├── Run both fine-tuned models on 108 test images
  ├── Compare against Task 3 baseline
  └── Compare Approach A vs Approach B
```

## Files

| File | Description |
|------|-------------|
| `prepare_training_data.py` | Select 18 training images, create JSONL for Approach A |
| `generate_augmented_data.py` | SEAL-inspired augmentation via frontier model (Approach B) |
| `fine_tune.py` | QLoRA fine-tuning script for both approaches |
| `evaluate.py` | Run fine-tuned model on test set, compare with baseline |
| `training_data_direct.jsonl` | Approach A training data (18 examples) |
| `training_data_augmented.jsonl` | Approach B training data (~150 examples) |

## Hardware

- Training: `slm-workbench` — 2x V100 16GB ($6.12/hr)
- QLoRA with 4-bit quantization fits in single V100
- Estimated training time: ~15-30 min per approach
- Estimated cost: ~$2-5 total

## References

- [SEAL: Self-Adapting Language Models (MIT, 2025)](https://arxiv.org/abs/2506.10943)
- [Reproducing SEAL — external editors match RL](https://wtzhang99.github.io/blog/reproducing-seal/)
- [Qwen2.5-VL Fine-Tuning Guide](https://connectaman.hashnode.dev/fine-tuning-the-qwen25-7b-vl-instruct-model-a-comprehensive-guide)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [Qwen2.5-VL LoRA with ai-toolkit](https://www.kombitz.com/2025/09/15/how-to-train-a-qwen-image-lora-with-ai-toolkit-with-ai-toolkit/)
