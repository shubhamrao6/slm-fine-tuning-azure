# Task 4 Technical Notes — LoRA Fine-Tuning for Granulometry

Comprehensive reference covering LoRA, training concepts, and design decisions for this task.

---

## 1. What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that freezes the original model weights and injects small trainable matrices into specific layers.

Paper: [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)

### How it works

In a standard transformer, each layer has weight matrices like W (e.g. 4096×4096 = 16M parameters). In full fine-tuning, you update all of W via gradient descent.

LoRA instead decomposes the weight update into two small matrices:
```
W_new = W_frozen + B × A
```
Where:
- W_frozen: original weights (frozen, no gradients)
- A: shape (d × r) — the "down projection"
- B: shape (r × d) — the "up projection"
- r: the rank (typically 8-64, we use 16)

For r=16 and d=4096: A has 65K params, B has 65K params = 130K total.
Compare to the full W: 16M params. That's a 99.2% reduction.

### Why it works

The key insight from the paper: weight updates during fine-tuning have low intrinsic rank. You don't need to update all 16M parameters — the meaningful changes can be captured in a much smaller subspace. LoRA exploits this by constraining updates to a low-rank decomposition.

### Training process

The training is standard gradient descent (backpropagation + Adam optimizer):
1. Forward pass: compute `output = (W_frozen + B×A) × input`
2. Compute loss (cross-entropy for language modeling)
3. Backward pass: compute gradients for A and B only (W_frozen gets no gradients)
4. Optimizer step: update A and B using Adam

This is identical to normal DNN training — the only difference is which parameters receive gradients.

---

## 2. LoRA Hyperparameters

### Rank (r = 16)

The rank determines the capacity of the LoRA adapter.

| Rank | Trainable params (per layer) | Use case |
|------|------------------------------|----------|
| 4 | ~33K | Very simple tasks, text classification |
| 8 | ~65K | Standard text tasks |
| 16 | ~130K | Vision-language tasks, our choice |
| 32 | ~262K | Complex tasks, larger datasets |
| 64 | ~524K | Approaching full fine-tuning capacity |

We chose r=16 because:
- Vision tasks need more capacity than pure text (images have more information)
- But we only have 18 training examples — too high a rank risks overfitting
- r=16 is the standard recommendation for VLM fine-tuning with small datasets

Paper reference: [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) found r=16 optimal for most tasks.

### Alpha (α = 32)

Alpha is a scaling factor. The LoRA output is scaled by α/r before being added to the frozen weights:
```
output = W_frozen × input + (α/r) × B × A × input
```

With α=32 and r=16, the scaling factor is 2.0. This means the LoRA contribution is amplified 2x.

| α/r ratio | Effect |
|-----------|--------|
| 0.5 | LoRA has weak influence, mostly relies on base model |
| 1.0 | Equal contribution |
| 2.0 | LoRA has strong influence (our choice) |
| 4.0 | Very strong, risk of instability |

We use α=2r (alpha = 2 × rank) which is the standard convention from the original LoRA paper.

### Dropout (0.05)

Random dropout applied to LoRA layers during training. Prevents overfitting by randomly zeroing some LoRA activations.

- 0.0: no regularization (risk of overfitting with small datasets)
- 0.05: light regularization (our choice — small dataset but also very few parameters)
- 0.1-0.2: stronger regularization (for larger datasets)

We use 0.05 because with only 18 examples, every training signal matters. Too much dropout wastes information.

### Target Modules

Which weight matrices in the transformer get LoRA adapters:

| Module | What it does | Why we target it |
|--------|-------------|-----------------|
| q_proj | Query projection in attention | Controls what the model "looks for" |
| k_proj | Key projection in attention | Controls what information is available |
| v_proj | Value projection in attention | Controls what information is extracted |
| o_proj | Output projection in attention | Controls how attention output is combined |
| gate_proj | MLP gating mechanism | Controls information flow in FFN |
| up_proj | MLP up-projection | Expands representation in FFN |
| down_proj | MLP down-projection | Compresses representation in FFN |

We target all 7 modules (attention + MLP) because:
- Vision tasks require changes in both how the model attends to image regions (attention) and how it processes features (MLP)
- With r=16, even targeting all modules keeps total trainable params small (~2-3% of the model)
- The Qwen2.5-VL fine-tuning guides recommend targeting all projection layers

### Learning Rate (2e-5)

| LR | Use case |
|----|----------|
| 1e-6 | Very conservative, for large datasets |
| 1e-5 | Standard full fine-tuning |
| 2e-5 | Standard LoRA fine-tuning (our choice) |
| 5e-5 | Aggressive, risk of instability |
| 1e-4 | Too high for most LoRA setups |

LoRA uses a slightly higher LR than full fine-tuning because the adapter parameters are initialized near zero and need to move further to have an effect.

### Epochs (20)

With only 18 training examples and effective batch size 4, one epoch = ~4-5 gradient steps. 20 epochs = ~90-100 total steps. This is low by normal standards but appropriate because:
- Very small dataset — the model sees each example 20 times
- LoRA has few parameters — converges faster than full fine-tuning
- Risk of overfitting increases with more epochs on small data
- We use cosine LR schedule which naturally decays

### Gradient Accumulation (4)

Simulates a larger batch size without needing more VRAM:
- Actual batch size: 1 (one image at a time)
- Accumulate gradients for 4 steps before updating
- Effective batch size: 1 × 4 = 4

Larger effective batch = more stable gradients = smoother training.

---

## 3. Numerical Precision: BF16 vs FP16 vs FP32

### What are these formats?

All three are floating-point number representations. The key difference is how they allocate bits:

```
FP32:  1 sign + 8 exponent + 23 mantissa = 32 bits
FP16:  1 sign + 5 exponent + 10 mantissa = 16 bits
BF16:  1 sign + 8 exponent +  7 mantissa = 16 bits
```

| Format | Size | Dynamic range | Precision | Max value |
|--------|------|--------------|-----------|-----------|
| FP32 | 4 bytes | ±3.4×10³⁸ | ~7 decimal digits | 3.4×10³⁸ |
| FP16 | 2 bytes | ±6.5×10⁴ | ~3 decimal digits | 65,504 |
| BF16 | 2 bytes | ±3.4×10³⁸ | ~2 decimal digits | 3.4×10³⁸ |

### Why BF16 for training?

BF16 (Brain Float 16) was designed by Google Brain specifically for deep learning:

1. Same range as FP32 (8 exponent bits) — no overflow/underflow during training
2. Half the memory of FP32 — a 3B model uses ~7 GB instead of ~14 GB
3. FP16 is dangerous for LLMs because its limited range (max 65,504) causes overflow in attention scores and gradient computations
4. Qwen2.5-VL was trained in BF16 originally — loading in BF16 is lossless
5. V100 GPUs support BF16 natively

The tradeoff: BF16 has less precision than FP16 (7 mantissa bits vs 10). But for classification into 9 classes, this precision loss is irrelevant.

Paper reference: [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740)

### Why not FP32?

FP32 would work but wastes VRAM. The model would take ~14 GB instead of ~7 GB, leaving less room for activations and gradients. No quality benefit for this task.

### Why not FP16?

FP16 can cause NaN errors during training of large models. The limited exponent range (5 bits, max 65,504) means attention scores can overflow. BF16 avoids this entirely.

### Quantization formats (INT8, INT4) — for inference only

| Format | Size | Use |
|--------|------|-----|
| INT8 | 1 byte | Post-training quantization for inference |
| INT4 (NF4) | 0.5 bytes | Aggressive quantization for edge deployment |

These are NOT used during training (except in QLoRA, which we chose not to use). They're for Task 5 — after training, we merge the LoRA adapter and quantize the full model for edge deployment.

---

## 4. LoRA vs QLoRA vs Full Fine-Tuning

| | Full Fine-Tuning | LoRA (our choice) | QLoRA |
|---|---|---|---|
| What's trained | All 3B parameters | ~2-3% of parameters (adapters) | ~2-3% of parameters (adapters) |
| Base model precision | BF16 | BF16 (full precision) | INT4 (4-bit quantized) |
| VRAM for 3B model | ~24 GB (model + optimizer) | ~10-12 GB (model + small optimizer) | ~5-6 GB (quantized model + small optimizer) |
| Training quality | Best | Very close to full | Slight degradation (training on quantized weights) |
| Training speed | Slowest | Fast | Fastest |
| Output | Modified full model (~7 GB) | Small adapter file (~50-100 MB) | Small adapter file (~50-100 MB) |
| Our hardware | Doesn't fit in 1x V100 16GB | Fits comfortably | Fits easily |

We chose LoRA over QLoRA because:
- We have 2x V100 = 32 GB total — no VRAM constraint
- LoRA trains on full-precision weights → better gradient signal → better quality
- The adapter files are the same size either way
- Quantization happens later in Task 5 for edge deployment

Paper: [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)

---

## 5. SEAL Framework

SEAL (Self-Adapting LLMs) enables models to generate their own fine-tuning data.

Paper: [Self-Adapting Language Models (Zweiger et al., MIT, 2025)](https://arxiv.org/abs/2506.10943)
Reproduction: [Reproducing SEAL (Zhang, 2025)](https://wtzhang99.github.io/blog/reproducing-seal/)

### Two steps

Step A — Self-editing: Given new information, the model generates training Q&A pairs, then fine-tunes on them via LoRA.

Step B — RL optimization: Train the model to become a better self-editor using reinforcement learning. The reward = performance after fine-tuning on the generated edits.

### Key findings from reproduction study

1. For instruction-tuned models, Step A alone provides most of the benefit
2. Step B (RL) adds minimal improvement but is 10x more expensive
3. Using a strong external editor (e.g. GPT-5) achieves comparable results at a fraction of the cost
4. The external editor approach is recommended for practical use

### Our adaptation (novel)

SEAL was designed for text-only LLMs internalizing factual knowledge. We adapt it for:
- Vision-language models (multimodal, not text-only)
- Visual classification (not knowledge incorporation)
- Industrial domain (DIN 1045 aggregate grading)
- Extremely low-data regime (18 images)

We use GPT-4.1 as the external editor because:
- Best grading accuracy among tested models (59.3% vs GPT-5's 33.3%)
- Supports temperature control (GPT-5 doesn't)
- 3x faster and cheaper than GPT-5
- Deployed on existing Azure infrastructure

---

## 6. Training Data Strategy

### Approach A: Direct (18 examples)

Each of the 18 images gets one training pair:
- Input: image + classification prompt
- Output: correct JSON label

Risk: severe overfitting. 18 examples with 20 epochs = each example seen 20 times. The model may memorize rather than generalize.

### Approach B: SEAL-augmented (~144 examples)

Each of the 18 images gets ~8 training pairs with varied prompts and response styles. This helps because:

1. Prompt diversity: the model learns to respond to different phrasings, not just one template
2. Chain-of-thought: teaches intermediate reasoning steps
3. Contrastive examples: explicitly teach decision boundaries ("this is coarse because gaps are empty, unlike fine where gaps are filled")
4. Visual descriptions: ground the classification in observable features
5. Reduced overfitting: 8x more examples from the same images

### Why not more training images?

We deliberately use only 18 images (2 per class) to prove that LoRA + SEAL can work in extremely low-data regimes. This matches the real-world industrial scenario: a customer provides 10-20 labeled images, and the system must learn from that.

---

## 7. Evaluation Strategy

Both fine-tuned models are evaluated on the same 108 test images used in Task 3, with the same prompts and parsing logic. This ensures a fair comparison:

| What we compare | Source |
|----------------|--------|
| Qwen2.5-VL-3B base (zero-shot) | Task 3 results |
| Qwen2.5-VL-3B base (few-shot) | Task 3 results |
| GPT-4.1 (few-shot, t=0.7) | Task 3 frontier results |
| Qwen + LoRA direct (18 ex) | Task 4 Approach A |
| Qwen + LoRA augmented (~144 ex) | Task 4 Approach B |

The test set is NEVER used for training. The 18 training images come from the train split (791 images), the 108 test images come from the test split.

---

## 8. What Happens After Training

1. Training produces adapter files in `lora_direct/` and `lora_augmented/` (~50-100 MB each)
2. To use: load base Qwen model + load adapter → merged model
3. Evaluate on 108 test images → compare with baselines
4. Winner goes to Task 5: merge adapter into base model → quantize to INT4 → deploy on edge

### Merging LoRA

```python
from peft import PeftModel
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model = PeftModel.from_pretrained(model, "lora_direct/")
model = model.merge_and_unload()  # LoRA weights baked into base model
model.save_pretrained("merged_model/")  # full model, no adapter needed
```

After merging, the model is a standard Qwen2.5-VL-3B with modified weights. No adapter loading needed at inference time. This merged model then gets quantized in Task 5.

---

## 9. References

| Topic | Paper | Link |
|-------|-------|------|
| LoRA | Low-Rank Adaptation of Large Language Models (Hu et al., 2021) | [arxiv 2106.09685](https://arxiv.org/abs/2106.09685) |
| QLoRA | Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023) | [arxiv 2305.14314](https://arxiv.org/abs/2305.14314) |
| SEAL | Self-Adapting Language Models (Zweiger et al., 2025) | [arxiv 2506.10943](https://arxiv.org/abs/2506.10943) |
| SEAL reproduction | Reproducing SEAL (Zhang, 2025) | [blog](https://wtzhang99.github.io/blog/reproducing-seal/) |
| Mixed precision | Mixed Precision Training (Micikevicius et al., 2018) | [arxiv 1710.03740](https://arxiv.org/abs/1710.03740) |
| Qwen2.5-VL | Qwen2.5-VL Technical Report (Alibaba, 2025) | [arxiv 2502.13923](https://arxiv.org/abs/2502.13923) |
| Dataset | Learning to Sieve (Coenen et al., 2022) | [arxiv 2204.03333](https://arxiv.org/abs/2204.03333) |
| PEFT library | Parameter-Efficient Fine-Tuning (HuggingFace) | [github](https://github.com/huggingface/peft) |
| Qwen VL fine-tuning | Fine-Tuning Qwen2.5-VL Guide | [blog](https://connectaman.hashnode.dev/fine-tuning-the-qwen25-7b-vl-instruct-model-a-comprehensive-guide) |
| DIN 1045 | Concrete structures standard | [beuth.de](https://www.beuth.de/en/standard/din-1045-2/147411977) |
