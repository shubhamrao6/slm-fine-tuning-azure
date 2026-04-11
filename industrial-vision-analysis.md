# Industrial Computer Vision Device — Comprehensive Analysis

## Use Case

Building a computer vision device for industrial environments with requirements including:
- Object detection and counting
- Segmentation
- Granulometry (particle size measurement)
- Requirements change frequently across customers/deployments
- Changes need to be deployed quickly with minimal data (10-20 labeled images)
- Must run locally on NVIDIA edge devices — no internet dependency
- Low cost, fast inference, high precision

## Why Vision Language Models (VLMs)

Traditional CV approach: train a new model from scratch for each task. Needs hundreds/thousands of images, days of training, ML expertise to deploy.

VLM approach: one base model that already understands images. Change behavior via:
- **Prompt changes** (zero data): "count bolts" → "count screws" — instant
- **LoRA fine-tuning** (10-20 images): teach a new output format or domain-specific task in ~15-30 min on a cloud GPU, produces a ~50 MB adapter file that swaps in <1 second on the edge device

The base model stays the same across all deployments. Only the LoRA adapter changes per use case.

---

## Models Evaluated

### Tiny Models — Run at Full Precision (FP16), No Quantization Needed

| Model | Params | Image | Video | Segmentation | FP16 VRAM | License | Best For |
|-------|--------|-------|-------|-------------|-----------|---------|----------|
| Florence-2-base | 0.23B | Yes | No | Yes (native) | ~0.5 GB | MIT | Fast detection + grounding with native bbox output |
| SmolVLM-256M | 0.26B | Yes | No | No | <1 GB | Apache 2.0 | Ultra-lightweight image Q&A |
| LFM2.5-VL-450M | 0.45B | Yes | No | No | ~1 GB | Apache 2.0 | Built-in bbox prediction |
| SmolVLM-500M | 0.5B | Yes | No | No | ~1 GB | Apache 2.0 | Basic VQA |
| Florence-2-large | 0.77B | Yes | No | Yes (native) | ~1.5 GB | MIT | Better quality Florence |
| Mini-InternVL-1B | 1B | Yes | No | No | ~2 GB | MIT | Compact general VLM |
| Moondream2 | 1.9B | Yes | No | No | ~3.8 GB | Apache 2.0 | Good spatial understanding |

### Small Models — Need INT4 Quantization for Edge Deployment

| Model | Params | Image | Video | Audio | FP16 VRAM | INT4 VRAM | License | Best For |
|-------|--------|-------|-------|-------|-----------|-----------|---------|----------|
| Gemma 4 E2B | 2B (MoE) | Yes | Yes | Yes | ~4 GB | ~1.5 GB | Apache 2.0 | Full omni-modal, very efficient |
| SmolVLM2-2.2B | 2.2B | Yes | Yes | No | ~5.2 GB | ~2-3 GB | Apache 2.0 | Designed for low-resource devices |
| PaliGemma2-3B | 3B | Yes | No | No | ~6 GB | ~2 GB | Apache 2.0 | Strong OCR, captioning, detection |
| Kimi-VL-A3B | 2.8B active | Yes | Yes | No | ~6 GB | ~2.5 GB | MIT | MoE, 128K context |
| Qwen2.5-VL-3B | 3B | Yes | Yes | No | ~7 GB | ~2.5 GB | Apache 2.0 | Best quality at 3B size |
| Gemma 4 E4B | 4B (MoE) | Yes | Yes | Yes | ~8 GB | ~2.5 GB | Apache 2.0 | Bigger E2B sibling |

### Medium Models — Best Quality, Need 8-16 GB VRAM

| Model | Params | Image | Video | Audio | FP16 VRAM | INT4 VRAM | License | Best For |
|-------|--------|-------|-------|-------|-----------|-----------|---------|----------|
| Phi-3.5-vision | 4.2B | Yes | Frames | No | ~8.4 GB | ~3 GB | MIT | Microsoft Phi family |
| Phi-4-multimodal | 5.6B | Yes | Frames | Yes | ~12 GB | ~4-5 GB | MIT | Multimodal with audio |
| Qwen2.5-VL-7B | 7B | Yes | Yes | No | ~14 GB | ~4.5 GB | Apache 2.0 | Best overall quality for edge |
| InternVL3-8B | 8B | Yes | Yes | No | ~16 GB | ~5 GB | Apache 2.0 | Strong benchmarks |
| MiniCPM-V 2.6 | 8B | Yes | Yes | No | ~16 GB | ~5 GB | Apache 2.0 | GPT-4o level on phone |

---

## Model Type Comparison

| | Pure VLM (Phi-4, Qwen, InternVL) | Vision Foundation (Florence-2) | Traditional Detector (YOLO) |
|---|---|---|---|
| How it works | Generates text tokens that describe/analyze the image | Generates structured tokens (bbox, masks) for predefined vision tasks | Directly outputs bbox coordinates via regression heads |
| Prompt flexibility | Any natural language prompt — unlimited task variety | Fixed task prompts only (`<OD>`, `<SEGMENTATION>`, etc.) | No prompts — one model per task |
| Bbox precision | Approximate — coordinates generated as text | Good — trained on 5.4B spatial annotations | Excellent — purpose-built for this |
| Segmentation | Via prompt (quality varies) | Native for predefined tasks | Instance segmentation variants exist |
| Granulometry | Can reason about it via prompt | Not built-in | Not built-in |
| Custom tasks (10-20 images) | LoRA fine-tune output format, model already understands concepts | LoRA fine-tune for new object types, but can't add new task types | Full retrain needed, needs 100+ images |
| Speed (edge, INT4) | 20-70 tok/s depending on size | 100-200ms per image | 10-50ms per image |
| Structured output reliability | Can be inconsistent (we tested this with Phi-4) | Very reliable (native structured output) | Always reliable (fixed output format) |

---

## What We Tested: Phi-4-Multimodal on Azure

We deployed Phi-4-multimodal-instruct as a serverless endpoint on Azure AI Foundry and tested object detection with bounding boxes.

### Results
- **Text understanding**: Excellent — correctly identified objects in images
- **Structured JSON output**: Inconsistent — sometimes returned valid JSON with good bounding boxes, sometimes garbled text
- **Bounding box precision**: When it worked, boxes were approximate but reasonable (e.g., dog detected at [150,100,350,400] on a 500x500 image)
- **Reliability**: ~50% of attempts produced usable output. Retries helped but added latency
- **Root cause**: Serverless endpoint variability + small model struggling with simultaneous image understanding AND structured text generation

### Key Learnings
- VLMs are great at understanding images but unreliable at producing precise spatial coordinates as text
- Running locally on fixed hardware (vs serverless) would improve consistency
- Fine-tuning the output format with LoRA would significantly improve structured output reliability
- The model already knows what objects are — it just needs to learn the output format

---

## LoRA Fine-Tuning Comparison

| | Florence-2 | Phi-4-multimodal | Qwen2.5-VL-3B | Qwen2.5-VL-7B |
|---|---|---|---|---|
| LoRA supported | Yes | Yes | Yes | Yes |
| QLoRA (4-bit training) | Yes | Yes | Yes | Yes |
| Adapter size | ~10-30 MB | ~50-100 MB | ~30-80 MB | ~50-100 MB |
| Training time (10-20 images) | ~5-15 min | ~15-30 min | ~10-20 min | ~15-30 min |
| VRAM for training | ~2-3 GB | ~6-8 GB | ~4-6 GB | ~6-8 GB |
| Train on Jetson? | Possible on Orin NX | No — cloud only | Tight on Orin NX | No — cloud only |
| What you fine-tune | New object types, new detection domains | Output format + domain knowledge | Output format + domain knowledge | Output format + domain knowledge |
| Adapter swap time | <1 second | <1 second | <1 second | <1 second |

### How Use Case Switching Works in Production

```
Edge Device (Jetson Orin NX/AGX)
├── Base model: Qwen2.5-VL-7B INT4 (~4.5 GB, loaded once)
├── Adapter A: bolt_counting.bin (~50 MB)
├── Adapter B: particle_size.bin (~50 MB)  
├── Adapter C: defect_detection.bin (~50 MB)
└── Adapter D: label_ocr.bin (~50 MB)
```

Switching from "count bolts" to "measure particles" = load a different 50 MB file. Takes <1 second. No model reload.

### Workflow for Each New Customer/Use Case
1. Customer provides 10-20 labeled images
2. LoRA fine-tune on cloud GPU (~15-30 min, ~$1 on Azure)
3. Get a ~50 MB adapter file
4. Deploy to edge device over USB/network
5. New use case is live

---

## Inference Speed Analysis

### Single Image Inference (INT4 quantized, TensorRT optimized)

For a short structured response (~30-40 tokens, e.g., JSON with 2-3 detected objects):

| Model | Jetson Orin NX 16GB | Jetson AGX Orin 32GB | Jetson AGX Orin 64GB |
|-------|--------------------|--------------------|---------------------|
| Florence-2-base (0.23B) | ~100-150ms | ~60-100ms | ~50-80ms |
| Qwen2.5-VL-3B INT4 | ~0.6-0.8s | ~0.45-0.55s | ~0.4-0.5s |
| Qwen2.5-VL-7B INT4 | ~1.5-1.9s | ~0.9-1.1s | ~0.75-0.95s |
| InternVL3-8B INT4 | ~1.8-2.2s | ~1.0-1.3s | ~0.8-1.0s |

### Video / Continuous Processing (sustained FPS)

| Model | Orin NX 16GB | AGX Orin 32GB | AGX Orin 64GB |
|-------|-------------|-------------|--------------|
| Florence-2-base | 5-10 FPS | 10-15 FPS | 12-20 FPS |
| YOLO-nano (for comparison) | 30+ FPS | 30+ FPS | 30+ FPS |
| Qwen2.5-VL-3B INT4 | ~1.2-1.5 FPS | ~1.8-2.2 FPS | ~2-2.5 FPS |
| Qwen2.5-VL-7B INT4 | ~0.5-0.7 FPS | ~0.9-1.1 FPS | ~1-1.3 FPS |

### Can You Hit 1 FPS with Qwen2.5-VL?

| Model | Hardware Needed | Achievable? |
|-------|----------------|-------------|
| Qwen2.5-VL-3B INT4 | Jetson Orin NX 16GB (~$600) | Yes, ~1.2-1.5 FPS |
| Qwen2.5-VL-7B INT4 | Jetson AGX Orin 32GB (~$1000) | Yes, ~1 FPS with TensorRT + short output |
| Qwen2.5-VL-7B INT4 | Jetson AGX Orin 64GB (~$2000) | Yes, ~1.3 FPS comfortably |

Key optimizations for hitting 1 FPS:
- Fine-tune LoRA to output minimal JSON (fewer tokens = faster)
- TensorRT optimization for Jetson GPU architecture (30-50% speedup)
- Keep max_tokens low (50-100 instead of 512)

---

## Hardware Comparison for Edge Deployment

| Device | Memory | AI TOPS | Price | Best Model Fit | 1 FPS VLM? |
|--------|--------|---------|-------|---------------|------------|
| Jetson Nano | 4 GB shared | 21 | ~$150 | Florence-2 only | No |
| Jetson Orin Nano Super | 8 GB shared | 67 | ~$249 | Florence-2, SmolVLM, 3B models (tight) | No |
| **Jetson Orin NX 16GB** | **16 GB shared** | **100** | **~$600** | **Qwen2.5-VL-3B comfortably, 7B tight** | **Yes with 3B** |
| Jetson AGX Orin 32GB | 32 GB shared | 200 | ~$1000 | Qwen2.5-VL-7B comfortably | Yes with 7B |
| Jetson AGX Orin 64GB | 64 GB shared | 275 | ~$2000 | Any model, room for multi-model | Yes, 1.3 FPS |

---

## Recommended Architecture

### Option A: Best Quality (AGX Orin 32GB, ~$1000)

```
Camera → Qwen2.5-VL-7B INT4 (with task-specific LoRA adapter)
           ↓
         Structured JSON output
           ↓
         Post-processing (pixel→mm, counting, statistics)
           ↓
         Result display / PLC / database
```

- 1 FPS continuous processing
- Best reasoning quality for complex tasks (granulometry, defect classification)
- Single model handles all tasks via LoRA swap
- ~$1 per new use case (cloud fine-tuning cost)

### Option B: Best Value (Orin NX 16GB, ~$600)

```
Camera → Qwen2.5-VL-3B INT4 (with task-specific LoRA adapter)
           ↓
         Structured JSON output
           ↓
         Post-processing
           ↓
         Result display / PLC / database
```

- 1.2-1.5 FPS continuous processing
- Good quality — 3B with LoRA fine-tuning on specific task approaches 7B base quality
- Same LoRA swap workflow
- Cheaper hardware

### Option C: Maximum Speed + Quality (AGX Orin 32/64GB)

For use cases needing both real-time video AND deep analysis:

```
Camera → YOLO-nano (30 FPS, detection + tracking)
           ↓ triggers on events
         Qwen2.5-VL-7B INT4 (detailed analysis on selected frames)
           ↓
         Combined output
```

- YOLO handles continuous monitoring at 30 FPS
- VLM triggered for classification, measurement, defect analysis
- Both models fit in memory simultaneously
- Best of both worlds but more complex to build

---

## Final Recommendation

**Model: Qwen2.5-VL-7B** with INT4 quantization and LoRA adapters per use case.

**Why:**
- Best quality among models that fit on edge hardware
- Natural language flexibility — any task describable in words
- LoRA fine-tuning with 10-20 images works because the model already understands visual concepts; you're only teaching it your output format
- Apache 2.0 license — full commercial use
- Largest community and ecosystem for fine-tuning resources
- Native video support

**Hardware: Jetson AGX Orin 32GB** (~$1000) for production deployments.
- Comfortably runs 7B INT4 at 1 FPS
- Room for future model upgrades
- TensorRT support for maximum optimization

**If budget is tight: Qwen2.5-VL-3B** on **Jetson Orin NX 16GB** (~$600).
- 90% of the quality at 60% of the cost
- 1.2-1.5 FPS
- Fine-tuned 3B on a specific task can match base 7B performance

**Validation plan using Azure credits ($3,500):**
1. Deploy Qwen2.5-VL-7B on Azure GPU VM (~$0.53/hr)
2. Test with actual industrial images from your use cases
3. LoRA fine-tune with 10-20 examples per task
4. Quantize to INT4, benchmark quality loss
5. Validate structured output reliability
6. Total cost: ~$20-50, leaving $3,400+ for further experimentation

---

## Appendix: Quantization Explained

| | Before Quantization (FP16) | After Quantization (INT4) |
|---|---|---|
| What it is | Original model, full 16-bit precision | Compressed model, 4-bit weights |
| Model size (7B) | ~14 GB | ~4.5 GB |
| VRAM needed | ~16 GB | ~6-8 GB (model + overhead) |
| Speed | Baseline | ~2x faster (less data to move through memory) |
| Quality | Baseline (100%) | ~97-98% (2-3% loss on benchmarks) |
| Edge deployable? | No — too large | Yes |

Quantization makes models both smaller AND faster. The small quality loss is negligible for industrial tasks, especially after LoRA fine-tuning compensates for it.

## Appendix: VRAM Breakdown (Why Models Need More Than Just Weight Size)

For Qwen2.5-VL-7B INT4 on Jetson AGX Orin 32GB:

| Component | Memory |
|-----------|--------|
| Model weights (INT4) | ~4.5 GB |
| Vision encoder (image processing) | ~0.5-1 GB |
| KV cache (token generation) | ~0.5-1 GB |
| Activations (forward pass) | ~0.5-1 GB |
| CUDA/TensorRT runtime | ~0.5 GB |
| LoRA adapter | ~0.05 GB |
| **Total** | **~6.5-8 GB** |
| **Available on AGX Orin 32GB** | **32 GB** |
| **Headroom** | **~24 GB (room for YOLO, app code, OS)** |
