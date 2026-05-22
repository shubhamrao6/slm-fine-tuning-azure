# Project Summary & Technology Stack

## Project Overview

This project investigates whether a small Vision-Language Model (3B parameters) can be taught to perform expert-level industrial visual inspection through chain-of-thought distillation from frontier models, using only 18-30 labeled training images per task.

The core method — Answer-Conditioned CoT Distillation — works as follows:
1. A frontier model (GPT-4.1) is given an industrial image AND the correct classification
2. It generates a detailed justification explaining WHY the classification is correct based on visual features
3. A small model (Qwen2.5-VL-3B) is fine-tuned on these reasoning-augmented examples via LoRA
4. The small model internalizes the reasoning patterns and applies them to unseen images

This was validated across 4 industrial tasks spanning 3 different image modalities, consistently outperforming direct fine-tuning and, in one case, surpassing the frontier model itself.

---

## Results at a Glance

| Industrial Task | Image Type | Base Model | After SEAL LoRA | GPT-4.1 FS |
|----------------|-----------|------------|-----------------|------------|
| Concrete Aggregate Grading | Macro photo | 12.0% | **79.6%** | 29.6% |
| Steel Surface Defects | Surface photo | 21.7% | **66.7%** | 91.1% |
| Steel Microstructure | Microscopy | 60.8% | **68.4%** | 71.7% |
| Weld Defect Classification | X-ray | 30.8% | **75.8%** | 65.0% |

---

## Technology Stack

### 1. Cloud Platform — Microsoft Azure (Tasks 1–4, credits expired May 2026)

| Service | Resource | Purpose |
|---------|----------|---------|
| Azure Machine Learning | Workspace: `slm-workbench` | Managed ML environment with Jupyter access |
| Azure Compute Instance | Standard_NC12s_v3 (2x V100 16GB) | GPU training and inference |
| Azure OpenAI Service | `ether-openai` (East US 2) | GPT-4.1 deployment for CoT generation and benchmarking |
| Azure OpenAI Service | `ether-project-resource` | GPT-5 deployment for frontier benchmarking |
| Azure Files | Mounted at `/mnt/batch/tasks/shared/LS_root/mounts/clusters/` | Persistent storage for datasets, models, code |
| Azure Resource Group | `CashAPI` | Resource organization |
| Azure Subscription | `fe37b5f6-efa5-43a5-ba04-2d3684b07345` | Billing |

**Why Azure**: Provides integrated GPU compute + OpenAI API access in a single ecosystem. The Azure ML workspace handles environment management, and Azure OpenAI gives direct access to GPT-4.1/GPT-5 without rate limits of the public API.

### 1b. Cloud Platform — Google Cloud (Tasks 5+, active from May 2026)

Migrated after Azure credits expired. Project data transferred via GCP Storage Transfer Service (Azure Files → Azure Blob → GCS).

| Service | Resource | Purpose |
|---------|----------|---------|
| Vertex AI Workbench | `slm-workbench-l4` (asia-southeast1-b) | JupyterLab with GPU, auto-shutdown |
| Compute Engine | g2-standard-12 (1× NVIDIA L4 24GB) | GPU training and inference |
| Cloud Storage | `gs://slm-fine-tuning-transfer-4ffe5e` | Backup of project zip (30 GB) |
| Billing | GFS Cloud Program — $25,000 (expires May 2028) | Google for Startups credits |
| Project | `project-162f6734-044f-424a-9ad` | Resource organization |

**Why GCP**: Credits available ($25K via Google for Startups). L4 GPU at $1.00/hr vs Azure V100 at $6.12/hr. Auto-shutdown prevents idle billing. LoRA on 3B model fits easily in 24 GB VRAM.

**Key differences from Azure setup**:
- Single GPU (L4 24GB) vs dual GPU (2× V100 16GB) — no `max_memory` split needed
- Training is ~2-3× slower but 4× cheaper per hour
- Python environment: `/opt/micromamba/envs/jupyterlab/` (not system Python)
- Workspace path: `/home/jupyter/workspace/slm-fine-tuning-azure/`

### 2. AI Models

| Model | Parameters | Role | Deployment |
|-------|-----------|------|------------|
| **Qwen2.5-VL-3B-Instruct** | 3.8B | Base model for fine-tuning | Local (HuggingFace, loaded in BF16) |
| **GPT-4.1** | Unknown (frontier) | SEAL teacher + benchmark | Azure OpenAI API (t=0.7) |
| **GPT-5** | Unknown (reasoning) | Benchmark comparison | Azure OpenAI API (t=1, locked) |

**Why Qwen2.5-VL-3B**: Small enough to fine-tune on 2x V100 with LoRA, supports interleaved image+text input natively, strong base VLM architecture, 3B sweet spot for edge deployment.

**Why GPT-4.1 as teacher**: 3-4x faster than GPT-5, supports temperature control, better accuracy on most tasks, cheaper per call.

### 3. Machine Learning Framework

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Deep Learning | **PyTorch** 2.x | Tensor operations, autograd, training |
| Model Hub | **HuggingFace Transformers** 4.57+ | Model loading, tokenization, generation |
| Parameter-Efficient FT | **PEFT** 0.14+ | LoRA adapter implementation |
| API Client | **OpenAI Python SDK** | Azure OpenAI API calls |

### 4. LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha (alpha) | 32 |
| Dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down projections |
| Trainable params | ~37M (0.98% of model) |
| Precision | BF16 (full, no quantization) |
| Learning rate | 2e-5 |
| Epochs | 40 |
| Effective batch size | 4 (gradient accumulation) |
| Scheduler | Cosine with 10% warmup |

### 5. Data Processing Libraries

| Library | Purpose |
|---------|---------|
| **Pillow (PIL)** | Image loading, conversion, resizing |
| **NumPy** 1.26.4 | Numerical operations |
| **openpyxl** | Reading UHCS metadata from Excel |
| **Matplotlib** | Loss plots, reference grid generation |
| **rarfile + 7-Zip** | RIAWELC RAR extraction |
| **base64** | Image encoding for API calls |

### 6. Development Tools

| Tool | Purpose |
|------|---------|
| **Git + GitHub** | Version control (github.com/shubhamrao6/slm-fine-tuning-azure) |
| **Jupyter Notebooks** | All training and benchmarking code |
| **Kiro AI IDE** | Local development, code management |
| **Python 3.10** | Runtime on Azure ML |

### 7. Hardware

| Component | Specification |
|-----------|--------------|
| GPU | 2x NVIDIA Tesla V100 16GB PCIe |
| Architecture | Volta (compute capability 7.0) |
| Total VRAM | 32 GB (split: 6 + 15 via max_memory) |
| VM Cost | $6.12/hr |

---

## Datasets

| Dataset | Source | Total Images | Train Used | Test | Classes | Modality | Size |
|---------|--------|-------------|-----------|------|---------|----------|------|
| Granulometry | Coenen et al. 2022 | 899 | 18 | 108 | 9 | Macro photo | 2200x3000 |
| NEU-CLS | Northeastern Univ. | 1,800 | 30 | 360 | 6 | Surface (grayscale) | 200x200 |
| UHCS | NIST/CMU | 598 | 30 | 117 | 5 | Microscopy (RGB) | 645x481 |
| RIAWELC | Univ. Valparaiso | 24,407 | 24 | 240 | 4 | X-ray (grayscale) | 227x227 |

---

## Project Structure

```
slm-fine-tuning-azure/
├── task3-benchmarking/          # Baseline evaluation
│   ├── granulometry/
│   ├── steel-surface/
│   ├── uhcs-microstructure/
│   └── riawelc-weld/
├── task4-fine-tuning/           # LoRA training
│   ├── granulometry/
│   ├── steel-surface/
│   ├── uhcs-microstructure/
│   ├── riawelc-weld/
│   ├── APPROACH.md
│   ├── NOTES.md
│   ├── RESEARCH.md
│   └── README.md
├── task5-quantization/          # Edge deployment (pending)
├── datasets/                    # All datasets (gitignored)
├── docs/                        # Documentation
│   ├── industrial-use-cases.md
│   ├── publication-plan.md
│   └── project-summary-and-tech-stack.md
├── papers/                      # Paper drafts
│   └── paper1-workshop/
└── README.md
```

---

## Task 5: Quantization & Edge Deployment (Blocked — Findings)

Attempted to merge the winning LoRA adapter into the base model and quantize for edge deployment. The LoRA merge succeeded (7.53 GB merged model), but quantization save/load is broken for VLMs in the current tooling ecosystem.

### Tools Attempted

| Tool | Result | Issue |
|------|--------|-------|
| **llm-compressor (GPTQ)** | Quantizes in memory ✓ | `save_pretrained()` fails: `KeyError: visual.patch_embed.proj.weight` |
| **AutoAWQ** | Failed | pyarrow binary incompatibility |
| **AWQ (native)** | Failed | Architecture mismatch for Qwen2.5-VL |
| **GPTQModel** | Failed | Cannot install (build dependency issues) |
| **bitsandbytes NF4** | Works at runtime ✓ | Cannot save quantized model to disk |
| **vLLM** | Loaded BF16 model ✓ | 125s/image (V100 lacks Flash Attention 2, compute cap 7.0 < 8.0) |
| **GGUF (llama.cpp)** | Saved Q8_0 (3.29 GB) ✓ | Image inference: 15 min/image (CPU vision encoding) |
| **Manual safetensors** | Saved packed weights ✓ | Transformers ignores weight_packed/scale/shape on load |

### Key Finding

VLM quantization save/load is not production-ready in the current ecosystem (May 2026). Text-only model quantization works fine — the issue is specifically with vision encoders (patch embeddings, visual projections) that don't follow the standard linear layer pattern expected by quantization tools.

### Planned Resolution

- **Ollama + GGUF**: Ollama has native Qwen2.5-VL support with GPU-accelerated vision encoding. Build llama.cpp with CUDA, quantize Q8_0 → Q4_K_M (~2 GB), deploy via Ollama on edge VM.
- **Wait for tooling**: llm-compressor and AutoAWQ are actively fixing VLM support.

### Additional Tools Used in Task 5

| Tool | Purpose |
|------|---------|
| **llm-compressor** | GPTQ quantization library |
| **AutoAWQ** | AWQ quantization |
| **bitsandbytes** | NF4/INT8 runtime quantization |
| **vLLM** 0.19 | High-throughput inference engine |
| **llama-cpp-python** | GGUF inference |
| **llama.cpp** | GGUF conversion and quantization |
| **safetensors** | Model weight serialization |

---

## Cost Summary

| Category | Details | Cost |
|----------|---------|------|
| GPU Compute | ~60 hours training + eval | ~$367 |
| GPT-4.1 API | ~2000 calls (CoT + benchmarks) | ~$20 |
| GPT-5 API | ~1000 calls (benchmarks) | ~$30 |
| Storage | ~50 GB, 3 months | ~$9 |
| **Total** | | **~$426** |

---

## Key Technical Decisions

| Decision | Chosen | Why |
|----------|--------|-----|
| Model size | 3B | Fits V100 LoRA, viable for edge |
| Fine-tuning | LoRA (BF16) | Best quality within VRAM budget |
| Teacher | GPT-4.1 | Faster, cheaper, temperature control, best on most tasks |
| Training data | Answer-conditioned CoT | Guarantees 100% correct reasoning |
| Prompt design | Match benchmarking exactly | Proven: weak→strong prompts gave +18pp |
| LR | 2e-5 | Lower rates consistently underperformed |
| Images/class | 5-6 | Sweet spot: 3 too few, 8+ saturates |
