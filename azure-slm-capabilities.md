# Azure SLM Project — Capabilities & Approach

Budget: ~$3,500 credits (expires ~May 1, 2026)
Current monthly burn: ~$115–135/mo (from existing resources)
Effective available budget: ~$3,350 after existing costs for the remaining ~3 weeks

---

## Recommended Model: Microsoft Phi-4 Family

Azure has first-party support for the Phi model family — Microsoft's own SLMs. These are the most natural fit on Azure because they get managed compute, serverless APIs, and native fine-tuning support. Key options:

| Model | Parameters | Context | Strengths |
|-------|-----------|---------|-----------|
| Phi-4-mini | ~3.8B | 128K | Cheapest, great for edge/local, strong reasoning |
| Phi-4 | ~14B | 128K | Best quality SLM, rivals Llama 70B on reasoning |
| Phi-3.5-MoE | ~42B (MoE) | 128K | Mixture-of-experts, high quality, efficient |
| Phi-4-multimodal | ~5.6B | 128K | Text + image + audio |

---

## Requirement 1: Run & Inference an SLM in the Cloud

You have two options on Azure:

### Option A: Serverless API (Models-as-a-Service) — Recommended
- Deploy via Azure AI Foundry with zero infrastructure management
- Pay-per-token, no idle costs
- Pricing (Phi-4-mini): $0.000075/1K input tokens, $0.0003/1K output tokens
- Pricing (Phi-4): $0.000125/1K input tokens, $0.0005/1K output tokens
- Setup: Azure AI Foundry portal → Model Catalog → Deploy as Serverless API
- Cost estimate: Even 10M tokens/day ≈ $1–5/day

### Option B: Managed Compute Endpoint (self-hosted)
- Deploy on a GPU VM managed by Azure ML
- You pay for the VM while it's running
- Cheapest GPU: Standard_NC4as_T4_v3 (1x T4 GPU, 16GB VRAM) at ~$0.53/hr (~$380/mo 24/7)
- Better for high-throughput or latency-sensitive workloads
- Can scale to zero when not in use

**Recommendation:** Start with Serverless API. It's the cheapest path for experimentation and moderate usage. Switch to managed compute only if you need sustained high throughput.

---

## Requirement 2: Run & Inference an SLM Locally

Azure doesn't run on your local machine, but the Phi models are open-weight and designed for local execution:

### Tools for Local Inference
- **Ollama**: `ollama run phi4-mini` — simplest option, runs on CPU or GPU
- **llama.cpp**: C++ inference engine, supports GGUF quantized models
- **ONNX Runtime**: Microsoft's own runtime, optimized for Phi models (DirectML for Windows GPU)
- **AI Toolkit for VS Code**: Microsoft extension that lets you download, run, and fine-tune Phi models locally

### Hardware Requirements (Phi-4-mini, 3.8B params)
- FP16: ~8 GB VRAM (any modern GPU with 8GB+)
- INT4 quantized: ~2–3 GB VRAM (runs on integrated GPUs or even CPU-only)
- Phi-4 (14B): ~28 GB FP16, ~8 GB INT4

### Azure's Role
- Use Azure AI Foundry Model Catalog to download the model weights
- Models are available on HuggingFace under Microsoft's org as well
- ONNX-optimized versions are published by Microsoft specifically for local/edge deployment

---

## Requirement 3: Benchmark with Industry Standard Datasets

### Standard Benchmarks You Can Run
| Benchmark | What It Measures |
|-----------|-----------------|
| MMLU | Multitask language understanding (57 subjects) |
| MT-Bench | Multi-turn conversation quality |
| HumanEval | Code generation |
| GSM8K | Grade school math reasoning |
| ARC-Challenge | Science reasoning |
| HellaSwag | Commonsense reasoning |
| TruthfulQA | Factual accuracy |
| MBPP | Python programming |

### How to Run Benchmarks

- **lm-evaluation-harness** (by EleutherAI): The industry standard tool. Run locally or on an Azure GPU VM.
  ```bash
  pip install lm-eval
  lm_eval --model hf --model_args pretrained=microsoft/Phi-4-mini-instruct --tasks mmlu,hellaswag,arc_challenge,gsm8k --batch_size 8
  ```
- **Azure AI Foundry Evaluation**: Built-in evaluation in Azure AI Foundry for safety, groundedness, coherence, and custom metrics.
- Run on an Azure GPU VM (NC4as_T4_v3 at $0.53/hr) — a full benchmark suite takes a few hours = ~$2–5.

### Comparison Targets
| Model | Params | MMLU | HumanEval | GSM8K |
|-------|--------|------|-----------|-------|
| Phi-4-mini | 3.8B | ~70 | ~65 | ~80 |
| Phi-4 | 14B | ~80 | ~75 | ~89 |
| Llama-3.1-8B | 8B | ~66 | ~62 | ~77 |
| Gemma-2-9B | 9B | ~71 | ~54 | ~76 |
| Mistral-7B | 7B | ~63 | ~30 | ~52 |
| GPT-3.5-turbo | ~175B | ~70 | ~48 | ~57 |

> Phi-4 (14B) punches well above its weight class, competing with models 5x its size.

---

## Requirement 4: Quantize the Model

Quantization reduces model precision (FP16 → INT8 → INT4) to shrink size and run on weaker hardware.

### Quantization Options

| Method | Tool | Output Format | Target |
|--------|------|---------------|--------|
| GPTQ | AutoGPTQ | Safetensors | GPU inference |
| AWQ | AutoAWQ | Safetensors | GPU inference |
| GGUF | llama.cpp | GGUF | CPU + GPU (Ollama, llama.cpp) |
| ONNX INT4 | Olive (Microsoft) | ONNX | Windows/Edge/Mobile (DirectML) |

### Where to Quantize
- **Locally**: If you have a GPU with 8GB+ VRAM, you can quantize Phi-4-mini locally in ~30 min
- **Azure GPU VM**: Spin up an NC4as_T4_v3 ($0.53/hr), quantize, download the model, shut down. Cost: ~$1–2
- **Pre-quantized models**: Microsoft and the community publish GGUF/GPTQ/AWQ versions on HuggingFace — you may not need to quantize at all

### Phi-4-mini Quantized Sizes (approximate)
| Precision | Model Size | VRAM Needed | Quality Loss |
|-----------|-----------|-------------|--------------|
| FP16 | ~7.6 GB | ~8 GB | None (baseline) |
| INT8 | ~3.8 GB | ~4 GB | Minimal |
| INT4 (Q4_K_M) | ~2.2 GB | ~2.5 GB | Small (~1-2% on benchmarks) |
| INT4 (Q4_0) | ~2.0 GB | ~2.2 GB | Moderate (~3-5%) |

### Microsoft Olive (Recommended for Azure/Windows)
Microsoft's own optimization toolkit, designed for Phi models:
```bash
pip install olive-ai
olive quantize --model microsoft/Phi-4-mini-instruct --precision int4 --output ./phi4-mini-int4
```
Outputs ONNX format optimized for Windows/DirectML/edge devices.

---

## Requirement 5: Fine-Tune the Model with a Custom Dataset

### Option A: Azure AI Foundry Serverless Fine-Tuning — Recommended
- No GPU provisioning needed — Azure manages the compute
- Pricing for Phi models:
  - Training: $0.003 per 1K tokens
  - Hosting fine-tuned model: $0.80/hr (only while deployed)
  - Inference: Same as base model token pricing
- Supports Phi-3-mini, Phi-3-medium, Phi-3.5-mini, Phi-3.5-MoE, Phi-4, Phi-4-mini
- Dataset format: JSONL with `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`
- Cost estimate for a typical fine-tune job:
  - 10K training examples × ~500 tokens each × 3 epochs = 15M tokens
  - Training cost: 15M × $0.003/1K = ~$45
  - Hosting for testing (4 hours): ~$3.20
  - Total: ~$50 per fine-tune run

### Option B: Azure ML Managed Compute Fine-Tuning
- Use Azure ML with a GPU compute cluster
- More control over hyperparameters, training framework (LoRA, QLoRA, full fine-tune)
- Cheapest GPU: NC4as_T4_v3 (1x T4, 16GB VRAM) at $0.53/hr
- Better GPU: NC6s_v3 (1x V100, 16GB VRAM) at $3.06/hr
- QLoRA fine-tune of Phi-4-mini on T4: ~2–4 hours = ~$1–2
- Full fine-tune of Phi-4 (14B) needs A100 (NC24ads_A100_v4) at ~$3.67/hr

### Option C: Fine-Tune Locally
- QLoRA fine-tuning of Phi-4-mini works on a single GPU with 8GB+ VRAM
- Tools: Hugging Face Transformers + PEFT + bitsandbytes
- Free but slower than cloud GPUs

---

## Budget Allocation Suggestion

Given ~$3,350 available and ~3 weeks until expiry:

| Activity | Approach | Est. Cost |
|----------|----------|-----------|
| Cloud inference (experimentation) | Serverless API (Phi-4-mini / Phi-4) | ~$10–50 |
| Benchmarking on Azure | GPU VM (NC4as_T4_v3) for ~6 hrs | ~$3–5 |
| Quantization on Azure | GPU VM for ~2 hrs | ~$1–2 |
| Fine-tuning (serverless) | 3–5 fine-tune runs | ~$150–250 |
| Fine-tuning (managed compute) | GPU VM for ~10 hrs | ~$5–30 |
| Existing resource burn (~3 weeks) | Current infra | ~$80–100 |
| **Total estimated** | | **~$250–440** |

You'd still have ~$3,000+ in credits remaining. The Phi models on serverless are extremely cheap. Even aggressive experimentation won't dent your budget much.

### If You Want to Maximize Credit Usage
- Try fine-tuning Phi-4 (14B) on a larger dataset with more epochs
- Deploy a managed compute endpoint for sustained inference testing
- Run comprehensive benchmarks across multiple models (Phi-4, Llama, Mistral, Gemma)
- Experiment with Phi-4-multimodal if your use case involves images/audio

---

## Quick Start Path

1. Go to [Azure AI Foundry](https://ai.azure.com) → Create a project (uses your existing `ether-project-resource`)
2. Model Catalog → Search "Phi-4-mini" → Deploy as Serverless API
3. Test inference via the playground or REST API
4. Download the model for local inference via Ollama or ONNX Runtime
5. Run benchmarks using lm-evaluation-harness (locally or on Azure GPU VM)
6. Quantize using Olive or grab pre-quantized GGUF from HuggingFace
7. Fine-tune via Azure AI Foundry with your custom JSONL dataset

> Sources: [Azure AI Foundry Model Catalog](https://ai.azure.com/catalog), [Phi Pricing Announcement](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/announcing-new-phi-pricing-empowering-your-business-with-small-language-models/4395112), [Azure VM Pricing](https://instances.vantage.sh/azure). Content was rephrased for compliance with licensing restrictions.
