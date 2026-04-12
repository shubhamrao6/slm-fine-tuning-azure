# SLM Fine-Tuning on Azure

Evaluate and compare Small Language Models (SLMs) for industrial computer vision and NLP tasks, with LoRA fine-tuning on small datasets, targeting edge deployment on NVIDIA Jetson hardware.

## Models

| Model | Parameters | Modality | Use Case |
|-------|-----------|----------|----------|
| Florence-2-large | 0.77B | Vision | Fast native object detection |
| Qwen2.5-VL-3B | 3B | Vision | Granulometry (particle size detection) |
| Qwen2.5-VL-7B | 7B | Vision | Quality comparison target |
| Phi-4-multimodal | 5.6B | Vision + Text + Audio | Cybersecurity agents, general multimodal |

## Project Structure

```
├── task1-serverless-inference/   # Phi-4 serverless API testing (Done)
├── task2-cloud-vm-inference/     # All 4 models on Azure ML GPU VMs (Done)
├── task3-benchmarking/           # Base model benchmarking on test sets (Done)
├── task4-fine-tuning/            # LoRA fine-tuning (Planned)
├── task5-quantization/           # INT4 quantization + edge sim (Planned)
├── experiments/                  # Early Phi-4 local inference & object detection tests
│   ├── phi4-local-inference/
│   └── phi4-object-detection/
├── datasets/                     # Training/test data (gitignored — too large)
│   └── granulometry/             # 899 images (108 test + 791 train)
└── docs/                         # Research & analysis documents
```

## Azure Resources

| Resource | Purpose | Cost |
|----------|---------|------|
| Subscription | Ether_POC_Subscription | ~$3,500 credits (expires ~May 1, 2026) |
| Workspace | `slm-workspace` (West US) | — |
| GPU VM (main) | `slm-workbench` — Standard_NC12s_v3 (2x V100, 32GB) | $6.12/hr |
| GPU VM (edge sim) | `slm-edge-sim` — Standard_NC4as_T4_v3 (1x T4, 16GB) | $0.53/hr |
| Serverless endpoint | `phi4-mm-serverless` (East US 2) | Pay-per-token |

### Compute Management

```bash
# Stop when not using (saves money)
az ml compute stop --name slm-workbench --resource-group CashAPI --workspace-name slm-workspace
az ml compute stop --name slm-edge-sim --resource-group CashAPI --workspace-name slm-workspace

# Start when needed
az ml compute start --name slm-workbench --resource-group CashAPI --workspace-name slm-workspace
```

## Task Progress

| Task | Status | Details |
|------|--------|---------|
| Task 1: Serverless Inference | Done | Phi-4 deployed, tested via API |
| Task 2: Cloud VM Inference | Done | All 4 models compared on V100 |
| Task 3: Benchmarking | Done | Base model baseline on granulometry test set |
| Task 4: LoRA Fine-Tuning | Planned | Qwen-3B on granulometry, Phi-4 on cybersecurity |
| Task 5: Quantization + Edge | Planned | INT4 quantization, test on T4 (edge sim) |
| Task 6: Industrial Validation | Planned | 100+ image validation run |
| Task 7: LoRA Swap Demo | Planned | Adapter hot-swap proof of concept |
| Task 8: Final Comparison | Planned | Head-to-head scoring |

## Key Findings So Far

- Florence-2: Fastest (0.2–3s per image), reliable native detection
- Qwen2.5-VL-3B: Best VLM balance — good quality, reasonable speed (3–12s), reliable structured output
- Qwen2.5-VL-7B: Slower than 3B with worse results — not worth the extra size
- Phi-4-multimodal: Slowest (12–210s), inconsistent structured output — not suitable for detection

### Task 3 Baseline Results (Qwen2.5-VL-3B, granulometry)

| Metric | Zero-Shot (1500px) | Few-Shot (1400px + ref) |
|--------|-------------------|------------------------|
| JSON validity | 100% | 100% |
| Size accuracy | 36.1% | 36.1% |
| Grading accuracy | 34.3% | 24.1% |
| Both correct | 12.0% | 8.3% |
| Avg inference time | 8.9s | 9.4s |

The base model performs near random chance (33%) on both axes. It understands the concepts when reasoning in natural language but cannot reliably map visual input to correct JSON classifications without fine-tuning. This establishes the baseline that Task 4 (LoRA) should improve.

## Datasets

Datasets are gitignored (too large for git). Transfer separately to the VM.

### Granulometry
- 108 test images + 791 train images (9 classes × 2 samples)
- Ground truth: max particle size (8/16/32mm) and grading (coarse/medium/fine)
- Manifests: `test_manifest.json`, `train_manifest.json`

## Budget

| Category | Est. Cost |
|----------|-----------|
| Tasks 1–8 total | ~$35–90 |
| Existing infra burn (~3 weeks) | ~$80–100 |
| Total estimated | ~$115–190 |
| Remaining credits | ~$3,300+ |

---

## Azure Subscription Cost Estimate

Subscription: Ether_POC_Subscription (fe37b5f6-efa5-43a5-ba04-2d3684b07345)
Date: April 8, 2026

### Summary

| Category | Est. Monthly Cost |
|----------|-------------------|
| VM (chaosai — running 24/7) | ~$70 |
| Managed Disk (Premium SSD 40GB) | ~$10 |
| Cosmos DB (Gremlin provisioned) | ~$24 |
| Cosmos DB (2x Serverless) | ~$0–10 |
| Public IPs (2x static) | ~$8 |
| Storage Accounts (3x) | ~$2–9 |
| Cognitive Services (S0, usage-based) | $0–100+ |
| Everything else (Free/Consumption) | ~$0–5 |
| **Total (idle/low usage)** | **~$115–135/mo** |
| **Total (moderate AI/API usage)** | **~$150–250/mo** |

### Cost Saving Recommendations

1. Deallocate the `chaosai` VM when not in use — saves ~$70/mo
2. Review `demoacccosmo123` (Gremlin/provisioned) — delete or switch to serverless to save ~$24/mo
3. Monitor OpenAI/AI Services usage — S0 resources are pay-per-use
4. Release unused Public IPs — ~$4/mo each

### Detailed Resource Breakdown

#### Virtual Machines
| Resource | Size | Location | Est. Monthly Cost |
|----------|------|----------|-------------------|
| chaosai | Standard_D2s_v3 (2 vCPU, 8 GB RAM) | East US | ~$70/mo |

#### Managed Disks
| Resource | Size | SKU | Est. Monthly Cost |
|----------|------|-----|-------------------|
| chaosai_OsDisk | 40 GB | Premium_LRS (P6) | ~$10/mo |

#### Cosmos DB
| Resource | Kind | Mode | Location | Est. Monthly Cost |
|----------|------|------|----------|-------------------|
| sqlcosmossanta | NoSQL | Serverless | Central India | ~$0–5/mo |
| cashapi-cosmosdb | NoSQL | Serverless | Central India | ~$0–5/mo |
| demoacccosmo123 | Gremlin (Graph) | Provisioned | West US | ~$24+/mo |

#### Cognitive Services / AI
| Resource | Kind | SKU | Est. Monthly Cost |
|----------|------|-----|-------------------|
| demorao | Custom Vision Training | F0 (Free) | $0 |
| demorao-Prediction | Custom Vision Prediction | F0 (Free) | $0 |
| DocumentRecognizerOS | Form Recognizer | F0 (Free) | $0 |
| ReadifyAI-mvp | AI Services | S0 | Pay-per-use |
| ether-openai | OpenAI | S0 | Pay-per-use |
| ether-project-resource | AI Services | S0 | Pay-per-use |

#### App Service Plans
All Free (F1) or Consumption/Dynamic (Y1) — effectively $0.

#### Storage Accounts
| Resource | SKU | Est. Monthly Cost |
|----------|-----|-------------------|
| lumostore5344 | Standard_RAGRS | ~$1–5/mo |
| raokaworkspace5618175407 | Standard_LRS | ~$0.50–2/mo |
| cashapifunctions | Standard_LRS | ~$0.50–2/mo |

> Pricing sourced from [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) and [Holori Azure VM pricing](https://calculator.holori.com/azure/vm/standard-d2s-v3). Content was rephrased for compliance with licensing restrictions.

---

## SLM Capabilities & Approach

### Recommended Models: Microsoft Phi-4 Family

| Model | Parameters | Context | Strengths |
|-------|-----------|---------|-----------|
| Phi-4-mini | ~3.8B | 128K | Cheapest, great for edge/local, strong reasoning |
| Phi-4 | ~14B | 128K | Best quality SLM, rivals Llama 70B on reasoning |
| Phi-3.5-MoE | ~42B (MoE) | 128K | Mixture-of-experts, high quality, efficient |
| Phi-4-multimodal | ~5.6B | 128K | Text + image + audio |

### Cloud Inference Options

**Serverless API (Recommended):** Deploy via Azure AI Foundry, pay-per-token, no idle costs.
- Phi-4-mini: $0.000075/1K input, $0.0003/1K output
- Phi-4: $0.000125/1K input, $0.0005/1K output

**Managed Compute:** GPU VM, pay while running. Cheapest: NC4as_T4_v3 at ~$0.53/hr.

### Local Inference Options
- Ollama: `ollama run phi4-mini`
- llama.cpp: GGUF quantized models
- ONNX Runtime: Microsoft's runtime, optimized for Phi (DirectML for Windows GPU)

### Quantization

| Precision | Model Size (3.8B) | VRAM Needed | Quality Loss |
|-----------|-------------------|-------------|--------------|
| FP16 | ~7.6 GB | ~8 GB | None |
| INT8 | ~3.8 GB | ~4 GB | Minimal |
| INT4 (Q4_K_M) | ~2.2 GB | ~2.5 GB | ~1–2% |

### Fine-Tuning Options

**Serverless (Azure AI Foundry):** $0.003/1K tokens training, ~$50 per run.
**Managed Compute (QLoRA):** NC4as_T4_v3 at $0.53/hr, ~$1–2 per run.
**Local:** Free, needs 8GB+ VRAM GPU.

### Benchmark Targets

| Model | Params | MMLU | HumanEval | GSM8K |
|-------|--------|------|-----------|-------|
| Phi-4-mini | 3.8B | ~70 | ~65 | ~80 |
| Phi-4 | 14B | ~80 | ~75 | ~89 |
| Llama-3.1-8B | 8B | ~66 | ~62 | ~77 |
| Gemma-2-9B | 9B | ~71 | ~54 | ~76 |

> Sources: [Azure AI Foundry Model Catalog](https://ai.azure.com/catalog), [Phi Pricing](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/announcing-new-phi-pricing-empowering-your-business-with-small-language-models/4395112), [Azure VM Pricing](https://instances.vantage.sh/azure). Content was rephrased for compliance with licensing restrictions.
