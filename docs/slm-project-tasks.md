# SLM Project — Task Definitions

Models: Florence-2-large (0.77B), Qwen2.5-VL-3B, Qwen2.5-VL-7B, Phi-4-multimodal (5.6B)
Budget: ~$3,500 Azure credits (expires ~May 1, 2026)
Goal: Evaluate and compare models for industrial computer vision and NLP tasks with LoRA fine-tuning on small datasets, targeting edge deployment.

## Use Cases for Fine-Tuning

| Use Case | Model | Modality | Task |
|----------|-------|----------|------|
| Granulometry | Qwen2.5-VL-3B | Vision (CV) | Particle size detection from images |
| Cybersecurity | Phi-4-multimodal | Text (NLP) | Agent creation — discover, exploit, exfil agents |

---

## Task 1: Serverless Inference

Quick API-based testing of all 4 models without managing infrastructure.

### Status: Phi-4 deployed and tested
- Endpoint: `phi4-mm-serverless` in eastus2
- Other models (Qwen, Florence) not available as Azure serverless — tested via HuggingFace or skipped to Task 2

### Deliverables
- Inference scripts in `task1-serverless-inference/`

### Estimated Cost: ~$5-15

---

## Task 2: Cloud Provisioned VM Inference (All 4 Models)

Deploy all models on Azure ML compute instances for controlled testing.

### Status: Complete
- Workspace: `slm-workspace` (West US)
- Workbench: `slm-workbench` — Standard_NC12s_v3 (2x V100, 32GB, $7.96/hr)
- Edge sim: `slm-edge-sim` — Standard_NC4as_T4_v3 (1x T4, 16GB, $0.53/hr)

### Key Results (Object Detection Comparison)

| Image | Florence-2 | Qwen-3B | Qwen-7B | Phi-4 |
|-------|-----------|---------|---------|-------|
| dog | 1 obj, 2.88s | 2 obj, 7.27s | 1 obj, 37.52s | 3 obj, 21.29s |
| dog_cat_house | 2 obj, 0.23s | 2 obj, 6.85s | 2 obj, 59.37s | 3 obj, 13.80s |
| image | 1 obj, 0.20s | 2 obj, 6.92s | 0 obj, 99.14s | 4 obj, 185.42s |
| room | 14 obj, 1.26s | 4 obj, 12.40s | 4 obj, 149.81s | 4 obj, 210.96s |
| street | 1 obj, 0.16s | 1 obj, 3.69s | 1 obj, 36.09s | 3 obj, 12.45s |

### Findings
- Florence-2: Fastest (0.2-3s), reliable native detection
- Qwen2.5-VL-3B: Best VLM — good quality, reasonable speed (3-12s), reliable structured output
- Qwen2.5-VL-7B: Slower than 3B with worse results — not worth the extra size
- Phi-4-multimodal: Slowest (12-210s), inconsistent — not suitable for detection tasks

### Deliverables
- Scripts and notebooks in `task2-cloud-vm-inference/`

### Estimated Cost: ~$10-30

---

## Task 3: Benchmarking (Base Models, Before Fine-Tuning)

Benchmark the base (un-fine-tuned) models on the specific datasets that will be used for fine-tuning. This establishes the baseline performance that fine-tuning should improve upon.

### Status: Done (Granulometry)

### Use Case A: Granulometry — Qwen2.5-VL-3B

**Dataset**: 108 test images, 9 classes (A8/A16/A32/B8/B16/B32/C8/C16/C32)
- Max particle size: 8, 16, or 32mm
- Grading (DIN 1045): coarse (A), medium (B), fine (C) — describes size distribution, not absolute size

**Results**:

| Metric | Zero-Shot (1500px) | Few-Shot (1400px + ref) |
|--------|-------------------|------------------------|
| JSON validity | 100% | 100% |
| Size accuracy | 36.1% | 36.1% |
| Grading accuracy | 34.3% | 24.1% |
| Both correct | 12.0% | 8.3% |
| Avg inference time | 8.9s | 9.4s |

**Key findings**:
- Base model performs at random chance (~33%) on both axes
- Zero-shot biases toward 16mm/medium; few-shot biases toward 32mm/fine
- 100% JSON validity — format works, content is wrong
- Model can reason about the task in natural language but fails at direct JSON classification
- LoRA fine-tuning is necessary to teach the visual-to-classification mapping

### Use Case B: Cybersecurity — Phi-4-multimodal

**Dataset**: Cybersecurity agent dataset (text-based scenarios with expected agent responses)
- Split: train (10-15 examples for Task 4) / test (5-10 examples for benchmarking)
- Task types:
  - Discover agent: Given a network description, identify attack surfaces
  - Exploit agent: Given a vulnerability, suggest exploitation approach
  - Exfil agent: Given access context, plan data exfiltration

**Metrics**:
- Task completion accuracy: % of responses that correctly address the scenario
- Output format compliance: % of responses matching expected structure
- Relevance score: human evaluation (1-5) of response quality
- Inference time per prompt

**Baseline expectation**: Phi-4 has general cybersecurity knowledge but won't follow the specific agent output format without fine-tuning.

### Deliverables
- Benchmark scripts for both use cases
- Baseline metrics table for each model on each dataset
- Test split held out (never used for training)

### Estimated Cost: ~$5-10

---

## Task 4: Fine-Tuning with LoRA

Fine-tune each model on the TRAIN SPLIT of its respective dataset, then re-benchmark on the TEST SPLIT to measure improvement.

### Objective
Prove that LoRA fine-tuning with 10-15 training examples produces measurable improvement on domain-specific tasks.

### Use Case A: Granulometry — Qwen2.5-VL-3B + LoRA

**Training data**: 10-15 labeled granulometry images (train split)
**Training format** (JSONL):
```jsonl
{"image": "particle_001.jpg", "prompt": "Analyze particles and return sizes", "response": "[{\"label\":\"particle\",\"size_mm\":2.3,\"bbox\":[120,80,200,160]}]"}
```

**QLoRA config**:
- Rank: 16, Alpha: 32
- Learning rate: 1e-5
- Epochs: 15-20
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

**After training**: Run test split through fine-tuned model, compare metrics with Task 3 baseline.

### Use Case B: Cybersecurity — Phi-4-multimodal + LoRA

**Training data**: 10-15 cybersecurity agent scenarios (train split)
**Training format** (JSONL):
```jsonl
{"messages": [{"role": "system", "content": "You are a cybersecurity discover agent."}, {"role": "user", "content": "Analyze this network: 192.168.1.0/24 with web server on port 80, SSH on 22"}, {"role": "assistant", "content": "{\"agent\":\"discover\",\"findings\":[{\"host\":\"192.168.1.1\",\"services\":[\"http:80\",\"ssh:22\"],\"risk\":\"high\"}]}"}]}
```

**QLoRA config**: Same as above, adjusted for text-only task.

**After training**: Run test split through fine-tuned model, compare with Task 3 baseline.

### Expected Results

| Metric | Before (Task 3) | After (Task 4) | Expected Improvement |
|--------|-----------------|-----------------|---------------------|
| JSON validity | 100% | ~100% | Already perfect |
| Size accuracy | ~36% | ~80%+ | Model learns visual-to-size mapping |
| Grading accuracy | ~34% | ~80%+ | Model learns distribution patterns |
| Both correct | ~12% | ~70%+ | Combined improvement |

### Deliverables
- Fine-tuning scripts for both models
- LoRA adapter files (~50-100 MB each)
- Before/after comparison table with metrics
- Training loss curves

### Estimated Cost: ~$5-15

---

## Task 5: Quantization and Edge Deployment

Quantize the fine-tuned models to INT4 and test on edge-like hardware.

### Objective
Take the fine-tuned models from Task 4, quantize them, and verify they still perform well on the edge simulation VM (T4, 16GB — matching Jetson Orin NX).

### Pipeline
1. Merge LoRA adapter into base model
2. Quantize to INT4 (GPTQ or AWQ)
3. Deploy on `slm-edge-sim` (T4 VM)
4. Re-run the same test split benchmarks
5. Compare: fine-tuned FP16 (Task 4) vs fine-tuned INT4 (Task 5)

### What to Measure

| Metric | Target |
|--------|--------|
| Quality vs FP16 fine-tuned | <5% degradation |
| Model size reduction | ~3-4x smaller |
| Inference speed on T4 | Faster than FP16 |
| Memory usage | Fits in 16GB with headroom |
| Sustained FPS (granulometry) | ≥1 FPS |

### Deliverables
- Quantized model files (INT4)
- Edge inference scripts
- Performance comparison: FP16 vs INT4 (quality + speed)
- Memory usage report

### Estimated Cost: ~$5-10

---

## Task 6: Industrial Output Validation (Brief)

Run 100+ images through the top model. Measure JSON validity, detection accuracy, failure modes. Go/no-go gate.

## Task 7: LoRA Adapter Swap Demo (Brief)

Demo: one base model, swap between granulometry adapter and a different task adapter. Measure swap time (<1 sec target).

## Task 8: Head-to-Head Final Comparison (Brief)

Final scoring of all approaches with data backing the recommendation.

---

## Execution Order

```
Task 1: Serverless Inference ──────── Done
  ↓
Task 2: Cloud VM Inference ────────── Done
  ↓
Task 3: Benchmarking (base models) ── Done — baseline: ~36% size, ~34% grading
  ↓
Task 4: Fine-Tuning (LoRA) ────────── Train on train split, re-benchmark on test split
  ↓
Task 5: Quantize + Edge Deploy ────── Quantize fine-tuned models, test on T4
  ↓
Task 6: Industrial Validation ─────── Go/no-go
  ↓
Task 7: LoRA Swap Demo ───────────── Product concept proof
  ↓
Task 8: Final Comparison ─────────── Decision
```

## Budget Summary

| Task | Estimated Cost |
|------|---------------|
| Task 1: Serverless inference | ~$5-15 |
| Task 2: Cloud VM | ~$10-30 |
| Task 3: Benchmarking | ~$5-10 |
| Task 4: Fine-tuning | ~$5-15 |
| Task 5: Edge quantization | ~$5-10 |
| Tasks 6-8 | ~$5-10 |
| Existing infra burn | ~$80-100 |
| **Total** | **~$115-190** |
