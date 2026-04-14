# Task 5: Quantization and Edge Deployment

Merge the winning LoRA adapter (SEAL CoT distillation) into the base model, quantize to INT4, and deploy on edge-simulating hardware.

## Pipeline

```
Workbench VM (2x V100):
  Step 1: Merge LoRA adapter into base Qwen2.5-VL-3B → standalone merged model
  Step 2: Quantize merged model to INT4 (AWQ or GPTQ)
  Step 3: Quality check — run 9 + 108 test images on both BF16 and INT4
  Step 4: Size / Memory / Speed comparison (BF16 vs INT4)
  Step 5: Export quantized model for edge deployment

Edge Sim VM (1x T4 16GB):
  Step 6: Deploy quantized model
  Step 7: Run 108 test images — verify accuracy holds
  Step 8: FPS benchmark — sustained inference speed
  Step 9: Memory profiling — VRAM usage under load
```

## Input

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- LoRA adapter: `task4-fine-tuning/granulometry/lora_augmented/`
- Task 4 results: 91.7% size, 86.1% grading, 79.6% both (BF16)

## Targets

| Metric | Target |
|--------|--------|
| Quality vs BF16 | <5% degradation on both axes |
| Model size | ~3-4x smaller (BF16 ~7GB → INT4 ~2GB) |
| Inference speed | Faster than BF16 |
| Memory usage | Fits in T4 16GB with headroom |
| Sustained FPS | ≥1 FPS on edge sim |

## Files

| File | Description |
|------|-------------|
| `quantize_and_eval.ipynb` | Workbench notebook: merge, quantize, quality/speed comparison |
| `edge_deploy.ipynb` | Edge VM notebook: deploy, accuracy, FPS benchmark |
| `config.py` | Shared config (symlinked from task4) |

## Hardware

| VM | GPU | VRAM | Cost/hr | Purpose |
|----|-----|------|---------|---------|
| `slm-workbench` | 2x V100 | 32 GB | $6.12 | Merge, quantize, initial eval |
| `slm-edge-sim` | 1x T4 | 16 GB | $0.53 | Edge deployment, FPS benchmark |
