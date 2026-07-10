# Answer-Conditioned CoT Distillation for Few-Shot Industrial Vision

Fine-tune a 3B vision-language model (Qwen2.5-VL-3B) for industrial image classification using answer-conditioned chain-of-thought distillation from frontier models. Validated across 4 industrial tasks with only 18-30 labeled images per task.

## Results

### Main Results (mean ± std across 4 random seeds)

| Task | Classes | Train Images | Base ZS | GPT-4.1 FS | Direct LoRA | CoT-Aug |
|------|---------|-------------|---------|------------|-------------|---------|
| Concrete Aggregate Grading | 9 | 18 | 12.0% | 29.6% | 75.2±1.2% | **77.8±1.5%** |
| Steel Surface Defects | 6 | 30 | 21.7% | 91.1% | 63.6±0.9% | **66.4±1.3%** |
| Steel Microstructure (UHCS) | 5 | 30 | 60.8% | 71.7% | 64.4±0.4% | **68.8±1.1%** |
| Weld Defect Classification | 4 | 24 | 30.8% | 65.0% | 73.3±0.3% | **75.0±0.8%** |

CoT-Aug wins on **all 16/16** seed × task combinations (4 seeds × 4 tasks). UHCS achieves p=0.002 (paired t-test).

### Ablation: Equal-Budget Control

| Task | Direct | Direct-4× (same steps) | CoT-Aug |
|------|--------|------------------------|---------|
| Granulometry | 75.2 | 74.1 | **77.8** |
| Steel Surface | 63.6 | 63.1 | **66.4** |
| UHCS | 64.4 | **70.8** | 68.8 |
| Weld Defects | 73.3 | 72.9 | **75.0** |

On 3/4 tasks, duplicating training data 4× does not help. The improvement comes from reasoning quality, not training budget.

### Ablation: Answer-Conditioning Necessity

| Task | Gemini 2.5 Pro Accuracy | Unconditioned CoT | Direct | Conditioned CoT-Aug |
|------|------------------------|-------------------|--------|---------------------|
| Granulometry | 24.1% (ZS) / 27.8% (FS) | 57.4% | 75.2% | **77.8%** |
| Weld | 63.7% (ZS) / 62.1% (FS) | 73.3% | 73.3% | **75.0%** |

Without answer-conditioning, wrong reasoning drops performance 17.8pp below Direct on granulometry. Answer-conditioning is essential when the teacher model is unreliable.

## Method

1. A frontier VLM (GPT-4.1) receives each training image **with the correct label**
2. It generates a justified visual explanation of why the classification is correct
3. A 3B model (Qwen2.5-VL-3B) is fine-tuned on these reasoning-augmented examples via LoRA
4. The frontier model is not classifying — it already knows the answer. It explains what visual features justify it.

Each image produces 4 training pairs: 3 CoT descriptions + 1 direct JSON label.

## Datasets

| Dataset | Source | Modality | Classes | Train | Test |
|---------|--------|----------|---------|-------|------|
| Granulometry | Coenen et al. 2022 | Photography | 9 | 18 | 108 |
| NEU-CLS | Song & Yan 2013 | Photography | 6 | 30 | 360 |
| UHCS | DeCost et al. 2019 | Microscopy | 5 | 30 | 120 |
| RIAWELC | Totino et al. 2022 | X-ray | 4 | 24 | 240 |

Three modalities: visible-light photography, optical/SEM microscopy, X-ray radiography.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-VL-3B-Instruct (BF16) |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q, k, v, o, gate, up, down proj |
| Trainable parameters | 37.2M (0.98% of total) |
| Learning rate | 2e-5 |
| Epochs | 40 |
| Effective batch size | 4 (gradient accumulation) |
| Scheduler | Cosine with 10% warmup |
| Training time | 19 min (Direct, granulometry) to 111 min (CoT-Aug, UHCS) on L4 |

## Project Structure

```
├── task3-benchmarking/              # Base model + frontier model baselines
├── task4-fine-tuning/               # LoRA training (Direct + CoT-Aug)
│   ├── granulometry/
│   ├── steel-surface/
│   ├── uhcs-microstructure/
│   └── riawelc-weld/
├── experiments/
│   ├── ablation-multi-seed/         # 32 training runs across 4 seeds
│   ├── ablation-equal-budget/       # Direct-4× control experiment
│   └── ablation-unconditioned-cot/  # Unconditioned CoT baseline
├── datasets/                        # Image data (gitignored)
├── papers/
│   ├── arxiv/                       # arXiv preprint
│   └── bmvc2026/                    # BMVC 2026 submission
└── docs/                            # Documentation
```

## Hardware

| Phase | Hardware | Cost |
|-------|----------|------|
| Training (Azure) | 2× NVIDIA V100 16GB | $6.12/hr |
| Training (GCP) | 1× NVIDIA L4 24GB | $1.00/hr |
| CoT generation | GPT-4.1 via Azure OpenAI | ~$20 total |
| Unconditioned baseline | Gemini 2.5 Pro via Vertex AI | ~$5 total |

## How to Reproduce

1. Place datasets in `datasets/` (see each task folder for expected structure)
2. Run CoT generation notebooks in `task4-fine-tuning/<task>/` to create training JSONL
3. Run training notebooks to produce LoRA adapters
4. Run evaluation cells to get accuracy on test sets
5. For multi-seed ablation: `python experiments/ablation-multi-seed/ablation_multi_seed.py`

## Citation

Paper under review. Preprint forthcoming on arXiv.

## Author

Shubham Rao — [Entropy AI Research Labs](https://entropyresearch.ai)
