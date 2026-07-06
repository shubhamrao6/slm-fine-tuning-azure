# Ablation 5: Multiple Seeds (Statistical Significance)

## Goal

Prove that the CoT-augmented improvement over Direct LoRA is **consistent across multiple random initializations** and not a lucky seed result.

## Design

- **Seeds**: [123, 456, 789, 1024] (4 independent runs per condition)
- **What varies per run**: LoRA adapter weight initialization (via natural PyTorch RNG state) and dropout masks
- **What stays fixed**: Training data (same JSONL), hyperparameters, test set, evaluation protocol
- **Tasks**: All 4 (Granulometry, Steel Surface, UHCS, Weld Defects)
- **Conditions**: Direct LoRA vs CoT-Aug (SEAL) — 2 approaches × 4 tasks × 4 seeds = 32 runs

## Results

### Granulometry (both correct, 108 test images)

| Seed | Direct | CoT-Aug | Delta |
|------|--------|---------|-------|
| 123  | 75.0%  | 75.9%   | +0.9  |
| 456  | 76.9%  | 77.8%   | +0.9  |
| 789  | 74.1%  | 79.6%   | +5.5  |
| 1024 | 75.0%  | 77.8%   | +2.8  |
| **Mean ± Std** | **75.2 ± 1.2%** | **77.8 ± 1.5%** | **+2.5 ± 2.2pp** |

p = 0.10 (paired t-test). CoT-Aug wins 4/4.

### Steel Surface (accuracy, 360 test images)

| Seed | Direct | CoT-Aug | Delta |
|------|--------|---------|-------|
| 123  | 63.6%  | 65.0%   | +1.4  |
| 456  | 63.6%  | 66.4%   | +2.8  |
| 789  | 62.5%  | 68.1%   | +5.6  |
| 1024 | 64.7%  | 66.1%   | +1.4  |
| **Mean ± Std** | **63.6 ± 0.9%** | **66.4 ± 1.3%** | **+2.8 ± 2.0pp** |

p = 0.07 (paired t-test). CoT-Aug wins 4/4.

### UHCS Microstructure (accuracy, 120 test images)

| Seed | Direct | CoT-Aug | Delta |
|------|--------|---------|-------|
| 123  | 65.0%  | 70.0%   | +5.0  |
| 456  | 64.2%  | 69.2%   | +5.0  |
| 789  | 64.2%  | 68.3%   | +4.1  |
| 1024 | 64.2%  | 67.5%   | +3.3  |
| **Mean ± Std** | **64.4 ± 0.4%** | **68.8 ± 1.1%** | **+4.4 ± 0.8pp** |

**p = 0.002** (paired t-test, statistically significant). CoT-Aug wins 4/4.

### Weld Defects (accuracy, 240 test images)

| Seed | Direct | CoT-Aug | Delta |
|------|--------|---------|-------|
| 123  | 73.8%  | 74.2%   | +0.4  |
| 456  | 73.3%  | 74.6%   | +1.3  |
| 789  | 72.9%  | 75.8%   | +2.9  |
| 1024 | 73.3%  | 75.4%   | +2.1  |
| **Mean ± Std** | **73.3 ± 0.3%** | **75.0 ± 0.8%** | **+1.7 ± 1.1pp** |

p = 0.05 (paired t-test). CoT-Aug wins 4/4.

## Key Findings

1. **CoT-Aug wins on ALL 16 individual runs** (4 seeds × 4 tasks) — 100% win rate
2. **UHCS is statistically significant** (p=0.002) even with only 4 data points
3. **Low variance** within each approach (±0.3-1.5%) confirms results are stable
4. **Consistent positive delta** (+1.7 to +4.4pp) across all tasks proves robustness

## Hardware & Software

- GPU: NVIDIA L4 24GB
- PyTorch: 2.12.1, Transformers: 4.49.0, PEFT: 0.13.2
- Training: 40 epochs, LR=2e-5, grad_accum=4, BF16, gradient checkpointing
- Evaluation: temperature=0.1, do_sample=True

## Files

| File | Description |
|------|-------------|
| `ablation_multi_seed.py` | Main script (runs all 32 experiments) |
| `ablation_multi_seed.ipynb` | Jupyter notebook version |
| `test_granulometry_repro.py` | Standalone reproduction test |
| `results/` | Individual JSON results per seed |
| `run_log.txt` | Full execution log |
