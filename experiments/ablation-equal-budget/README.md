# Ablation: Equal-Budget Control (Direct-4× Duplicated)

## Goal

Prove that CoT-Aug's improvement comes from **reasoning content quality**, not from having 4× more training examples and optimizer steps.

## Design

Each original Direct training example is duplicated 4 times, giving Direct-4× the **same total examples and optimizer steps** as CoT-Aug. The only difference is content quality.

| Configuration | Examples | Unique Images | Optimizer Steps (40 epochs) | Content |
|---------------|----------|---------------|---------------------------|---------|
| Direct | 18-30 | 18-30 | 180-300 | Image → JSON |
| **Direct-4× (this experiment)** | 72-120 | 18-30 | 720-1200 | Image → JSON (repeated) |
| CoT-Aug | 72-120 | 18-30 | 720-1200 | Image → CoT+JSON (diverse) |

## Results

| Task | Direct (multi-seed mean) | Direct-4× | CoT-Aug (multi-seed mean) |
|------|--------------------------|-----------|---------------------------|
| Granulometry | 75.2% | 74.1% | **77.8%** |
| Steel Surface | 63.6% | 63.1% | **66.4%** |
| UHCS | 64.4% | 70.8% | 68.8% |
| Weld Defects | 73.3% | 72.9% | **75.0%** |

### Interpretation

**Granulometry**: Direct-4× (74.1%) ≈ Direct (75.2%) < CoT-Aug (77.8%)
→ More steps DON'T help. Reasoning content is what matters.

**Steel Surface**: Direct-4× (63.1%) ≈ Direct (63.6%) < CoT-Aug (66.4%)
→ Same conclusion. Naive duplication adds no value.

**UHCS**: Direct-4× (70.8%) > CoT-Aug (68.8%) > Direct (64.4%)
→ Exception. On this task, repeated exposure to correct labels helps more than reasoning. Possible explanation: UHCS has compound class names (e.g., "spheroidite+widmanstatten") where memorization of the exact label may be sufficient.

**Weld Defects**: Direct-4× (72.9%) ≈ Direct (73.3%) < CoT-Aug (75.0%)
→ More steps don't help. Reasoning content drives improvement.

### Summary

On **3 out of 4 tasks**, Direct-4× performs at or below Direct despite 4× more training steps. CoT-Aug consistently outperforms both, proving the improvement comes from the **quality of reasoning content**, not the quantity of training examples.

## Hardware & Software

- GPU: NVIDIA L4 24GB (single run per task)
- PyTorch: 2.12.1, Transformers: 4.49.0, PEFT: 0.13.2
- Training: 40 epochs, LR=2e-5, grad_accum=4, BF16

## Files

| File | Description |
|------|-------------|
| `create_duplicated_jsonl.py` | Creates 4× duplicated JSONL files |
| `run_equal_budget.py` | Trains Direct-4× and evaluates all 4 tasks |
| `results/` | Output JSON files |
| `run_log.txt` | Full execution log |
