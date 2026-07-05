# Ablation: Equal-Budget Control (Direct-4× Duplicated)

## Goal

Prove that CoT-Aug's improvement comes from **reasoning content quality**, not just from having 4× more training examples and optimizer steps.

## Design

| Configuration | Examples | Unique Images | Total Steps (40 epochs) | Content |
|---------------|----------|---------------|-------------------------|---------|
| Direct (baseline) | 18 | 18 | 180 | Image → JSON |
| **Direct-4× (duplicated)** | 72 | 18 | 720 | Image → JSON (repeated 4×) |
| CoT-Aug (our method) | 72 | 18 | 720 | Image → CoT+JSON (diverse) |

Direct-4× and CoT-Aug have the **same number of examples and optimizer steps**. The only difference is the content of the training responses.

## Expected Outcomes

- **If CoT-Aug ≈ Direct-4×**: Improvement was just from more steps/exposure (bad for paper)
- **If CoT-Aug > Direct-4×**: Reasoning content genuinely helps (good — proves CoT quality matters)
- **If Direct-4× < Direct**: Naive duplication hurts via overfitting, making CoT-Aug's diverse augmentation even more valuable

## Tasks

- Granulometry (18 images → 72 duplicated, ~24 min/run)
- Weld Defects (24 images → 96 duplicated, ~15 min/run)

## How to Run

```bash
# Step 1: Create duplicated JSONL files
python create_duplicated_jsonl.py

# Step 2: Train and evaluate
python run_equal_budget.py
```

## Files

| File | Description |
|------|-------------|
| `create_duplicated_jsonl.py` | Creates 4× duplicated JSONL files |
| `run_equal_budget.py` | Trains Direct-4× and evaluates |
| `results/` | Output JSON files |
| `README.md` | This file |
