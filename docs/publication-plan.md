# Publication Plan — CoT Distillation for Industrial VLMs

## Overview

Three papers, each with a distinct contribution, building on each other sequentially.

```
Paper 1 (Workshop)          Paper 2 (Journal)              Paper 3 (Journal, stronger)
─────────────────          ──────────────────              ────────────────────────────
Proof of concept           Cross-domain validation         Scientific depth
Granulometry only          4 use cases, 3 modalities       Ablations + analysis
4-6 pages                  12-16 pages                     14-20 pages
Submit first               Cites Paper 1                   Cites Papers 1 & 2
```

---

## Paper 1: Workshop Paper

**Title**: "Teaching a Small VLM to Grade Concrete: Answer-Conditioned CoT Distillation from Frontier Models"

**Contribution**: Introduces the method — answer-conditioned CoT distillation from GPT-4.1 to Qwen2.5-VL-3B. Single use case (granulometry), 18 training images, 79.6% accuracy from a 3B model that outperforms GPT-4.1 (29.6%) by 50pp.

**Content**:
- Method description (answer-conditioned CoT, LoRA, prompt design)
- Granulometry results (Direct 71.3% vs SEAL 79.6% vs baselines)
- Brief analysis of why CoT helps (contrastive reasoning, generalizable features)
- 4-6 pages

**Status**: Ready to write. APPROACH.md and RESEARCH.md in `task4-fine-tuning/granulometry/` are essentially a draft.

**Target Venues**:

| Venue | Type | Fit | Typical Deadline |
|-------|------|-----|-----------------|
| AAAI Workshop on AI for Manufacturing | Workshop | Excellent | ~October |
| ECCV Workshop on Vision for Industry | Workshop | Excellent | ~June |
| NeurIPS Workshop on Efficient ML | Workshop | Good | ~September |
| ICMECE (Interdisciplinary Conf. on Mechanics, Computers, Electrics) | Conference | Good | Rolling |
| IEEE CASE (Automation Science & Engineering) | Conference | Good | ~February |

**Timeline**: Submit within 2-4 weeks. Workshop papers have fast turnaround (2-4 weeks review).

---

## Paper 2: Journal Paper (Cross-Domain Validation)

**Title**: "Answer-Conditioned CoT Distillation Enables Small VLMs to Outperform Frontier Models on Industrial Vision Tasks"

**Contribution**: Proves the method generalizes across 4 industrial domains and 3 image modalities. The cross-domain consistency is the main selling point — SEAL wins on all 4 tasks, and on weld radiography the 3B model beats GPT-4.1 FS by 10.8pp.

**Content**:
- Extended method description with cross-task design decisions
- 4 use cases with full results:
  - Granulometry: 9 classes, macro photo, 79.6%
  - Steel surface: 6 classes, surface photo, 66.7%
  - UHCS microstructure: 5 classes, microscopy, 68.4%
  - Weld defects: 4 classes, X-ray, 75.8%
- Per-class analysis and confusion patterns
- Discussion of when the method works well vs struggles
- Comparison with frontier models (GPT-4.1, GPT-5)
- 12-16 pages

**Status**: Results complete. Needs writing, figures, and formatting.

**Target Venues**:

| Venue | Type | IF | Fit |
|-------|------|-----|-----|
| Computers in Industry | Journal | 10.0 | Excellent — applied AI for manufacturing |
| Journal of Manufacturing Systems | Journal | 12.1 | Excellent — manufacturing + ML |
| Engineering Applications of AI | Journal | 7.5 | Good — applied AI methods |
| Expert Systems with Applications | Journal | 7.5 | Good — applied ML |
| NDT&E International | Journal | 4.2 | Good — weld defect focus |
| Computational Materials Science | Journal | 3.3 | Good — microstructure focus |

**Timeline**: Submit 2-3 months after Paper 1. Reference Paper 1 as "our preliminary work."

---

## Paper 3: Journal Paper (Ablations + Scientific Depth)

**Title**: "When and Why Does CoT Distillation Help Small VLMs? An Empirical Study Across Industrial Vision Domains"

**Contribution**: Systematic ablation study answering: How many images are needed? How many CoT descriptions? Does model size matter? When does SEAL fail? Statistical significance across multiple runs.

**Content**:
- All ablation experiments (see below)
- Statistical analysis (confidence intervals, significance tests)
- Scaling curves (accuracy vs training images, accuracy vs CoT count)
- Failure analysis (when and why the method struggles)
- Comparison with other distillation methods (logit distillation, feature distillation)
- Recommendations for practitioners
- 14-20 pages

**Status**: Needs ablation experiments (estimated 40-60 GPU hours).

**Target Venues**:

| Venue | Type | IF | Fit |
|-------|------|-----|-----|
| Pattern Recognition | Journal | 7.5 | Excellent — methods paper |
| IEEE Trans. on Industrial Informatics | Journal | 11.7 | Excellent — industrial AI |
| Engineering Applications of AI | Journal | 7.5 | Good — if Paper 2 goes elsewhere |
| CVPR/ICCV Workshop on Vision + Language | Workshop | — | Good — if targeting top venue exposure |

**Timeline**: Submit 4-6 months after Paper 2. Reference both Papers 1 & 2.

---

## Ablation Plan for Paper 3

### Ablation 1: Training Images Per Class

**Question**: How does accuracy scale with the number of training images?

**Setup**: Run on 2 tasks (granulometry + weld defects) with varying images per class.

| Images/class | Total (Granulometry) | Total (Weld) | Direct examples | SEAL examples |
|-------------|---------------------|-------------|----------------|---------------|
| 1 | 9 | 4 | 9 / 4 | 36 / 16 |
| 2 | 18 | 8 | 18 / 8 | 72 / 32 |
| 4 | 36 | 16 | 36 / 16 | 144 / 64 |
| 6 | 54 | 24 | 54 / 24 | 216 / 96 |
| 8 | 72 | 32 | 72 / 32 | 288 / 128 |
| 10 | 90 | 40 | 90 / 40 | 360 / 160 |

**Expected outcome**: Diminishing returns curve. SEAL should show bigger advantage at lower image counts (where data augmentation matters most).

**Estimated time**: ~16 hours (12 training runs × ~80 min each)

### Ablation 2: CoT Descriptions Per Image

**Question**: What's the optimal number of CoT descriptions per training image?

**Setup**: Run on granulometry (18 images) with varying CoT count.

| CoT/image | Direct examples | CoT examples | Total SEAL examples |
|-----------|----------------|-------------|-------------------|
| 0 | 18 | 0 | 18 (= Direct) |
| 1 | 18 | 18 | 36 |
| 2 | 18 | 36 | 54 |
| 3 | 18 | 54 | 72 (current) |
| 5 | 18 | 90 | 108 |
| 8 | 18 | 144 | 162 |

**Expected outcome**: Diminishing returns after 3-5 descriptions. The first few add diversity; later ones are repetitive at t=0.7.

**Estimated time**: ~8 hours (5 training runs × ~90 min each)

### Ablation 3: Data Quantity Control

**Question**: Is SEAL better because of more data or better data?

**Setup**: Compare SEAL (72 examples from 18 images) vs Direct with the same number of examples (72 images, 8 per class). Run on granulometry.

| Method | Images | Examples | Source |
|--------|--------|----------|--------|
| Direct-18 | 18 | 18 | 2 per class |
| SEAL-18 | 18 | 72 | 2 per class × 4 |
| Direct-72 | 72 | 72 | 8 per class |

**Expected outcome**: If SEAL-18 beats Direct-72, it proves the CoT quality matters more than data quantity. This is the key ablation for the paper.

**Estimated time**: ~3 hours (1 additional training run)

### Ablation 4: Model Size

**Question**: Does the method work on larger models?

**Setup**: Run granulometry on Qwen2.5-VL-7B-Instruct (if VRAM allows with 2x V100). Compare base, Direct, SEAL.

**Expected outcome**: Larger model should have higher base accuracy but SEAL should still add value. The gap between Direct and SEAL may shrink (larger models need less help).

**Estimated time**: ~6 hours (2 training runs, slower due to model size). May need QLoRA if BF16 doesn't fit.

### Ablation 5: Statistical Significance

**Question**: Are the results reproducible across random seeds?

**Setup**: Run granulometry and weld defects with 5 different seeds each. Report mean ± std.

| Seed | Direct Acc | SEAL Acc |
|------|-----------|----------|
| 42 | (current) | (current) |
| 123 | ? | ? |
| 456 | ? | ? |
| 789 | ? | ? |
| 1024 | ? | ? |

**Expected outcome**: SEAL should consistently beat Direct. Variance should be moderate (±2-5pp) given the small test sets.

**Estimated time**: ~16 hours (8 additional training runs × ~90 min each)

### Ablation 6: Without Answer-Conditioning

**Question**: How much does answer-conditioning matter?

**Setup**: Let GPT-4.1 classify AND describe (without giving it the correct answer). Use its predictions as training labels. Compare with answer-conditioned SEAL.

**Expected outcome**: Accuracy drops significantly because ~30-70% of GPT-4.1's own classifications are wrong (depending on task). This proves answer-conditioning is essential.

**Estimated time**: ~4 hours (1 CoT generation run + 1 training run)

### Total Ablation Budget

| Ablation | Training Runs | GPU Hours |
|----------|--------------|-----------|
| 1. Images per class | 12 | ~16h |
| 2. CoT per image | 5 | ~8h |
| 3. Data quantity control | 1 | ~3h |
| 4. Model size | 2 | ~6h |
| 5. Statistical significance | 8 | ~16h |
| 6. Without answer-conditioning | 1 | ~4h |
| **Total** | **29** | **~53h** |

At $6.12/hr (2x V100 workbench), total compute cost: ~$325.

---

## Sequencing Summary

```
Month 1:     Write + submit Paper 1 (workshop, granulometry only)
Month 2-3:   Write Paper 2 while waiting for Paper 1 review
Month 3:     Submit Paper 2 (journal, 4 use cases)
Month 4-5:   Run ablations for Paper 3
Month 5-6:   Write + submit Paper 3 (journal, ablations)
```

Each paper cites the previous ones, building a coherent research narrative from proof-of-concept → cross-domain validation → scientific understanding.
