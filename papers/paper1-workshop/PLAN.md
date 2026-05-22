# Paper 1: Workshop Paper Plan

## Paper Details

**Title**: "Teaching a Small VLM to Grade Concrete: Answer-Conditioned Chain-of-Thought Distillation from Frontier Models"

**Scope**: Granulometry only. Proof of concept for the method.

**Key result**: Qwen2.5-VL-3B fine-tuned on 18 images achieves 79.6% combined accuracy, outperforming GPT-4.1 (29.6%) by 50pp.

---

## Target Venues (Upcoming — as of April 25, 2026)

AAAI-26 (Jan 2026) and IEEE CASE (March 2026 deadline) have already passed. The following are still open or upcoming:

### Option 1: ECCV 2026 Workshop (PRIMARY TARGET)

| Detail | Info |
|--------|------|
| Conference | ECCV 2026, Malmö, Sweden |
| Conference dates | Late September / Early October 2026 |
| Workshop paper deadlines | ~June-July 2026 (workshop-specific, not yet announced) |
| Status | Workshop proposals accepted; individual workshop CFPs expected ~May-June 2026 |
| Where to check | [eccv2024.ecva.net/Conferences/2026/](https://eccv2024.ecva.net/Conferences/2026/) |
| Relevant workshops | Efficient Deep Learning for CV ([ecv-workshop.github.io](https://ecv-workshop.github.io/)), Vision for Industry |

**Action**: Monitor the ECCV 2026 workshops page starting now. Once accepted workshops publish their CFPs (~May-June), identify the best fit and submit.

### Option 2: BMVC 2026 (STRONG BACKUP)

| Detail | Info |
|--------|------|
| Conference | BMVC 2026, Lancaster, UK |
| Conference dates | November 23-26, 2026 |
| Paper deadline | ~July 2026 (BMVC 2025 was May 16; 2026 reviewing deadline is July 17) |
| Format | 9 pages + references, BMVC template |
| Where to check | [bmvc2026.bmva.org](https://bmvc2026.bmva.org/calls/call-for-papers/) |
| Template | [Overleaf BMVC 2026](https://ru.overleaf.com/latex/templates/official-template-for-the-british-machine-vision-conference-2026/vrgpzptmywcz) |

**Note**: BMVC is a full conference (not a workshop), but it's more accessible than ECCV/CVPR main tracks. Good for a first publication.

### Option 3: NeurIPS 2026 Workshop

| Detail | Info |
|--------|------|
| Conference | NeurIPS 2026, December 2026 |
| Workshop dates | December 11-12, 2026 |
| Workshop paper deadlines | ~September-October 2026 (workshop-specific) |
| Status | Workshop list not yet announced; expected ~July-August 2026 |
| Where to check | [neurips.cc/Conferences/2026/](https://neurips.cc/Conferences/2026/) |
| Relevant workshops | Efficient ML, Foundation Models for Science, Deployable AI |

**Action**: Monitor NeurIPS 2026 for workshop announcements starting ~July 2026.

### Option 4: ICCPR 2026

| Detail | Info |
|--------|------|
| Conference | ICCPR 2026, Wuxi, China |
| Conference dates | October 29 - November 1, 2026 |
| Paper deadline | ~July 2026 (early bird registration July 15) |
| Where to check | [iccpr.org](http://www.iccpr.org/) |
| Endorsed by | IAPR (International Association for Pattern Recognition) |

---

## Recommended Submission Strategy

```
May-June 2026:  Write the paper (4 weeks)
June 2026:      ECCV workshop CFPs published → identify best fit → submit
July 2026:      If no ECCV fit, submit to BMVC 2026 (full conference)
Sep-Oct 2026:   If needed, submit to NeurIPS 2026 workshop
```

---

## Paper Outline (4-6 pages for workshop, 9 pages for BMVC)

### Abstract (~150 words)
Small VLMs cannot classify domain-specific industrial images. We propose answer-conditioned CoT distillation: GPT-4.1 generates justified reasoning for training images (given correct answers), and Qwen2.5-VL-3B is fine-tuned on these via LoRA. On DIN 1045 concrete aggregate classification with 18 training images, our 3B model achieves 79.6% combined accuracy — outperforming GPT-4.1 (29.6%) by 50pp.

### 1. Introduction (~0.5-1 page)
- Industrial vision needs domain expertise
- VLMs lack it, fine-tuning data is scarce
- Our contribution: CoT distillation from frontier VLM to small VLM

### 2. Method (~1-1.5 pages)
- Answer-conditioned description generation (GPT-4.1 given correct answer + image)
- LoRA fine-tuning with reasoning-augmented data
- Prompt design (DIN 1045 definitions, GSD, consistent train/eval)

### 3. Experiments (~1.5-2 pages)
- Dataset: Coenen et al. granulometry (108 test, 18 train, 9 classes)
- Baselines: Qwen base ZS/FS, GPT-5, GPT-4.1
- Direct LoRA vs SEAL LoRA
- Results table + per-class breakdown

### 4. Analysis (~0.5-1 page)
- Why CoT helps (contrastive reasoning teaches generalizable features)
- Why answer-conditioning is necessary (GPT-4.1 only 29.6% on its own)
- Qualitative examples of generated descriptions

### 5. Conclusion (~0.25 page)
- Summary + future work (more domains, edge deployment)

### References (~15-20 citations)

---

## Figures Needed

1. **Method diagram**: Image → GPT-4.1 (with correct answer) → Description → LoRA training → Fine-tuned model
2. **Results table**: Base vs GPT-4.1 vs Direct vs SEAL (size, grading, both)
3. **Per-class bar chart**: 9 classes, comparing Direct vs SEAL
4. **Qualitative example**: Training image + GPT-4.1 description + model output

---

## Writing Timeline

| Week | Task |
|------|------|
| 1 | Draft introduction + method sections |
| 2 | Write experiments + results, create figures |
| 3 | Write analysis + conclusion, polish |
| 4 | Internal review, final edits, submit |

---

## Submission Process (Step by Step)

### Before Submission
1. Create an [OpenReview](https://openreview.net/) profile (used by NeurIPS, ECCV, many workshops)
2. Create a [CMT](https://cmt3.research.microsoft.com/) account (used by some IEEE/ECCV workshops)
3. Link your ORCID and institutional affiliation
4. Download the correct LaTeX template from the venue website

### Submission Steps
1. Go to the workshop/conference submission portal (linked from CFP)
2. Upload PDF (compiled from LaTeX, all fonts embedded)
3. Fill metadata: title, abstract, authors, keywords, topic areas
4. Confirm submission before deadline (all deadlines are AOE = UTC-12)

### After Submission
1. Review period: 2-4 weeks (workshops), 2-3 months (conferences)
2. If accepted: revise based on feedback, submit camera-ready
3. Register for the conference (required for paper to appear)
4. Prepare poster or short oral presentation

---

## Files

| File | Description |
|------|-------------|
| `PLAN.md` | This file — paper plan, venues, outline, timeline |
| `FORMATTING.md` | LaTeX templates and formatting requirements per venue |
| `paper.tex` | LaTeX source (to be created) |
| `figures/` | Figures for the paper (to be created) |
