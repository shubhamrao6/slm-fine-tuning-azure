# Research Paper Feasibility — Answer-Conditioned CoT Distillation for Industrial VLMs

## Proposed Title

"Teaching Small Vision-Language Models to See Like Experts: Answer-Conditioned Chain-of-Thought Distillation for Industrial Image Classification"

Alternative titles:
- "From 18 Images to 80% Accuracy: Frontier-Guided LoRA Fine-Tuning for Domain-Specific VLMs"
- "SEAL-Inspired Chain-of-Thought Distillation from Frontier VLMs to Small VLMs for Industrial Vision"

---

## Our Key Results

| Method | Size Acc | Grading Acc | Both Correct |
|--------|----------|-------------|--------------|
| Qwen2.5-VL-3B base (zero-shot) | 36.1% | 34.3% | 12.0% |
| GPT-4.1 frontier (few-shot) | 62.0% | 59.3% | 29.6% |
| LoRA Direct (18 images → JSON) | 89.8% | 78.7% | 71.3% |
| LoRA + CoT Distillation (18 images → description + JSON) | 91.7% | 86.1% | 79.6% |

Key findings:
- A 3B model fine-tuned on 18 images outperforms a frontier model (GPT-4.1) by 50 percentage points
- Chain-of-thought distillation adds +8.3 points over direct LoRA on combined accuracy
- The frontier model's reasoning descriptions teach domain knowledge the small model can't discover alone
- 100% JSON validity throughout — structured output is reliable

---

## What Makes This Novel

### 1. Answer-Conditioned Chain-of-Thought Generation

Existing CoT distillation asks the teacher to solve the problem and explain. Our approach gives the teacher the correct answer and asks it to justify WHY — ensuring 100% correct reasoning in training data.

This matters because the frontier model itself only achieves 59.3% grading accuracy. If we let it classify and explain, ~40% of training data would contain wrong reasoning. By conditioning on the answer, every description is correct.

No existing paper does this for VLMs.

### 2. VLM-to-VLM CoT Distillation

Most CoT distillation work is LLM→LLM (text only). We distill visual reasoning from a frontier VLM (GPT-4.1) to a small VLM (Qwen2.5-VL-3B). The teacher sees the actual image and describes image-specific visual features — not generic text.

### 3. Industrial Domain with No Pre-existing Training Data

DIN 1045 aggregate grading is a specialized civil engineering task. No vision model was trained on this. The method proves that domain-specific visual classification can be bootstrapped from just 18 labeled images + frontier model reasoning.

### 4. Extremely Low-Data Regime

18 images total (2 per class). Most VLM fine-tuning papers use hundreds or thousands of examples. We show that CoT distillation is especially valuable when data is scarce — the descriptions provide the diversity that more images would normally provide.

---

## Related Work

### Chain-of-Thought Distillation (LLM → LLM)

| Paper | Year | What they do | How we differ |
|-------|------|-------------|---------------|
| [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301) (Hsieh et al.) | 2023 | Distill CoT reasoning from large LLM to small LLM, outperforming fine-tuning with less data | We apply this to VLMs (multimodal), not text-only LLMs. We also condition on the correct answer. |
| [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) (Wei et al.) | 2022 | Show that CoT prompting improves LLM reasoning | We use CoT in training data, not just at inference time |
| [Enhancing Generalization in CoT for Smaller Models](https://arxiv.org/abs/2501.09804) | 2025 | Address memorization in CoT distillation for small LLMs | Text-only; we handle vision + text |

The closest work is "Distilling Step-by-Step" which showed that rationale-augmented training data helps small models outperform large ones. We extend this concept to the multimodal (vision-language) domain.

### Knowledge Distillation for VLMs

| Paper | Year | What they do | How we differ |
|-------|------|-------------|---------------|
| [VLsI: Verbalized Layers-to-Interactions](https://arxiv.org/abs/2412.01822) | 2024 | Layer-wise distillation from large VLM to small VLM using natural language "verbalizers" | They distill internal representations; we distill output reasoning |
| [Knowledge Distillation from VLM for Long-Tail Recognition](https://arxiv.org/abs/2408.16930) | 2024 | Transfer knowledge from VLM teacher to student for rare classes | They use logit distillation; we use text-based reasoning distillation |
| [Online In-Context Distillation](https://arxiv.org/abs/2510.18117) | 2025 | Small VLM collaborates with teacher at inference time | They need the teacher at inference; we bake knowledge into weights |
| [PartDistill: 3D Part Segmentation](https://arxiv.org/abs/2312.04016) | 2023 | VLM teacher makes 2D predictions, student learns 3D segmentation | Different task (segmentation vs classification), different distillation mechanism |

None of these use answer-conditioned CoT generation as the distillation mechanism.

### SEAL and Self-Adapting Models

| Paper | Year | What they do | How we differ |
|-------|------|-------------|---------------|
| [SEAL: Self-Adapting Language Models](https://arxiv.org/abs/2506.10943) (MIT) | 2025 | Models generate their own fine-tuning data ("self-edits") and update weights via LoRA | Text-only LLMs; we adapt for VLMs. We use external editor (GPT-4.1) instead of self-editing. |
| [Reproducing SEAL](https://wtzhang99.github.io/blog/reproducing-seal/) (Zhang) | 2025 | Found that external editors match RL-based self-editing | Validates our approach of using GPT-4.1 as external editor |
| [Search over Self-Edit Strategies](https://arxiv.org/abs/2601.14532) | 2026 | Extended SEAL with model-generated templates | Still text-only; we apply to vision |

Our work is SEAL-inspired but adapted for multimodal: the frontier model generates image-grounded reasoning (not text knowledge), and we condition on correct answers.

### VLMs for Industrial Vision

| Paper | Year | What they do | How we differ |
|-------|------|-------------|---------------|
| [Adapting Vision Foundation Models for Industrial Settings](https://arxiv.org/abs/2406.09637) | 2024 | Self-supervised transfer learning for industrial domain using web-crawled data | They need large datasets; we use 18 images |
| [Adapting CLIP for Few-Shot Manufacturing QC](https://arxiv.org/abs/2501.12596) | 2025 | CLIP for manufacturing defect detection with 50-100 examples | They use CLIP (contrastive); we use generative VLM with CoT distillation. They need 50-100 examples; we use 18. |
| [VLMs for Anomaly Classification](https://arxiv.org/abs/2601.13440) | 2026 | CLIP-based zero-shot anomaly detection in industrial settings | Zero-shot only; we fine-tune with CoT distillation |

No existing work applies CoT distillation to industrial vision classification.

### Granulometry / Concrete Aggregate

| Paper | Year | What they do | How we differ |
|-------|------|-------------|---------------|
| [Learning to Sieve](https://arxiv.org/abs/2204.03333) (Coenen et al.) | 2022 | CNN-based prediction of grading curves from aggregate images | They use CNNs trained on full dataset; we use VLM + LoRA on 18 images. They predict continuous curves; we classify into 9 discrete classes. |

This is the only ML paper on this specific dataset. They used traditional deep learning (CNNs). We are the first to apply VLMs and CoT distillation to granulometry.

---

## Paper Structure (Proposed)

### Abstract
Small VLMs can't classify domain-specific industrial images without fine-tuning. We propose answer-conditioned CoT distillation: a frontier VLM generates justified reasoning for each training image (given the correct answer), and a small VLM is fine-tuned on these reasoning-augmented examples via LoRA. On DIN 1045 concrete aggregate classification with only 18 training images, our 3B model achieves 79.6% combined accuracy — outperforming the frontier model itself (29.6%) and direct LoRA (71.3%).

### 1. Introduction
- Industrial vision needs domain-specific classification
- VLMs are powerful but lack domain knowledge
- Fine-tuning with few images is challenging
- Our contribution: CoT distillation from frontier VLM to small VLM

### 2. Related Work
- CoT distillation (text-only)
- VLM knowledge distillation
- SEAL framework
- Industrial vision with VLMs
- Granulometry

### 3. Method
- Problem formulation: 9-class classification with 18 images
- Answer-conditioned description generation
- LoRA fine-tuning with reasoning-augmented responses
- Consistent prompt design

### 4. Experiments
- Dataset: Coenen et al. granulometry (108 test, 18 train)
- Baselines: Qwen base, GPT-5, GPT-4.1
- Approach A: Direct LoRA
- Approach B: CoT-distilled LoRA
- Ablations: number of images, descriptions per image, epochs

### 5. Results
- Main comparison table
- Per-class analysis
- Training curves
- Qualitative examples (what the model outputs)

### 6. Discussion
- Why CoT distillation helps (reasoning teaches generalizable features)
- Why answer-conditioning is necessary (frontier model's own accuracy is insufficient)
- Limitations (single dataset, single domain)
- Future work (more domains, more models, edge deployment)

### 7. Conclusion

---

## What's Needed for a Strong Paper

### Already Done
- Main results (Direct vs Augmented vs baselines)
- Frontier model benchmarking (GPT-5, GPT-4.1)
- Training data analysis
- Multiple training runs with different configurations

### Still Needed

| Experiment | Purpose | Effort |
|-----------|---------|--------|
| Ablation: 9 images (1 per class) | Show CoT distillation matters more with less data | ~4 hours training |
| Ablation: 36 images (4 per class) | Show scaling behavior | ~6 hours training |
| Ablation: vary descriptions per image (1, 3, 5, 8) | Find optimal augmentation ratio | ~8 hours training |
| Ablation: Direct with 72 examples (same count as augmented) | Control for data quantity vs quality | ~3 hours training |
| Qualitative analysis | Show example outputs with reasoning | 1 hour manual |
| Error analysis | What classes/images fail and why | 2 hours manual |
| Per-class confusion matrices | Detailed accuracy breakdown | Code exists |
| Statistical significance | Multiple runs with different seeds | ~12 hours training |

### Target Venues

| Venue | Type | Fit | Deadline |
|-------|------|-----|----------|
| AAAI Workshop on AI for Manufacturing | Workshop | Excellent — industrial + AI | Usually December |
| ECCV Workshop on Vision for Industry | Workshop | Excellent — vision + industrial | Usually June |
| NeurIPS Workshop on Efficient ML | Workshop | Good — few-shot + distillation | Usually September |
| Automation in Construction (journal) | Journal | Excellent — domain-specific | Rolling |
| Computers in Industry (journal) | Journal | Good — applied AI | Rolling |
| CVPR Industry Track | Conference | Stretch — needs more ablations | Usually November |

---

## References

1. Hsieh et al. "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes" (2023) — [arxiv 2305.02301](https://arxiv.org/abs/2305.02301)
2. Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022) — [arxiv 2201.11903](https://arxiv.org/abs/2201.11903)
3. Zweiger et al. "Self-Adapting Language Models (SEAL)" (2025) — [arxiv 2506.10943](https://arxiv.org/abs/2506.10943)
4. Zhang. "Reproducing SEAL" (2025) — [blog](https://wtzhang99.github.io/blog/reproducing-seal/)
5. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — [arxiv 2106.09685](https://arxiv.org/abs/2106.09685)
6. Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023) — [arxiv 2305.14314](https://arxiv.org/abs/2305.14314)
7. Coenen et al. "Learning to Sieve: Prediction of Grading Curves from Images of Concrete Aggregate" (2022) — [arxiv 2204.03333](https://arxiv.org/abs/2204.03333)
8. Micikevicius et al. "Mixed Precision Training" (2018) — [arxiv 1710.03740](https://arxiv.org/abs/1710.03740)
9. VLsI: "Verbalized Layers-to-Interactions from Large to Small VLMs" (2024) — [arxiv 2412.01822](https://arxiv.org/abs/2412.01822)
10. "Knowledge Distillation from VLM for Long-Tail Visual Recognition" (2024) — [arxiv 2408.16930](https://arxiv.org/abs/2408.16930)
11. "Online In-Context Distillation for Low-Resource VLMs" (2025) — [arxiv 2510.18117](https://arxiv.org/abs/2510.18117)
12. "Adapting Vision Foundation Models for Industrial Settings" (2024) — [arxiv 2406.09637](https://arxiv.org/abs/2406.09637)
13. "Adapting CLIP for Few-Shot Manufacturing Quality Control" (2025) — [arxiv 2501.12596](https://arxiv.org/abs/2501.12596)
14. "Enhancing Generalization in CoT Reasoning for Smaller Models" (2025) — [arxiv 2501.09804](https://arxiv.org/abs/2501.09804)
15. "Improving Chain of Thought Training in Vision Language Models" (2026) — [arxiv 2603.18656](https://arxiv.org/abs/2603.18656)
16. "Reasoning Transfer from LLMs to VLMs via On-Policy Distillation (VOLD)" (2025) — [arxiv 2510.23497](https://arxiv.org/abs/2510.23497)
17. "Latent Chain-of-Thought for Visual Reasoning" (2025) — [arxiv 2510.23925](https://arxiv.org/abs/2510.23925)
18. DIN 1045 — Concrete, reinforced and prestressed concrete structures — [beuth.de](https://www.beuth.de/en/standard/din-1045-2/147411977)
