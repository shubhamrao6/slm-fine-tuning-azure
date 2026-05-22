# BMVC 2026 Submission

## Submission Details

| Field | Value |
|-------|-------|
| Conference | 37th British Machine Vision Conference (BMVC 2026) |
| Venue | Lancaster, UK |
| Dates | November 23-26, 2026 |
| Submission Number | 995 |
| Submission Portal | [OpenReview](https://openreview.net/forum?id=t2Re8x527K) |
| Status | Abstract registered |

## Key Dates

| Event | Date (23:59 AoE) |
|-------|-------------------|
| Abstract Submission Deadline | Friday, 22 May 2026 (DONE) |
| Paper & Supplementary Submission Deadline | Friday, 29 May 2026 |
| Review Period Begins | Friday, 5 June 2026 |
| Reviews Due | Friday, 26 June 2026 |
| Rebuttal Period | Friday, 3 July - Friday, 10 July 2026 |
| Author Notification | Friday, 7 August 2026 |
| Camera-Ready Deadline | Friday, 28 August 2026 |

## Title

Answer-Conditioned Chain-of-Thought Distillation for Few-Shot Industrial Vision with Small VLMs

## Authors

Shubham Rao

## Keywords

vision-language models, knowledge distillation, chain-of-thought reasoning, industrial inspection, few-shot learning, LoRA fine-tuning, manufacturing AI, low-data learning

## Topics

- Primary: Transfer, low-shot, continual, long-tail learning
- Secondary: Vision and language

## Abstract

Deploying AI-based visual inspection in manufacturing is challenging because production requirements change frequently, new defect types emerge, and large labeled datasets are rarely available on the factory floor. Traditional vision models require thousands of annotated images and lengthy retraining cycles, making them impractical when only a handful of expert-labeled samples exist. We propose answer-conditioned chain-of-thought (CoT) distillation as a method for rapidly adapting small vision-language models (VLMs) to new industrial tasks using minimal labeled data. A frontier VLM is given each training image along with its correct label and generates a justified visual explanation describing why the classification is correct. A 3B-parameter model is then fine-tuned on these reasoning-augmented examples via LoRA. By conditioning on correct answers, we guarantee accurate reasoning in all training data, which is critical because the frontier model's own classification accuracy ranges from 29.6% to 91.1% across our tasks. We validate the method on four industrial classification tasks spanning three image modalities (macro photography, optical microscopy, and X-ray radiography) using only 18 to 30 labeled images per task. Our method consistently outperforms direct fine-tuning by 2.5 to 8.3 percentage points across all four domains. On weld radiograph defect classification, the fine-tuned 3B model outperforms the frontier model by 10.8 percentage points using just 24 training images. On concrete aggregate grading, a task with 9 classes, the model achieves 79.6% accuracy from only 18 images where the base model scores 12.0% and the frontier model scores 29.6%. These results demonstrate that answer-conditioned CoT distillation enables practical industrial deployment of VLMs in data-scarce environments where collecting large datasets is infeasible due to cost, rarity of defects, or rapidly changing inspection criteria.

## Format Requirements

- Template: Official BMVC 2026 LaTeX template
- Page limit: 14 pages (excluding references)
- References: Unlimited
- Supplementary: Optional, single .zip file
- Review: Double-blind (no author/institution names in paper)
- Dual submission: Not allowed (arXiv preprint is fine)

## Submission Statistics (BMVC 2024 reference)

- Total submissions: ~1,200
- Accepted papers: 263 (30 oral + 233 poster)
- Acceptance rate: ~22%

## Next Steps

1. Download BMVC 2026 LaTeX template
2. Write 14-page paper (deadline: May 29)
3. Create figures (method diagram, results tables, per-class analysis)
4. Compile PDF and upload to OpenReview
5. Optional: prepare supplementary material (.zip)
