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

> **Note**: The registered abstract says "2.5 to 8.3pp". The paper body (paper-content.md) correctly reports "0.9 to 8.3pp" which includes the UHCS result. The paper body is authoritative — the abstract was registered before the UHCS correction. The paper PDF abstract should use "0.9 to 8.3pp" (matching paper-content.md).

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

1. ~~Download BMVC 2026 LaTeX template~~
2. ~~Write all 7 sections of paper content (in markdown)~~
3. ~~Create supplementary material~~
4. ~~Collect training times (both hardware configs)~~
5. ~~Verify all accuracy numbers against results JSON files~~
6. ~~Generate sample images for figures~~
7. **Convert paper-content.md to LaTeX using BMVC 2026 template** ← CURRENT
8. Create proper figures (method pipeline diagram, dataset grids, per-class bar charts)
9. Format tables in LaTeX
10. Compile PDF and verify it meets 14-page limit
11. Upload PDF to OpenReview before **May 29, 23:59 AoE**
12. Convert supplementary.md to PDF and upload as .zip (single ZIP file, max 100MB)

## Current Status

| Item | Status |
|------|--------|
| Abstract registered on OpenReview | ✅ Done (Submission #995) |
| Paper content (all sections) | ✅ Done (`paper-content.md`) |
| Supplementary material | ✅ Done (`supplementary.md`) |
| Training times measured (L4) | ✅ Done |
| Training times extracted (V100) | ✅ Done |
| Accuracy numbers verified | ✅ Done (UHCS corrected: 65.8/66.7) |
| Sample images generated | ✅ Done (granulometry, steel, weld working; UHCS needs path fix) |
| LaTeX conversion | ❌ Not started |
| Figures created | ❌ Not started |
| PDF compiled | ❌ Not started |
| PDF uploaded to OpenReview | ❌ Not started |

---

## LaTeX Conversion Plan

### Template: BMVC 2026 Official

- **Repository**: https://github.com/lwpyh/BMVCTemplate2026
- **Overleaf**: https://tr.overleaf.com/latex/templates/official-template-for-the-british-machine-vision-conference-2026/vrgpzptmywcz
- **Key file for review submission**: `bmvc_review.tex`
- **Class file**: `bmvc2k.cls`
- **Bibliography style**: `bmvc2k.bst` (uses `natbib`)

### Required Files for Compilation

| File | Purpose | Source |
|------|---------|--------|
| `bmvc2k.cls` | BMVC document class | Template repo |
| `bmvc2k.bst` | Bibliography style | Template repo |
| `bmvc2k_natbib.sty` | Natbib support for BMVC | Template repo |
| `bmvc_review.tex` | Main paper (to adapt) | Template repo → rewrite with our content |
| `egbib.bib` | Bibliography entries | Rewrite with our 19 references |
| `images/` | Figure directory | Our `paper_figures/` |

### Preamble Setup (Review Mode)

```latex
\documentclass{bmvc2k}

% Paper number from OpenReview registration
\bmvcreviewcopy{995}

\title{Answer-Conditioned Chain-of-Thought Distillation for\\Few-Shot Industrial Vision with Small VLMs}

% ANONYMIZED for review — no author names
% Authors section is handled by \bmvcreviewcopy which replaces with
% "BMVC 2026 Submission #995"

% Useful macros
\def\eg{\emph{e.g}\bmvaOneDot}
\def\Eg{\emph{E.g}\bmvaOneDot}
\def\etal{\emph{et al}\bmvaOneDot}
\def\ie{\emph{i.e}\bmvaOneDot}
```

### Double-Blind Anonymization Checklist

| Item | Action Required |
|------|-----------------|
| Author names | Removed — `\bmvcreviewcopy{995}` handles this |
| Institution names | Removed — not included in review version |
| Acknowledgements | Removed — add only in camera-ready |
| Self-citations | Use third person ("Rao [X]" not "our previous work [X]") |
| GitHub links | Do NOT include links to our repo |
| Azure/GCP details | Anonymize to "cloud GPU instances" if needed |
| Dataset links | Keep external dataset citations (they aren't identifying) |
| Supplementary material | Must also be anonymized |

### Section-to-LaTeX Mapping

| Paper Section | LaTeX Commands |
|---------------|----------------|
| Abstract | `\begin{abstract}...\end{abstract}` |
| 1. Introduction | `\section{Introduction}` |
| 2. Related Work | `\section{Related Work}` with `\subsection{}` for 2.1-2.5 |
| 3. Method | `\section{Method}` with `\subsection{}` for 3.1-3.6 |
| 4. Experimental Setup | `\section{Experimental Setup}` |
| 5. Results | `\section{Results}` |
| 6. Discussion | `\section{Discussion}` |
| 7. Conclusion | `\section{Conclusion}` |
| References | `\bibliography{egbib}` (auto-generated from .bib) |

### Table Formatting (LaTeX)

Tables should use `\begin{table}[t]` with `\caption{}` below. Example:

```latex
\begin{table}[t]
\begin{center}
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
Task & Classes & Train & Base ZS & GPT-4.1 FS & Direct & CoT-Aug \\
\hline\hline
Granulometry & 9 & 18 & 12.0\% & 29.6\% & 71.3\% & \textbf{79.6\%} \\
Steel Surface & 6 & 30 & 21.7\% & 91.1\% & 63.1\% & \textbf{66.7\%} \\
UHCS & 5 & 30 & 60.8\% & 71.7\% & 65.8\% & \textbf{66.7\%} \\
Weld Defects & 4 & 24 & 30.8\% & 65.0\% & 73.3\% & \textbf{75.8\%} \\
\hline
\end{tabular}
\end{center}
\caption{Main results across four industrial classification tasks.}
\label{tab:main_results}
\end{table}
```

### Figures Plan

| Figure # | Content | Format |
|----------|---------|--------|
| 1 | Method pipeline diagram (training flow) | TikZ or included PDF |
| 2 | Concrete aggregate sample grid (9 classes) | PNG grid image |
| 3 | Steel surface defect sample grid (6 classes) | PNG grid image |
| 4 | UHCS microstructure sample grid (5 classes) | PNG grid image |
| 5 | Weld radiograph sample grid (4 classes) | PNG grid image |

Use `\begin{figure}[t]` for single-column, `\begin{figure*}[t]` for full-width figures.

### Bibliography (egbib.bib)

19 references to include. Key entries:

```bibtex
@article{czimmermann2020visual,
  title={Visual-based defect detection and classification approaches for industrial applications: a survey},
  author={Czimmermann, T. and others},
  journal={Sensors},
  volume={20},
  number={5},
  pages={1459},
  year={2020}
}

@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, E.J. and others},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

@inproceedings{hsieh2023distilling,
  title={Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes},
  author={Hsieh, C.Y. and others},
  booktitle={Findings of ACL},
  year={2023}
}
```

### Page Budget Estimate

| Section | Est. Pages |
|---------|-----------|
| Title + Abstract | 0.5 |
| Introduction | 1.5 |
| Related Work | 2.0 |
| Method | 3.0 |
| Experimental Setup | 2.5 |
| Results | 3.0 |
| Discussion | 1.5 |
| Conclusion | 0.5 |
| **Total (excl. refs)** | **~14.5** |

**Risk**: slightly over 14 pages. May need to:
- Compress Related Work (move some to supplementary)
- Reduce per-class tables (move to supplementary, keep only summary in main)
- Tighten Method section (reduce example verbosity)

### Supplementary Material

- Convert `supplementary.md` to PDF (LaTeX or pandoc)
- Include full prompts, code snippets, per-class detailed results
- Package as single ZIP file: `0995_supp.zip`
- Must be anonymized (no author names, no repo links)

---

## Submission Policies (BMVC 2026)

### Hard Requirements (desk-rejection if violated)

| Requirement | Status |
|-------------|--------|
| Uses official BMVC 2026 LaTeX template | ❌ Needs LaTeX conversion |
| Does not exceed 14 pages (excl. references) | ⚠️ Tight — may need compression |
| Fully anonymized (main paper + supplementary) | ❌ Need to verify during conversion |
| No altered margins or formatting | Will use official cls file |
| Line numbers present (review version) | Auto by `\bmvcreviewcopy{995}` |
| Paper ID shown instead of author names | Auto by `\bmvcreviewcopy{995}` |
| No identifying links (GitHub, personal pages) | Must verify |
| Bibliography immediately after text | Standard LaTeX behavior |
| All appendices in supplementary only | ✅ Supplementary is separate |

### Submission Checklist (Final Upload)

- [ ] PDF compiled with `pdflatex` using `bmvc2k.cls`
- [ ] `\bmvcreviewcopy{995}` in preamble
- [ ] No author names anywhere in the PDF
- [ ] No acknowledgements section
- [ ] No links to personal GitHub/websites
- [ ] Self-citations in third person
- [ ] Line numbers visible
- [ ] ≤ 14 pages (check after compilation)
- [ ] Figures readable in grayscale (for monochrome printing)
- [ ] All references in 9pt Times, single-spaced
- [ ] Upload main PDF to OpenReview
- [ ] Upload supplementary ZIP to OpenReview
- [ ] File naming: main paper auto-handled by OpenReview; supplementary as ZIP

### Key Dates Remaining

| Event | Date | Days Left |
|-------|------|-----------|
| **Paper & Supplementary Deadline** | **Friday, 29 May 2026, 23:59 AoE** | **3 days** |
| Review Period Begins | Friday, 5 June 2026 | — |
| Reviews Due | Friday, 26 June 2026 | — |
| Rebuttal Period | 3–10 July 2026 | — |
| Author Notification | Friday, 7 August 2026 | — |

### AoE Time Zone Note

AoE (Anywhere on Earth) = UTC-12. So May 29, 23:59 AoE = May 30, 11:59 UTC.
From IST: this is May 30, 17:29 IST (effectively end of day May 30 for India).

## Files

| File | Description |
|------|-------------|
| `SUBMISSION.md` | This file — submission tracker and LaTeX conversion plan |
| `paper-content.md` | Full paper content in markdown (source for LaTeX conversion) |
| `supplementary.md` | Supplementary material (prompts, code, samples) |
| `paper_figures/` | Generated sample images and timing data |
| `FADE.pdf` | BMVC 2026 call for papers / guidelines reference |

### Files to Create (LaTeX)

| File | Description |
|------|-------------|
| `latex/bmvc2k.cls` | BMVC document class (from template repo) |
| `latex/bmvc2k.bst` | Bibliography style (from template repo) |
| `latex/bmvc2k_natbib.sty` | Natbib support (from template repo) |
| `latex/main.tex` | Our paper in LaTeX (converted from paper-content.md) |
| `latex/references.bib` | Our 19 bibliography entries |
| `latex/figures/` | All figure files (PNG/PDF) |
| `latex/supplementary.tex` | Supplementary in LaTeX (for PDF generation) |
