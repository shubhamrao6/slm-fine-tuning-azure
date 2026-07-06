# arXiv Submission

## Paper
Answer-Conditioned Chain-of-Thought Distillation for Few-Shot Industrial Vision with Small VLMs

## Author
Shubham Rao

## Category
cs.CV (Computer Vision and Pattern Recognition)
Secondary: cs.AI, cs.LG

## Submission Checklist
- [ ] Main tex file compiles without errors
- [ ] All figures included in images/ folder
- [ ] .bbl file included (pre-compiled bibliography)
- [ ] No hidden files or auxiliary files (.aux, .log, etc.)
- [ ] Author name and affiliation included (not anonymous)
- [ ] No double-spacing or referee mode
- [ ] License: CC BY 4.0 (recommended for arXiv)

## How to Submit
1. Go to https://arxiv.org/submit
2. Select category: cs.CV
3. Upload all files as a .zip or .tar.gz (main.tex, references.bib, bmvc2k.cls, bmvc2k_natbib.sty, images/, main.bbl)
4. Fill in metadata (title, abstract, authors, categories)
5. Preview the PDF
6. Submit

## Files to Include in Upload
- main.tex (primary source)
- references.bib
- main.bbl (pre-compiled bibliography)
- bmvc2k.cls
- bmvc2k_natbib.sty
- bmvc2k.bst
- images/ (all figure files)

## Files NOT to Include
- .aux, .log, .out, .toc, .synctex files
- .pdf output files
- Hidden files (starting with .)

## Key Differences from BMVC Submission
- Author name included (Shubham Rao)
- \bmvcreviewcopy{995} removed
- Ablation results added (multi-seed, equal-budget, unconditioned CoT)
- Updated abstract with multi-seed findings
- No page limit (but paper remains concise)

## arXiv Requirements
- LaTeX source must compile on arXiv's TeX Live 2025
- Figures must be PDF, PNG, or JPG (no EPS conversion on the fly)
- Include .bbl file for bibliography
- Do not use \pdfoutput or driver-specific options
- No JavaScript in PDFs
- Total upload size limit: 50MB (ours is well under)
