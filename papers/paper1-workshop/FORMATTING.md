# Formatting Guide — Paper 1

LaTeX templates and formatting requirements for each target venue.

---

## ECCV 2026 Workshop (Springer LNCS)

| Requirement | Detail |
|-------------|--------|
| Template | Springer LNCS (Lecture Notes in Computer Science) |
| LaTeX template | [Overleaf ECCV template](https://www.overleaf.com/latex/templates/template-and-author-guidelines-for-eccv-submission/gycdswmdkkyv) |
| Page limit | Workshops typically accept 4-8 pages (check individual workshop CFP) |
| References | Do NOT count toward page limit |
| Columns | Single-column |
| Font | Computer Modern (LaTeX default), 10pt body text |
| Paper size | US Letter or A4 (template handles both) |
| Submission format | PDF via CMT or OpenReview (workshop-specific) |
| Anonymity | Double-blind for main conference; workshops vary — check CFP |
| Proceedings | Published in Springer LNCS (archival) |
| Main conference page limit | 14 pages + references |

---

## BMVC 2026

| Requirement | Detail |
|-------------|--------|
| Template | BMVC 2026 official template |
| LaTeX template | [Overleaf BMVC 2026](https://ru.overleaf.com/latex/templates/official-template-for-the-british-machine-vision-conference-2026/vrgpzptmywcz) |
| GitHub template | [BritishMachineVisionAssociation/BMVCTemplate](https://github.com/BritishMachineVisionAssociation/BMVCTemplate) |
| Page limit | 9 pages (references excluded) |
| Supplementary | Submitted separately, no page limit |
| Columns | Single-column |
| Anonymity | Double-blind (remove all author identifying information) |
| Submission platform | CMT or OpenReview (check [bmvc2026.bmva.org](https://bmvc2026.bmva.org/)) |
| Proceedings | Published in BMVA proceedings (archival) |

---

## NeurIPS 2026 Workshop

| Requirement | Detail |
|-------------|--------|
| Template | NeurIPS 2026 format |
| LaTeX template | [Overleaf NeurIPS 2026](https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh) |
| Page limit | 4 pages (extended abstract) or 8 pages (full) — workshop-specific |
| References | Do NOT count toward page limit |
| Columns | Single-column |
| Font | Times, 10pt body text |
| Paper size | US Letter |
| Submission format | PDF via OpenReview |
| Anonymity | Usually single-blind for workshops (authors visible) |
| Proceedings | Non-archival (posted on workshop website only) |

**Note**: Use `\usepackage[workshop]{neurips_2026}` option in the template for workshop submissions.

---

## ICCPR 2026

| Requirement | Detail |
|-------------|--------|
| Template | Springer LNCS (same as ECCV) |
| Page limit | Typically 10-12 pages |
| Proceedings | Published in Springer LNCS |
| Submission | Via conference portal at [iccpr.org](http://www.iccpr.org/) |

---

## General Formatting Rules (All Venues)

### Must Do
- Use the LATEST template from the venue website (templates change yearly)
- Embed all fonts in the PDF
- Number all sections, figures, and tables
- Reference every figure/table in the text
- Keep abstract self-contained (no citations, no undefined acronyms)
- Include an ethics/broader impact statement if required by the venue

### Must NOT Do
- Modify margins, font sizes, or column widths — instant desk rejection
- Exceed the page limit (even by half a page)
- Include author names in double-blind submissions
- Use external links to supplementary content (some venues ban this)
- Submit in any format other than PDF

### Figure Guidelines
- Use vector graphics (PDF/SVG) for charts and diagrams, not rasterized PNG
- Ensure figures are readable when printed in grayscale
- Minimum font size in figures: 8pt
- Keep total PDF size under 10 MB (some portals have upload limits)
- Caption every figure with enough detail to understand without reading the text

### Citation Guidelines
- Use the venue's bibliography style (usually `\bibliographystyle{splncs04}` for LNCS, `\bibliographystyle{ieee}` for IEEE)
- Cite all datasets, models, and frameworks used
- Include DOIs or arXiv IDs where available
- Self-citations should not be excessive (reviewers notice)
