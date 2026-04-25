# Industrial Use Cases for CoT Distillation Research

## Overview

Extending the granulometry CoT distillation method to 4-5 industrial vision tasks.
Each use case follows the same pattern: domain-specific visual classification where expert knowledge is needed, few labeled images are available, and no model was pre-trained on the task.

---

## Use Case 1: Concrete Aggregate Granulometry (DONE)

**Status**: Complete — 91.7% size, 86.1% grading, 79.6% both correct

| Property | Value |
|----------|-------|
| Dataset | Coenen et al. "Learning to Sieve" (2022) |
| Classes | 9 (3 sizes × 3 gradings, DIN 1045) |
| Images | 899 total (108 test, 791 train, used 18 for training) |
| Domain knowledge | DIN 1045 grading curves, GSD, particle size distribution |
| Paper | [arxiv 2204.03333](https://arxiv.org/abs/2204.03333) |

---

## Use Case 2: Steel Surface Defect Classification

**Dataset: NEU Surface Defect Database (NEU-CLS)**

| Property | Value |
|----------|-------|
| Source | Northeastern University, China |
| Classes | 6: crazing, inclusion, patches, pitted surface, rolled-in scale, scratches |
| Images | 1,800 total (300 per class), 200×200 grayscale |
| Task | Classify the type of surface defect on hot-rolled steel strip |
| Benchmark | CNN-based: 97-99% accuracy with full training data |
| Our target | Match or exceed with 12-18 images (2-3 per class) + CoT distillation |
| Download | [NEU-CLS on Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |
| Paper | Song & Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects" (2013) |

**Why this is good for our method:**
- 6 classes with subtle visual differences (crazing vs scratches, patches vs pitted)
- Domain knowledge needed: what causes each defect type, visual characteristics
- Well-established benchmark — easy to compare
- Small images (200×200) — no VRAM issues

**Benchmark Results (360 test images):**
- Qwen2.5-VL-3B base: 21.7% ZS / 22.8% FS (near random chance of 16.7%)
- GPT-4.1: 46.9% ZS / 91.1% FS (best frontier — SEAL teacher)
- GPT-5: 45.8% ZS / 86.4% FS
- Few-shot reference image is critical (GPT-4.1: 47% → 91% with reference grid)
- Hardest classes: inclusion (2% ZS) and rolled-in_scale (0% ZS) — visually similar

**Fine-Tuning Results (Task 4, 30 training images):**
- Direct LoRA: 63.1% (from 21.7% base — 3× improvement)
- SEAL LoRA: 66.7% (winner — CoT distillation adds +3.6pp)
- Both beat GPT-4.1 zero-shot (46.9%) with just 30 training images
- Inclusion remains hardest (32%) — confused with scratches

**NEW: Steel-VL Dataset (2025)**

| Property | Value |
|----------|-------|
| Source | arxiv 2603.21824 — "Coarse-to-Fine Vision-Language Dataset for Steel Surface Defect Detection" |
| Description | Vision-language dataset specifically designed for VLMs on steel defects |
| Significance | First VL benchmark for steel defects — directly comparable to our approach |
| Paper | [arxiv 2603.21824](https://arxiv.org/abs/2603.21824) |

---

## Use Case 3: Weld Defect Classification (X-ray Radiography)

**Dataset: RIAWELC**

| Property | Value |
|----------|-------|
| Source | University of Valparaíso, Chile |
| Classes | 4: lack of penetration, cracks, porosity, no defect |
| Images | 24,407 radiographic images |
| Task | Classify weld defect type from X-ray radiograph |
| Benchmark | VGG-based: ~95% accuracy |
| Our target | Match with 8-12 images (2-3 per class) + CoT distillation |
| Paper | [RIAWELC paper](https://www.researchgate.net/publication/369294451) |

**Dataset: GDXray+ (Welds subset)**

| Property | Value |
|----------|-------|
| Source | Pontificia Universidad Católica de Chile |
| Classes | Multiple defect types in welds and castings |
| Images | 21,100+ X-ray images across 5 categories |
| Task | Defect detection and classification in NDT (Non-Destructive Testing) |
| Download | [GitHub](https://github.com/computervision-xray-testing/GDXray) |
| Paper | [Springer](https://link.springer.com/article/10.1007/s10921-015-0315-7) |

**Why this is good for our method:**
- X-ray images are very different from natural photos — tests VLM generalization
- NDT is a critical industrial application (safety-critical welds in pipelines, pressure vessels)
- Expert knowledge needed: what each defect looks like in radiography, acceptance criteria (AWS D1.1)
- Multiple established benchmarks to compare against

**Benchmark Results (240 test images, 4 classes from RIAWELC):**
- Qwen2.5-VL-3B base: 30.8% ZS / 51.2% FS (random chance = 25%)
- GPT-4.1: 57.5% ZS / 65.0% FS (best practical choice — only model detecting cracks)
- GPT-5: 62.5% ZS / 62.5% FS (faster ZS but 0% on cracks even with few-shot)
- Cracks class is catastrophically hard: 0% for all models ZS, only GPT-4.1 FS reaches 30%
- GPT-4.1 used as SEAL teacher (faster, and the only model that can detect cracks at all)

---

## Use Case 4: Rail Surface Defect Detection

**Dataset: RSDDs (Rail Surface Defect Datasets)**

| Property | Value |
|----------|-------|
| Source | Northeastern University, China (same group as NEU steel) |
| Classes | Type I: heavy rail defects, Type II: light rail defects |
| Images | Type I: 67 images, Type II: 128 images |
| Task | Detect and classify surface defects on railway tracks |
| Download | [GitHub](https://github.com/neu-rail-rsdds/rsdds), [IEEE DataPort](https://ieee-dataport.org/documents/rsdds) |
| Benchmark | YOLOv4-based: 92.68% accuracy |

**Dataset: Rail-5k**

| Property | Value |
|----------|-------|
| Source | Recent comprehensive benchmark |
| Description | 5,000+ images from diverse railway environments |
| Task | Object detection and semantic segmentation of rail defects |
| Significance | Supports both supervised and semi-supervised learning |

**Why this is good for our method:**
- Safety-critical application (rail failures cause derailments)
- Very few labeled images available (67-128) — perfect for few-shot
- Real-world industrial images with varying conditions
- Multiple defect types requiring expert knowledge to distinguish

---

## Use Case 5: Steel Microstructure Phase Classification

**Dataset: Aachen-Heerlen Annotated Steel Microstructure Dataset**

| Property | Value |
|----------|-------|
| Source | RWTH Aachen University + Zuyd University |
| Classes | Martensite-austenite (MA) islands, bainite, ferrite, and other phases |
| Images | Annotated SEM/optical microscopy images |
| Task | Identify and classify microstructural phases in steel |
| Download | Published in Nature Scientific Data |
| Paper | [Nature Scientific Data (2021)](https://www.nature.com/articles/s41597-021-00926-7) |

**Dataset: UHCS (Ultra-High Carbon Steel) Microstructure Dataset**

| Property | Value |
|----------|-------|
| Source | Carnegie Mellon University |
| Classes | Cementite particles, spheroidized matrix, grain boundary carbide, Widmanstätten cementite, denuded zones |
| Task | Segment and classify microstructural features |
| Significance | Openly available, well-documented |
| Paper | [ResearchGate](https://www.researchgate.net/publication/331755624) |

**Why this is perfect for your background:**
- Directly related to your Metallurgy & Materials Engineering degree
- Requires deep domain knowledge (phase transformation, heat treatment, CCT diagrams)
- Microscopy images are very different from macro photos — tests VLM on a new modality
- High industrial value: microstructure determines mechanical properties
- The CoT descriptions would include metallurgical reasoning ("this lamellar structure indicates pearlite formed during slow cooling...")

**Benchmark Results (120 test images, 6 classes from UHCS/NIST):**
- Qwen2.5-VL-3B base: 60.8% ZS / 42.5% FS (ZS inflated by spheroidite majority class bias)
- GPT-4.1: 46.7% ZS / 71.7% FS
- GPT-5: 61.7% ZS / 80.0% FS (best frontier — SEAL teacher for this task)
- Hardest task so far — even GPT-5 FS only 80%. Strong CoT distillation candidate.
- Compound classes (spheroidite+widmanstatten) go from 0% → 93% with few-shot reference

**Fine-Tuning Results (Task 4, 30 training images, 5 classes):**
- Direct LoRA: 67.5% (from 60.8% base)
- SEAL LoRA: 68.4% (winner — approaches GPT-4.1 FS 71.7%)
- Both beat GPT-4.1 zero-shot (46.7%) with just 30 images
- pearlite+widmanstatten dropped (only 5 images in dataset)

---

## Summary: 5 Use Cases

| # | Use Case | Domain | Image Type | Classes | Dataset Size | Domain Knowledge |
|---|----------|--------|-----------|---------|-------------|-----------------|
| 1 | Granulometry | Construction | Macro photo | 9 | 899 | DIN 1045 grading curves |
| 2 | Steel surface defects | Steel manufacturing | Surface photo | 6 | 1,800 | Defect formation mechanisms |
| 3 | Weld defect classification | NDT / Welding | X-ray radiograph | 4 | 24,407 | Radiographic interpretation, AWS D1.1 |
| 4 | Rail surface defects | Railway | Surface photo | 2+ types | 67-128 | Track maintenance standards |
| 5 | Steel microstructure | Metallurgy | Microscopy (SEM/optical) | 4-5 phases | Varies | Phase transformation, CCT diagrams |

### Variation across use cases (for ablation)

| Dimension | Range |
|-----------|-------|
| Classes | 2 (rail) to 9 (granulometry) |
| Image type | Macro photo, X-ray, microscopy |
| Visual similarity | Low (weld defects) to high (microstructure phases) |
| Dataset size | 67 (rail) to 24,407 (weld) |
| Domain expertise | Moderate (surface defects) to deep (microstructure) |

---

## References

1. NEU Surface Defect Database — Song & Yan (2013)
2. Steel-VL Dataset — [arxiv 2603.21824](https://arxiv.org/abs/2603.21824)
3. RIAWELC Weld Defect Dataset — [ResearchGate](https://www.researchgate.net/publication/369294451)
4. GDXray+ — [Springer](https://link.springer.com/article/10.1007/s10921-015-0315-7)
5. RSDDs Rail Defects — [IEEE DataPort](https://ieee-dataport.org/documents/rsdds)
6. Rail-5k — [EmergentMind](https://api.emergentmind.com/topics/rail-5k-dataset)
7. Aachen-Heerlen Steel Microstructure — [Nature Scientific Data (2021)](https://www.nature.com/articles/s41597-021-00926-7)
8. UHCS Microstructure — [ResearchGate](https://www.researchgate.net/publication/331755624)
9. Metallic Surface Defect Datasets — [IEEE DataPort](https://ieee-dataport.org/documents/metallic-surface-defect-datasets)
10. Awesome Surface Defect Dataset — [GitHub](https://github.com/LT1st/awesome-surface-defect-dataset)
