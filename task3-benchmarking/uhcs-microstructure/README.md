# Task 3: UHCS Microstructure Benchmarking

Benchmark Qwen2.5-VL-3B and frontier models (GPT-4.1, GPT-5) on 6-class UHCS microconstituent classification.

## Dataset: UHCS (Ultra-High Carbon Steel)

| Property | Value |
|----------|-------|
| Source | NIST / Carnegie Mellon University |
| Images | 598 labeled (961 total, 363 unlabeled) |
| Size | 645×481 px, RGB, PNG |
| Classes | 6 microconstituent types |
| Split | ~120 test (20% stratified), ~478 train pool |
| Download | [NIST](https://materialsdata.nist.gov/handle/11256/940) |

## Classes

| Class | Count | Description |
|-------|-------|-------------|
| spheroidite | 372 | Globular cementite particles in ferrite matrix |
| network | 101 | Continuous cementite along grain boundaries |
| spheroidite+widmanstatten | 77 | Spheroidized particles + needle-like plates |
| pearlite+spheroidite | 28 | Lamellar pearlite + spheroidized regions |
| pearlite | 15 | Alternating ferrite/cementite lamellae |
| pearlite+widmanstatten | 5 | Lamellar pearlite + Widmanstatten plates |

## Metadata

Each image has associated heat treatment parameters:
- Anneal temperature (700–1100°C)
- Anneal time + unit (minutes or hours)
- Cooling method (Q=quench, FC=furnace cool, AR=air cool)
- Magnification (49x–19641x)

Magnification is included in the prompt (like GSD in granulometry).

## Benchmark Results (120 test images)

| Method | Accuracy | JSON Valid | Time/img |
|--------|----------|-----------|----------|
| Qwen2.5-VL-3B (ZS) | 60.8% | 100% | 2.5s |
| Qwen2.5-VL-3B (FS) | 42.5% | 100% | 3.6s |
| GPT-4.1 (ZS, t=0.7) | 46.7% | 100% | 2.4s |
| GPT-4.1 (FS, t=0.7) | 71.7% | 100% | 3.7s |
| GPT-5 (ZS, t=1) | 61.7% | 100% | 11.5s |
| GPT-5 (FS, t=1) | 80.0% | 100% | 11.1s |
| Random chance | 16.7% | — | — |

### Per-Class Accuracy

| Class | N | Qwen ZS | Qwen FS | GPT-4.1 ZS | GPT-4.1 FS | GPT-5 ZS | GPT-5 FS |
|-------|---|---------|---------|------------|------------|----------|----------|
| spheroidite | 74 | 73% | 51% | 34% | 62% | 46% | 74% |
| network | 20 | 95% | 35% | 100% | 80% | 100% | 95% |
| spheroidite+widmanstatten | 15 | 0% | 13% | 60% | 93% | 93% | 93% |
| pearlite+spheroidite | 5 | 0% | 80% | 40% | 100% | 40% | 80% |
| pearlite | 3 | 0% | 0% | 0% | 67% | 67% | 100% |
| pearlite+widmanstatten | 3 | 0% | 0% | 0% | 100% | 67% | 33% |

### Key Findings

- GPT-5 FS (80.0%) is the best overall — outperforms GPT-4.1 FS (71.7%) on this task
- Qwen ZS (60.8%) surprisingly beats GPT-4.1 ZS (46.7%) — but Qwen is biased toward spheroidite (73% of test set is spheroidite, so predicting it often gets lucky)
- Qwen FS (42.5%) is worse than ZS — the reference grid confuses the small model on this domain
- Network class is easy for all models (80-100%) — the grain boundary web pattern is visually distinctive
- Compound classes (spheroidite+widmanstatten, pearlite+spheroidite) are hard zero-shot but frontier models handle them well few-shot
- Pearlite classes have very few test samples (3-5) — results are noisy but directionally useful
- This is the hardest task so far — even GPT-5 FS only reaches 80%, making it a strong candidate for CoT distillation improvement

## Fine-Tuning Results (Task 4, 5 classes)

| Method | Accuracy | Training Data |
|--------|----------|--------------|
| Direct LoRA | 67.5% | 30 images (6/class) |
| SEAL LoRA | 68.4% | 30 images → 120 augmented examples |

SEAL LoRA approaches GPT-4.1 FS (71.7%) with just 30 training images. pearlite+widmanstatten dropped (only 5 images).

## Files

| File | Description |
|------|-------------|
| `benchmark_uhcs.ipynb` | Qwen2.5-VL-3B benchmark (zero-shot + few-shot) |
| `benchmark_frontier.ipynb` | GPT-4.1 + GPT-5 benchmark |
| `config.py` | Shared config, prompts, class definitions |
| `uhcs_reference_grid.png` | 3×2 reference grid for few-shot |
