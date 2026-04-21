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

## Files

| File | Description |
|------|-------------|
| `benchmark_uhcs.ipynb` | Qwen2.5-VL-3B benchmark (zero-shot + few-shot) |
| `benchmark_frontier.ipynb` | GPT-4.1 + GPT-5 benchmark |
| `config.py` | Shared config, prompts, class definitions |
| `uhcs_reference_grid.png` | 3×2 reference grid for few-shot |
