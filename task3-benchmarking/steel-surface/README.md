# Task 3: Steel Surface Defect Benchmarking

Benchmark Qwen2.5-VL-3B and frontier models (GPT-4.1) on the NEU Steel Surface Defect dataset.
Same methodology as granulometry: zero-shot, few-shot, and frontier model comparison.

## Dataset: NEU-CLS

| Property | Value |
|----------|-------|
| Source | Northeastern University, China |
| Classes | 6: crazing (Cr), inclusion (In), patches (Pa), pitted surface (PS), rolled-in scale (RS), scratches (Sc) |
| Images | 1,800 total (300 per class) |
| Size | 200×200 grayscale |
| Split | TBD (e.g., 12 train, 108 test per class) |
| Download | [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |

## Defect Definitions

| Class | Code | Visual Description |
|-------|------|-------------------|
| Crazing | Cr | Network of fine surface cracks, web-like pattern |
| Inclusion | In | Dark spots or streaks embedded in the surface, foreign material |
| Patches | Pa | Irregular lighter/darker areas on the surface, uneven texture |
| Pitted Surface | PS | Small holes or depressions scattered across the surface |
| Rolled-in Scale | RS | Oxide scale pressed into the surface during rolling, elongated marks |
| Scratches | Sc | Linear marks/grooves on the surface, directional damage |

## Reference Image

`Sample-images-in-the-NEU-CLS-dataset.png` — shows examples of all 6 classes for few-shot prompting.

## Files

| File | Description |
|------|-------------|
| `benchmark_steel.ipynb` | Main benchmark notebook (Qwen + frontier) |
| `config.py` | Shared config and prompts |
| `Sample-images-in-the-NEU-CLS-dataset.png` | Reference image for few-shot |
