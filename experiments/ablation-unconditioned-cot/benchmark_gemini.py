"""
Benchmark Gemini 2.5 Pro accuracy on Granulometry and Weld test sets.
Uses zero-shot classification (same prompt as evaluation).
"""
import os
import json
import re
import time
from pathlib import Path
from google import genai
from google.genai import types

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

GCP_PROJECT = "project-162f6734-044f-424a-9ad"
GCP_LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-pro"

# Granulometry config
GRANULOMETRY_TEST_MANIFEST = REPO_ROOT / "datasets" / "granulometry" / "test_manifest.json"
GRANULOMETRY_TEST_DIR = REPO_ROOT / "datasets" / "granulometry" / "test"
ORIGINAL_GSD = 8.0
MAX_DIM = 800

# Weld config
WELD_TEST_DIR = REPO_ROOT / "datasets" / "riawelc" / "testing"
WELD_CLASSES = ["lack_of_penetration", "porosity", "cracks", "no_defect"]
WELD_SAMPLE_PER_CLASS = 60

GRANULOMETRY_REF_IMAGE = REPO_ROOT / "task3-benchmarking" / "granulometry" / "examples_classification_data.png"
WELD_REF_IMAGE = REPO_ROOT / "task3-benchmarking" / "riawelc-weld" / "riawelc_reference_grid.png"

# Few-shot prompts (from original benchmark notebooks)
GRAN_FS_REF_PROMPT = """First image is a reference chart: 3x3 grid of concrete aggregate photographs.

COLUMNS (left to right) = max particle size: 8mm | 16mm | 32mm
ROWS (top to bottom) = grading curve: A (coarse) | B (medium) | C (fine)

This follows DIN 1045 standard grading curves. Grading describes the particle size DISTRIBUTION shape, independent of max size:

ROW A — COARSE (uniformly graded):
- Particles are mostly one size, close to the column's max
- Gaps between stones are EMPTY — few small particles fill them
- Low packing density, visible voids/background between stones
- Looks like a single layer of similar-sized stones

ROW B — MEDIUM (well-graded):
- Balanced mix of sizes
- Gaps between large stones are PARTIALLY filled by smaller ones
- Moderate packing density

ROW C — FINE (continuously graded):
- Wide range of sizes present
- Gaps between large stones are COMPLETELY filled by smaller particles
- Very high packing density, almost no visible voids
- Surface looks dense, tightly packed, heterogeneous

CRITICAL VISUAL CUE: Compare row A vs row C in the same column (same max size).
- Row A: you can see gaps/background between the stones
- Row C: no gaps visible — small particles fill everything

Study each cell carefully before classifying the next image."""

GRAN_FS_QUERY_PROMPT = """Classify this photograph. GSD = 8.0 px/mm (8mm=~64px, 16mm=~128px, 32mm=~256px).

Compare to the reference grid:
1. COLUMN: What is the largest stone size? Match to 8, 16, or 32mm.
2. ROW: Look at the gaps between the largest stones.
   - Gaps EMPTY, single-size layer → coarse (A)
   - Gaps PARTIALLY filled → medium (B)
   - Gaps COMPLETELY filled, dense packed → fine (C)

Respond with ONLY JSON: {"max_particle_size_mm": <8|16|32>, "grading": "<coarse|medium|fine>"}"""

WELD_FS_REF_PROMPT = """First image: a 4×1 reference grid showing one example of each weld defect class from radiographic images.

From LEFT to RIGHT:
  (1) lack_of_penetration — A dark continuous or intermittent line/band running along the weld centerline. This indicates the weld root was not fully fused. The dark line is relatively straight and follows the joint geometry.
  (2) porosity — Scattered small dark circular spots within the weld area. Gas pores are characteristically round or oval. They can appear as individual dots or in clusters. The key feature is the circular shape of each dark spot.
  (3) cracks — Dark, sharp, irregular jagged lines in the weld. Thinner and more erratic than lack of penetration. May branch or change direction. The edges are sharp and the line path is irregular.
  (4) no_defect — Clean, uniform weld with no distinct dark spots, lines, or bands. The radiographic image shows relatively even gray intensity across the weld region.

Focus on the shape of dark features: lines (LP), circles (porosity), jagged lines (cracks), or uniform (no defect)."""

WELD_FS_QUERY_PROMPT = """Now classify this weld radiograph by comparing to the 4 reference examples.

Identify the dominant dark feature:
- Dark straight line/band along weld center → lack_of_penetration
- Scattered dark circular spots/dots → porosity
- Dark sharp jagged irregular lines → cracks
- Uniform gray, no distinct dark features → no_defect

Respond with ONLY JSON: {"defect_class": "<lack_of_penetration|porosity|cracks|no_defect>"}"""

WELD_ZS_PROMPT = """Classify this weld radiographic image.

This is a 227×227 pixel 8-bit grayscale radiographic (X-ray) image of a weld joint. Radiography reveals internal defects as variations in image intensity — defects appear as darker or lighter regions compared to the surrounding sound weld metal.

Possible defect classes:
1. lack_of_penetration: A dark continuous or intermittent line/band running along the weld centerline.
2. porosity: Scattered small dark circular spots within the weld area.
3. cracks: Dark, sharp, irregular jagged lines in the weld.
4. no_defect: Clean, uniform weld with no distinct dark spots, lines, or bands.

Respond with ONLY a JSON object:
{"defect_class": "<lack_of_penetration|porosity|cracks|no_defect>"}"""


def parse_json(raw):
    if not raw:
        return None
    raw = raw.replace('<', '').replace('>', '')
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw).strip()
    try:
        return json.loads(raw)
    except:
        pass
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return None


def benchmark_granulometry(client):
    """Benchmark Gemini on granulometry test set."""
    print(f"\n{'='*60}")
    print(f"  GRANULOMETRY — Gemini 2.5 Pro Zero-Shot")
    print(f"{'='*60}")

    with open(str(GRANULOMETRY_TEST_MANIFEST)) as f:
        manifest = json.load(f)

    correct_size = 0
    correct_grading = 0
    correct_both = 0
    total = 0

    for i, entry in enumerate(manifest):
        img_path = os.path.join(str(GRANULOMETRY_TEST_DIR), entry['image'])
        if not os.path.exists(img_path):
            continue

        # Compute GSD (same as training/eval)
        from PIL import Image
        img = Image.open(img_path)
        scale = min(MAX_DIM / max(img.size), 1.0)
        gsd = ORIGINAL_GSD * scale
        img.close()

        prompt = f"""Classify this concrete aggregate photograph.
Ground sampling distance (GSD) = {gsd:.1f} px/mm.
At this GSD: 8mm stone ≈ {8*gsd:.0f}px, 16mm ≈ {16*gsd:.0f}px, 32mm ≈ {32*gsd:.0f}px.

Classification axes:
1. MAX PARTICLE SIZE: estimate the largest stone's width in pixels, divide by GSD, round to 8, 16, or 32 mm.
2. GRADING (DIN 1045 standard — describes size DISTRIBUTION, not absolute size):
   - COARSE (A): particles concentrated near max size. Gaps between stones are EMPTY. Uniform, single-layer texture.
   - MEDIUM (B): balanced mix. Gaps PARTIALLY filled by smaller particles.
   - FINE (C): wide size range. Gaps COMPLETELY filled with small particles. Dense, packed texture.

Respond with ONLY a JSON object:
{{"max_particle_size_mm": <8, 16, or 32>, "grading": "<coarse, medium, or fine>"}}"""

        with open(img_path, 'rb') as f:
            image_bytes = f.read()
        ext = os.path.splitext(img_path)[1].lower()
        mime = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"

        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime), prompt],
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=2048),
            )
            raw = response.text.strip()
        except Exception as e:
            print(f'  ERROR on {entry["image"]}: {e}')
            time.sleep(1)
            continue

        parsed = parse_json(raw)
        gt_size = entry['max_particle_size_mm']
        gt_grading = entry['grading']

        if parsed:
            pred_size = parsed.get('max_particle_size_mm')
            if isinstance(pred_size, str):
                pred_size = int(pred_size) if pred_size.isdigit() else None
            pred_grading = parsed.get('grading', '').lower().strip()
            if pred_size == gt_size:
                correct_size += 1
            if pred_grading == gt_grading:
                correct_grading += 1
            if pred_size == gt_size and pred_grading == gt_grading:
                correct_both += 1
        total += 1

        if (i + 1) % 20 == 0:
            print(f'  [{i+1}/{len(manifest)}] Size:{correct_size}/{total}({correct_size/total*100:.0f}%) '
                  f'Grade:{correct_grading}/{total}({correct_grading/total*100:.0f}%) '
                  f'Both:{correct_both}/{total}({correct_both/total*100:.0f}%)')
        time.sleep(1.0)

    print(f'\n  FINAL: Size={correct_size}/{total} ({correct_size/total*100:.1f}%) | '
          f'Grading={correct_grading}/{total} ({correct_grading/total*100:.1f}%) | '
          f'Both={correct_both}/{total} ({correct_both/total*100:.1f}%)')
    return correct_both / total * 100 if total > 0 else 0


def benchmark_weld(client):
    """Benchmark Gemini on weld test set (60 per class, same as eval)."""
    import random
    random.seed(42)

    print(f"\n{'='*60}")
    print(f"  WELD DEFECTS — Gemini 2.5 Pro Zero-Shot")
    print(f"{'='*60}")

    # Build test manifest (same sampling as eval)
    manifest = []
    for cls in WELD_CLASSES:
        cls_dir = os.path.join(str(WELD_TEST_DIR), cls)
        images = sorted([f for f in os.listdir(cls_dir) if f.endswith('.png')])
        if len(images) > WELD_SAMPLE_PER_CLASS:
            images = random.sample(images, WELD_SAMPLE_PER_CLASS)
        for img in images:
            manifest.append({'image': os.path.join(cls_dir, img), 'class': cls})
    random.shuffle(manifest)

    correct = 0
    total = 0

    for i, entry in enumerate(manifest):
        img_path = entry['image']
        with open(img_path, 'rb') as f:
            image_bytes = f.read()

        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/png"), WELD_ZS_PROMPT],
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=2048),
            )
            raw = response.text.strip()
        except Exception as e:
            print(f'  ERROR: {e}')
            time.sleep(1)
            continue

        parsed = parse_json(raw)
        gt_class = entry['class']
        if parsed:
            pred = parsed.get('defect_class', '').lower().strip()
            if pred == gt_class:
                correct += 1
        total += 1

        if (i + 1) % 60 == 0:
            print(f'  [{i+1}/{len(manifest)}] Acc:{correct}/{total}({correct/total*100:.0f}%)')
        time.sleep(1.0)

    print(f'\n  FINAL: {correct}/{total} ({correct/total*100:.1f}%)')
    return correct / total * 100 if total > 0 else 0


def benchmark_granulometry_fs(client):
    """Benchmark Gemini on granulometry with few-shot (reference grid)."""
    print(f"\n{'='*60}")
    print(f"  GRANULOMETRY — Gemini 2.5 Pro Few-Shot")
    print(f"{'='*60}")

    with open(str(GRANULOMETRY_TEST_MANIFEST)) as f:
        manifest = json.load(f)

    # Load reference image
    with open(str(GRANULOMETRY_REF_IMAGE), 'rb') as f:
        ref_bytes = f.read()
    ref_mime = "image/png"

    correct_both = 0
    total = 0

    for i, entry in enumerate(manifest):
        img_path = os.path.join(str(GRANULOMETRY_TEST_DIR), entry['image'])
        if not os.path.exists(img_path):
            continue

        with open(img_path, 'rb') as f:
            image_bytes = f.read()
        ext = os.path.splitext(img_path)[1].lower()
        mime = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"

        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[
                    types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime),
                    GRAN_FS_REF_PROMPT,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    GRAN_FS_QUERY_PROMPT,
                ],
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=2048),
            )
            raw = response.text.strip()
        except Exception as e:
            print(f'  ERROR: {e}')
            time.sleep(1)
            continue

        parsed = parse_json(raw)
        gt_size = entry['max_particle_size_mm']
        gt_grading = entry['grading']

        if parsed:
            pred_size = parsed.get('max_particle_size_mm')
            if isinstance(pred_size, str):
                pred_size = int(pred_size) if pred_size.isdigit() else None
            pred_grading = parsed.get('grading', '').lower().strip()
            if pred_size == gt_size and pred_grading == gt_grading:
                correct_both += 1
        total += 1

        if (i + 1) % 20 == 0:
            print(f'  [{i+1}/{len(manifest)}] Both:{correct_both}/{total}({correct_both/total*100:.0f}%)')
        time.sleep(1.0)

    print(f'\n  FINAL FS: Both={correct_both}/{total} ({correct_both/total*100:.1f}%)')
    return correct_both / total * 100 if total > 0 else 0


def benchmark_weld_fs(client):
    """Benchmark Gemini on weld with few-shot (reference grid)."""
    import random
    random.seed(42)

    print(f"\n{'='*60}")
    print(f"  WELD DEFECTS — Gemini 2.5 Pro Few-Shot")
    print(f"{'='*60}")

    # Load reference image
    with open(str(WELD_REF_IMAGE), 'rb') as f:
        ref_bytes = f.read()
    ref_mime = "image/png"

    # Build test manifest
    manifest = []
    for cls in WELD_CLASSES:
        cls_dir = os.path.join(str(WELD_TEST_DIR), cls)
        images = sorted([f for f in os.listdir(cls_dir) if f.endswith('.png')])
        if len(images) > WELD_SAMPLE_PER_CLASS:
            images = random.sample(images, WELD_SAMPLE_PER_CLASS)
        for img in images:
            manifest.append({'image': os.path.join(cls_dir, img), 'class': cls})
    random.shuffle(manifest)

    correct = 0
    total = 0

    for i, entry in enumerate(manifest):
        img_path = entry['image']
        with open(img_path, 'rb') as f:
            image_bytes = f.read()

        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[
                    types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime),
                    WELD_FS_REF_PROMPT,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    WELD_FS_QUERY_PROMPT,
                ],
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=2048),
            )
            raw = response.text.strip()
        except Exception as e:
            print(f'  ERROR: {e}')
            time.sleep(1)
            continue

        parsed = parse_json(raw)
        gt_class = entry['class']
        if parsed:
            pred = parsed.get('defect_class', '').lower().strip()
            if pred == gt_class:
                correct += 1
        total += 1

        if (i + 1) % 60 == 0:
            print(f'  [{i+1}/{len(manifest)}] Acc:{correct}/{total}({correct/total*100:.0f}%)')
        time.sleep(1.0)

    print(f'\n  FINAL FS: {correct}/{total} ({correct/total*100:.1f}%)')
    return correct / total * 100 if total > 0 else 0


if __name__ == "__main__":
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

    # Zero-shot
    gran_zs = benchmark_granulometry(client)
    weld_zs = benchmark_weld(client)

    # Few-shot
    gran_fs = benchmark_granulometry_fs(client)
    weld_fs = benchmark_weld_fs(client)

    print(f'\n\n{"="*60}')
    print("GEMINI 2.5 PRO BENCHMARK RESULTS")
    print(f'{"="*60}')
    print(f'  Granulometry ZS (both correct): {gran_zs:.1f}%')
    print(f'  Granulometry FS (both correct): {gran_fs:.1f}%')
    print(f'  Weld Defects ZS (accuracy):     {weld_zs:.1f}%')
    print(f'  Weld Defects FS (accuracy):     {weld_fs:.1f}%')

    # Save
    results = {
        "model": "gemini-2.5-pro",
        "granulometry_zs_both_correct": round(gran_zs, 1),
        "granulometry_fs_both_correct": round(gran_fs, 1),
        "weld_zs_accuracy": round(weld_zs, 1),
        "weld_fs_accuracy": round(weld_fs, 1),
    }
    with open(SCRIPT_DIR / "gemini_benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved to gemini_benchmark_results.json')


