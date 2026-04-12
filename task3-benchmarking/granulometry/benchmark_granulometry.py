"""
Task 3: Benchmark Qwen2.5-VL-3B (base model) on granulometry test set.

Asks the model to classify each image by max particle size and grading.
Compares model output against ground truth and reports accuracy.

Supports two modes:
  zero-shot: No reference image, model guesses grading from visual appearance alone.
  few-shot:  Reference image (examples_classification_data.png) included in prompt
             so the model can learn the A=coarse, B=medium, C=fine convention.

Usage:
    python benchmark_granulometry.py                          # zero-shot, all images
    python benchmark_granulometry.py --mode few-shot          # few-shot with reference
    python benchmark_granulometry.py --limit 10               # quick test
    python benchmark_granulometry.py --mode few-shot --limit 5
"""
import os
import sys
import json
import re
import time
import torch
from PIL import Image
from collections import defaultdict

# --- Config ---
TEST_DIR = os.environ.get("TEST_DIR", "../../datasets/granulometry/test")
MANIFEST = os.environ.get("MANIFEST", "../../datasets/granulometry/test_manifest.json")
REF_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "examples_classification_data.png")
ORIGINAL_GSD = 8.0   # pixels per mm at original resolution (2200x3000)
MAX_DIM = 1500        # max pixels on longest side

PROMPT_ZERO_SHOT = """This is a top-down photograph of concrete aggregate particles.
The ground sampling distance (GSD) is {gsd:.1f} pixels per mm — use this to measure particle sizes.

Classification rules:
- max_particle_size_mm: measure the largest particle visible. Round to the nearest standard sieve size: 8, 16, or 32 mm.
- grading:
  - "coarse" means the majority of particles are larger than 16 mm
  - "medium" means the majority of particles are between 8 mm and 16 mm
  - "fine" means the majority of particles are smaller than 8 mm

Respond with ONLY a JSON object (no other text):
{{"max_particle_size_mm": <8, 16, or 32>, "grading": "<coarse, medium, or fine>"}}"""

PROMPT_FEW_SHOT_REF = """The first image is a reference chart for classifying concrete aggregate particles.
It is a 3x3 grid of example photographs with labeled rows and columns:
- The 3 COLUMNS represent max particle size, labeled left to right: 8 mm, 16 mm, 32 mm
- The 3 ROWS represent grading type, labeled top to bottom: A (coarse), B (medium), C (fine)
- Each cell shows a real example of that combination (e.g. top-right = A/32mm = coarse grading with 32 mm max particle size)

Grading definitions:
- Coarse (row A): majority of particles are larger than 16 mm
- Medium (row B): majority of particles are between 8 mm and 16 mm
- Fine (row C): majority of particles are smaller than 8 mm

Study this grid carefully, then classify the second image."""

PROMPT_FEW_SHOT_QUERY = """Now classify this photograph of concrete aggregate particles.
The ground sampling distance (GSD) is {gsd:.1f} pixels per mm — use this to measure particle sizes.

Compare the particle sizes and distribution to the reference grid.
Which row (A=coarse, B=medium, C=fine) and column (8mm, 16mm, 32mm) does it best match?

Respond with ONLY a JSON object (no other text):
{{"max_particle_size_mm": <8, 16, or 32>, "grading": "<coarse, medium, or fine>"}}"""


def parse_response(raw):
    """Extract max_particle_size_mm and grading from model response."""
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Regex fallback
    size_m = re.search(r'"max_particle_size_mm"\s*:\s*(\d+)', raw)
    grad_m = re.search(r'"grading"\s*:\s*"(\w+)"', raw)
    if size_m and grad_m:
        return {"max_particle_size_mm": int(size_m.group(1)), "grading": grad_m.group(1)}

    return None


def load_model():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading Qwen2.5-VL-3B...")
    t = time.time()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "6GiB", 1: "15GiB"},
    )
    print(f"Loaded in {time.time()-t:.1f}s")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {alloc:.1f} GB allocated")
    return model, processor


def infer(model, processor, img_path, mode="zero-shot", ref_image=None):
    """Load image, resize to MAX_DIM preserving aspect ratio, compute actual GSD, run inference, free memory."""
    image = Image.open(img_path).convert("RGB")
    orig_max = max(image.size)
    # Use smaller images for few-shot (2 images = 2x VRAM)
    max_dim = 800 if (mode == "few-shot" and ref_image is not None) else MAX_DIM
    scale = min(max_dim / orig_max, 1.0)  # don't upscale
    if scale < 1.0:
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    actual_gsd = ORIGINAL_GSD * scale

    if mode == "few-shot" and ref_image is not None:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": ref_image},
            {"type": "text", "text": PROMPT_FEW_SHOT_REF},
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT_FEW_SHOT_QUERY.format(gsd=actual_gsd)},
        ]}]
        images = [ref_image, image]
    else:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT_ZERO_SHOT.format(gsd=actual_gsd)},
        ]}]
        images = [image]

    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True).to(model.device)

    t = time.time()
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=True)
    elapsed = time.time() - t

    out = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    # Free GPU memory
    del inputs, ids
    image.close()
    torch.cuda.empty_cache()

    return out.strip(), elapsed


def main():
    # Parse args
    limit = None
    mode = "zero-shot"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--test-dir" and i + 1 < len(sys.argv) - 1:
            global TEST_DIR, MANIFEST
            TEST_DIR = sys.argv[i + 2]
            MANIFEST = os.path.join(TEST_DIR, "..", "test_manifest.json")
        if arg == "--limit" and i + 1 < len(sys.argv) - 1:
            limit = int(sys.argv[i + 2])
        if arg == "--mode" and i + 1 < len(sys.argv) - 1:
            mode = sys.argv[i + 2]
            if mode not in ("zero-shot", "few-shot"):
                print(f"Error: --mode must be 'zero-shot' or 'few-shot', got '{mode}'")
                sys.exit(1)

    results_file = f"benchmark_results_{mode}.json"

    # Load reference image for few-shot
    ref_image = None
    if mode == "few-shot":
        if not os.path.exists(REF_IMAGE_PATH):
            print(f"Error: Reference image not found at {REF_IMAGE_PATH}")
            print("Few-shot mode requires examples_classification_data.png in the script directory.")
            sys.exit(1)
        raw_ref = Image.open(REF_IMAGE_PATH).convert("RGBA")
        white_bg = Image.new("RGBA", raw_ref.size, (255, 255, 255, 255))
        ref_image = Image.alpha_composite(white_bg, raw_ref).convert("RGB")
        ref_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        print(f"Mode: few-shot (reference: {REF_IMAGE_PATH}, resized to {ref_image.size})")
    else:
        print(f"Mode: zero-shot (no reference image)")

    # Load manifest
    with open(MANIFEST) as f:
        manifest = json.load(f)
    if limit:
        manifest = manifest[:limit]

    print(f"Test set: {len(manifest)} images from {TEST_DIR}")
    print(f"Max image dimension: {MAX_DIM}px (original GSD: {ORIGINAL_GSD} px/mm, resized GSD: ~{ORIGINAL_GSD * MAX_DIM / 3000:.1f} px/mm)\n")

    model, processor = load_model()

    # Run benchmark
    results = []
    correct_size = 0
    correct_grading = 0
    valid_json = 0
    total_time = 0

    for i, entry in enumerate(manifest):
        img_path = os.path.join(TEST_DIR, entry["image"])
        if not os.path.exists(img_path):
            print(f"  Skipping {entry['image']} (not found)")
            continue

        raw, elapsed = infer(model, processor, img_path, mode=mode, ref_image=ref_image)
        total_time += elapsed

        parsed = parse_response(raw)
        gt_size = entry["max_particle_size_mm"]
        gt_grading = entry["grading"]

        size_correct = False
        grading_correct = False
        is_valid = parsed is not None

        if is_valid:
            valid_json += 1
            pred_size = parsed.get("max_particle_size_mm")
            pred_grading = parsed.get("grading", "").lower()

            if pred_size == gt_size:
                size_correct = True
                correct_size += 1
            if pred_grading == gt_grading:
                grading_correct = True
                correct_grading += 1

        result = {
            "image": entry["image"],
            "class": entry["class"],
            "gt_size": gt_size,
            "gt_grading": gt_grading,
            "predicted": parsed,
            "raw_response": raw,
            "size_correct": size_correct,
            "grading_correct": grading_correct,
            "valid_json": is_valid,
            "time_s": round(elapsed, 2),
        }
        results.append(result)

        status = "✓" if (size_correct and grading_correct) else "✗"
        pred_str = f"size={parsed.get('max_particle_size_mm','?')}, grading={parsed.get('grading','?')}" if parsed else "PARSE FAIL"
        print(f"  [{i+1}/{len(manifest)}] {entry['class']:>3} | GT: size={gt_size}, grading={gt_grading} | Pred: {pred_str} | {status} | {elapsed:.1f}s")

    # Summary
    n = len(results)
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS — Qwen2.5-VL-3B (base model) — {mode}")
    print(f"=" * 70)
    print(f"Images tested:        {n}")
    print(f"JSON validity:        {valid_json}/{n} ({valid_json/n*100:.1f}%)")
    print(f"Size accuracy:        {correct_size}/{n} ({correct_size/n*100:.1f}%)")
    print(f"Grading accuracy:     {correct_grading}/{n} ({correct_grading/n*100:.1f}%)")
    print(f"Both correct:         {sum(1 for r in results if r['size_correct'] and r['grading_correct'])}/{n}")
    print(f"Avg inference time:   {total_time/n:.2f}s")
    print(f"Total time:           {total_time:.1f}s")

    # Per-class breakdown
    print(f"\nPer-class accuracy:")
    print(f"{'Class':<6} {'Size':>8} {'Grading':>10} {'Both':>8}")
    print("-" * 35)
    class_results = defaultdict(list)
    for r in results:
        class_results[r["class"]].append(r)
    for cls in sorted(class_results):
        cr = class_results[cls]
        sc = sum(1 for r in cr if r["size_correct"])
        gc = sum(1 for r in cr if r["grading_correct"])
        bc = sum(1 for r in cr if r["size_correct"] and r["grading_correct"])
        print(f"{cls:<6} {sc}/{len(cr):>5} {gc}/{len(cr):>8} {bc}/{len(cr):>6}")

    # Save results
    with open(results_file, "w") as f:
        json.dump({
            "model": "Qwen2.5-VL-3B-Instruct",
            "mode": mode,
            "phase": "base_model",
            "total_images": n,
            "json_validity_pct": round(valid_json / n * 100, 1),
            "size_accuracy_pct": round(correct_size / n * 100, 1),
            "grading_accuracy_pct": round(correct_grading / n * 100, 1),
            "avg_inference_time_s": round(total_time / n, 2),
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")


if __name__ == "__main__":
    main()
