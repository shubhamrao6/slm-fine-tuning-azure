"""
Task 3: Benchmark Qwen2.5-VL-3B (base model) on granulometry test set.

Two modes:
  zero-shot: 1500px, no reference image.
  few-shot:  ref@800 + test@1400, reference classification chart included.

Usage:
    python benchmark_granulometry.py --mode zero-shot
    python benchmark_granulometry.py --mode few-shot
    python benchmark_granulometry.py --mode zero-shot --limit 10
"""
import os, sys, json, re, time, torch
from PIL import Image
from collections import defaultdict

# --- Config ---
TEST_DIR = os.environ.get("TEST_DIR", "../../datasets/granulometry/test")
MANIFEST = os.environ.get("MANIFEST", "../../datasets/granulometry/test_manifest.json")
REF_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples_classification_data.png")
ORIGINAL_GSD = 8.0
MAX_DIM_ZS = 1500
MAX_DIM_FS = 1400

PROMPT_ZERO_SHOT = """This is a top-down photo of concrete aggregate. GSD = {gsd:.1f} px/mm.

Classify it on two axes:

MAX PARTICLE SIZE — estimate the largest stone's width in pixels, divide by {gsd:.1f} to get mm, round to 8, 16, or 32.

GRADING — describes the size distribution relative to the max size:
- COARSE: most particles are similar in size, close to the max. Few small particles. Looks uniform.
- MEDIUM: moderate mix of large and small.
- FINE: wide range of sizes. Many small particles fill gaps between larger ones. Looks dense and packed.

Example: "coarse 32mm" = mostly 16-32mm stones, few small ones. "fine 32mm" = some 32mm stones but lots of 4-16mm particles filling every gap.

Respond with ONLY a JSON object (no other text):
{{"max_particle_size_mm": <8, 16, or 32>, "grading": "<coarse, medium, or fine>"}}"""

PROMPT_FEW_SHOT_REF = """Reference chart: 3x3 grid of concrete aggregate photos.

COLUMNS (left to right) = max particle size: 8mm | 16mm | 32mm
ROWS (top to bottom) = grading: A (coarse) | B (medium) | C (fine)

What grading means — it describes size DISTRIBUTION, not absolute size:
- COARSE (A): particles are mostly the same size, close to the max. Few small ones. Uniform texture.
- MEDIUM (B): moderate mix of large and small.
- FINE (C): many different sizes. Small particles fill all gaps between big ones. Dense, packed texture.

Look at column 32mm: row A has mostly big uniform stones, row C has big stones BUT also lots of small ones filling every gap. That is the difference."""

PROMPT_FEW_SHOT_QUERY = """Classify this photo. GSD = {gsd:.1f} px/mm.

Compare to the reference grid:
1. Which COLUMN? (8, 16, or 32mm) — match the largest stone size.
2. Which ROW? (A=coarse, B=medium, C=fine) — uniform texture = coarse, packed mixed texture = fine.

Respond with ONLY a JSON object (no other text):
{{"max_particle_size_mm": <8, 16, or 32>, "grading": "<coarse, medium, or fine>"}}"""


def parse_response(raw):
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict): return obj
    except: pass
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
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
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16,
        device_map="auto", max_memory={0: "6GiB", 1: "15GiB"},
    )
    print(f"Loaded in {time.time()-t:.1f}s")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.1f} GB")
    return model, processor


def infer(model, processor, img_path, mode="zero-shot", ref_image=None):
    image = Image.open(img_path).convert("RGB")
    max_dim = MAX_DIM_FS if (mode == "few-shot" and ref_image is not None) else MAX_DIM_ZS
    orig_max = max(image.size)
    scale = min(max_dim / orig_max, 1.0)
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
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
    del inputs, ids; image.close(); torch.cuda.empty_cache()
    return out.strip(), elapsed


def main():
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

    ref_image = None
    if mode == "few-shot":
        if not os.path.exists(REF_IMAGE_PATH):
            print(f"Error: {REF_IMAGE_PATH} not found"); sys.exit(1)
        raw_ref = Image.open(REF_IMAGE_PATH).convert("RGBA")
        white_bg = Image.new("RGBA", raw_ref.size, (255, 255, 255, 255))
        ref_image = Image.alpha_composite(white_bg, raw_ref).convert("RGB")
        ref_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        print(f"Mode: few-shot (ref: {ref_image.size}, test max: {MAX_DIM_FS}px)")
    else:
        print(f"Mode: zero-shot (test max: {MAX_DIM_ZS}px)")

    with open(MANIFEST) as f:
        manifest = json.load(f)
    if limit:
        manifest = manifest[:limit]
    print(f"Test set: {len(manifest)} images from {TEST_DIR}\n")

    model, processor = load_model()

    results = []
    correct_size = 0; correct_grading = 0; valid_json = 0; total_time = 0

    for i, entry in enumerate(manifest):
        img_path = os.path.join(TEST_DIR, entry["image"])
        if not os.path.exists(img_path):
            print(f"  Skipping {entry['image']} (not found)"); continue

        raw, elapsed = infer(model, processor, img_path, mode=mode, ref_image=ref_image)
        total_time += elapsed
        parsed = parse_response(raw)
        gt_size = entry["max_particle_size_mm"]; gt_grading = entry["grading"]
        size_ok = False; grading_ok = False; is_valid = parsed is not None

        if is_valid:
            valid_json += 1
            pred_size = parsed.get("max_particle_size_mm")
            if isinstance(pred_size, str): pred_size = int(pred_size) if pred_size.isdigit() else None
            pred_grading = parsed.get("grading", "").lower()
            if pred_size == gt_size: size_ok = True; correct_size += 1
            if pred_grading == gt_grading: grading_ok = True; correct_grading += 1

        results.append({
            "image": entry["image"], "class": entry["class"],
            "gt_size": gt_size, "gt_grading": gt_grading,
            "predicted": parsed, "raw_response": raw,
            "size_correct": size_ok, "grading_correct": grading_ok,
            "valid_json": is_valid, "time_s": round(elapsed, 2),
        })

        status = "✓" if (size_ok and grading_ok) else "✗"
        if parsed:
            ps = parsed.get('max_particle_size_mm', '?'); pg = parsed.get('grading', '?')
            print(f"  [{i+1}/{len(manifest)}] {entry['class']:>3} | GT: {gt_size}mm {gt_grading} | Pred: {ps}mm {pg} | {status} | {elapsed:.1f}s")
        else:
            print(f"  [{i+1}/{len(manifest)}] {entry['class']:>3} | GT: {gt_size}mm {gt_grading} | PARSE FAIL | {elapsed:.1f}s")

    # --- Summary ---
    n = len(results)
    both_correct = sum(1 for r in results if r["size_correct"] and r["grading_correct"])

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK RESULTS — Qwen2.5-VL-3B (base model) — {mode}")
    print(f"{'=' * 70}")
    print(f"Images tested:        {n}")
    print(f"JSON validity:        {valid_json}/{n} ({valid_json/n*100:.1f}%)")
    print(f"Size accuracy:        {correct_size}/{n} ({correct_size/n*100:.1f}%)")
    print(f"Grading accuracy:     {correct_grading}/{n} ({correct_grading/n*100:.1f}%)")
    print(f"Both correct:         {both_correct}/{n} ({both_correct/n*100:.1f}%)")
    print(f"Avg inference time:   {total_time/n:.2f}s")
    print(f"Total time:           {total_time:.1f}s")

    # Per-class breakdown
    print(f"\nPer-class accuracy:")
    print(f"{'Class':<6} {'Size':>8} {'Grading':>10} {'Both':>8}")
    print("-" * 35)
    class_results = defaultdict(list)
    for r in results: class_results[r["class"]].append(r)
    for cls in sorted(class_results):
        cr = class_results[cls]
        sc = sum(1 for r in cr if r["size_correct"])
        gc = sum(1 for r in cr if r["grading_correct"])
        bc = sum(1 for r in cr if r["size_correct"] and r["grading_correct"])
        print(f"{cls:<6} {sc}/{len(cr):>5} {gc}/{len(cr):>8} {bc}/{len(cr):>6}")

    # Save
    with open(results_file, "w") as f:
        json.dump({
            "model": "Qwen2.5-VL-3B-Instruct", "mode": mode, "phase": "base_model",
            "total_images": n,
            "json_validity_pct": round(valid_json / n * 100, 1),
            "size_accuracy_pct": round(correct_size / n * 100, 1),
            "grading_accuracy_pct": round(correct_grading / n * 100, 1),
            "both_correct_pct": round(both_correct / n * 100, 1),
            "avg_inference_time_s": round(total_time / n, 2),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
