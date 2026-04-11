"""
Task 3: Benchmark Qwen2.5-VL-3B (base model) on granulometry test set.

Asks the model to classify each image by max particle size and grading.
Compares model output against ground truth and reports accuracy.

Usage:
    python benchmark_granulometry.py [--test-dir ../../datasets/granulometry/test]
    python benchmark_granulometry.py --limit 10  # quick test with 10 images
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
IMG_SIZE = 500
RESULTS_FILE = "benchmark_results.json"

PROMPT = f"""This image shows concrete aggregate particles photographed from above.
The image resolution is 8 pixels per mm.

Analyze the particles and respond with ONLY a JSON object (no other text):
{{"max_particle_size_mm": <estimated maximum particle size as integer: 8, 16, or 32>, "grading": "<coarse, medium, or fine>"}}

The values 8, 16, 32 for max_particle_size_mm and coarse, medium, fine for grading are for reference. Use the actual values based on what you observe."""


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
    )
    print(f"Loaded in {time.time()-t:.1f}s")
    return model, processor


def infer(model, processor, image):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
    t = time.time()
    ids = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=True)
    elapsed = time.time() - t
    out = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return out.strip(), elapsed


def main():
    # Parse args
    limit = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--test-dir" and i + 1 < len(sys.argv) - 1:
            global TEST_DIR, MANIFEST
            TEST_DIR = sys.argv[i + 2]
            MANIFEST = os.path.join(TEST_DIR, "..", "test_manifest.json")
        if arg == "--limit" and i + 1 < len(sys.argv) - 1:
            limit = int(sys.argv[i + 2])

    # Load manifest
    with open(MANIFEST) as f:
        manifest = json.load(f)
    if limit:
        manifest = manifest[:limit]

    print(f"Test set: {len(manifest)} images from {TEST_DIR}")
    print(f"Prompt: {PROMPT[:100]}...\n")

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

        image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        raw, elapsed = infer(model, processor, image)
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
    print(f"BENCHMARK RESULTS — Qwen2.5-VL-3B (base model)")
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
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "model": "Qwen2.5-VL-3B-Instruct",
            "phase": "base_model",
            "total_images": n,
            "json_validity_pct": round(valid_json / n * 100, 1),
            "size_accuracy_pct": round(correct_size / n * 100, 1),
            "grading_accuracy_pct": round(correct_grading / n * 100, 1),
            "avg_inference_time_s": round(total_time / n, 2),
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
