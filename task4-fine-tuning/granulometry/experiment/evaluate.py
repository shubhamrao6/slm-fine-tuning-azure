"""
Evaluate fine-tuned Qwen2.5-VL-3B on the granulometry test set.
Compares fine-tuned model against Task 3 baseline.

Usage:
    # Evaluate Approach A (standard LoRA)
    python evaluate.py --adapter lora_direct --output results_direct.json

    # Evaluate Approach B (SEAL-augmented LoRA)
    python evaluate.py --adapter lora_augmented --output results_augmented.json

    # Compare all results
    python evaluate.py --compare
"""
import os, sys, json, re, time, torch
from PIL import Image
from collections import defaultdict, Counter

TEST_DIR = os.environ.get("TEST_DIR", "../../datasets/granulometry/test")
MANIFEST = os.environ.get("MANIFEST", "../../datasets/granulometry/test_manifest.json")
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_DIM = 1500
ORIGINAL_GSD = 8.0

PROMPT = (
    "Classify this concrete aggregate photo. GSD = {gsd:.1f} px/mm.\n"
    "Grading describes size distribution: coarse = uniform, mostly large; "
    "medium = moderate mix; fine = many small particles filling gaps, dense and packed.\n"
    'Respond with ONLY JSON: {{"max_particle_size_mm": <8|16|32>, "grading": "<coarse|medium|fine>"}}'
)


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


def load_model(adapter_path=None):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if adapter_path:
        print(f"Loading base model + LoRA adapter from {adapter_path}...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # merge for faster inference
        print("LoRA merged into base model.")
    else:
        print("Loading base model (no adapter)...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        )

    return model, processor


def run_eval(model, processor, manifest):
    results = []
    correct_size = 0; correct_grading = 0; valid_json = 0; total_time = 0

    for i, entry in enumerate(manifest):
        img_path = os.path.join(TEST_DIR, entry["image"])
        if not os.path.exists(img_path): continue

        image = Image.open(img_path).convert("RGB")
        scale = min(MAX_DIM / max(image.size), 1.0)
        if scale < 1.0:
            image = image.resize((int(image.width*scale), int(image.height*scale)), Image.Resampling.LANCZOS)
        actual_gsd = ORIGINAL_GSD * scale

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT.format(gsd=actual_gsd)},
        ]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)

        t = time.time()
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=True)
        elapsed = time.time() - t

        out = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        del inputs, ids; image.close(); torch.cuda.empty_cache()

        parsed = parse_response(out)
        gt_size = entry["max_particle_size_mm"]; gt_grading = entry["grading"]
        size_ok = False; grading_ok = False; is_valid = parsed is not None
        total_time += elapsed

        if is_valid:
            valid_json += 1
            ps = parsed.get("max_particle_size_mm")
            if isinstance(ps, str): ps = int(ps) if ps.isdigit() else None
            if ps == gt_size: size_ok = True; correct_size += 1
            if parsed.get("grading", "").lower() == gt_grading: grading_ok = True; correct_grading += 1

        results.append({
            "image": entry["image"], "class": entry["class"],
            "gt_size": gt_size, "gt_grading": gt_grading,
            "predicted": parsed, "raw": out,
            "size_correct": size_ok, "grading_correct": grading_ok,
            "valid_json": is_valid, "time_s": round(elapsed, 2),
        })

        if (i+1) % 10 == 0:
            n = i + 1
            print(f"  [{n}/{len(manifest)}] Size: {correct_size}/{n} ({correct_size/n*100:.0f}%) | "
                  f"Grading: {correct_grading}/{n} ({correct_grading/n*100:.0f}%) | "
                  f"JSON: {valid_json}/{n}")

    return results, correct_size, correct_grading, valid_json, total_time


def print_results(label, results, c_size, c_grading, v_json, t_time):
    n = len(results)
    both = sum(1 for r in results if r["size_correct"] and r["grading_correct"])
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Images: {n}  JSON: {v_json}/{n} ({v_json/n*100:.1f}%)")
    print(f"Size:   {c_size}/{n} ({c_size/n*100:.1f}%)")
    print(f"Grade:  {c_grading}/{n} ({c_grading/n*100:.1f}%)")
    print(f"Both:   {both}/{n} ({both/n*100:.1f}%)")
    print(f"Time:   {t_time:.0f}s ({t_time/n:.1f}s/img)")

    # Per-class
    by_class = defaultdict(list)
    for r in results: by_class[r["class"]].append(r)
    print(f"\n{'Class':<6} {'Size':>6} {'Grade':>6} {'Both':>6}")
    print("-" * 28)
    for cls in sorted(by_class):
        cr = by_class[cls]
        sc = sum(1 for r in cr if r["size_correct"])
        gc = sum(1 for r in cr if r["grading_correct"])
        bc = sum(1 for r in cr if r["size_correct"] and r["grading_correct"])
        print(f"{cls:<6} {sc:>3}/{len(cr)} {gc:>3}/{len(cr)} {bc:>3}/{len(cr)}")


def compare_results():
    """Compare baseline, direct LoRA, and augmented LoRA results."""
    files = {
        "Baseline (zero-shot)": "../../task3-benchmarking/granulometry/benchmark_results_zero-shot.json",
        "Baseline (few-shot)": "../../task3-benchmarking/granulometry/benchmark_results_few-shot.json",
        "LoRA Direct (18 ex)": "results_direct.json",
        "LoRA Augmented (~150 ex)": "results_augmented.json",
    }

    print(f"\n{'Method':<28} {'JSON':>6} {'Size':>6} {'Grade':>6} {'Both':>6} {'Time':>6}")
    print("-" * 62)
    for label, path in files.items():
        if not os.path.exists(path):
            print(f"{label:<28} {'—':>6} {'—':>6} {'—':>6} {'—':>6} {'—':>6}")
            continue
        with open(path) as f:
            d = json.load(f)
        print(f"{label:<28} {d['json_validity_pct']:>5.0f}% {d['size_accuracy_pct']:>5.1f}% "
              f"{d['grading_accuracy_pct']:>5.1f}% {d['both_correct_pct']:>5.1f}% "
              f"{d['avg_inference_time_s']:>5.1f}s")


def main():
    if "--compare" in sys.argv:
        compare_results()
        return

    adapter_path = None
    output_file = "results.json"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--adapter" and i + 1 < len(sys.argv) - 1: adapter_path = sys.argv[i + 2]
        if arg == "--output" and i + 1 < len(sys.argv) - 1: output_file = sys.argv[i + 2]

    with open(MANIFEST) as f:
        manifest = json.load(f)
    print(f"Test set: {len(manifest)} images")

    model, processor = load_model(adapter_path)
    results, c_size, c_grading, v_json, t_time = run_eval(model, processor, manifest)

    label = f"Fine-tuned ({adapter_path})" if adapter_path else "Base model"
    print_results(label, results, c_size, c_grading, v_json, t_time)

    # Save
    n = len(results)
    both = sum(1 for r in results if r["size_correct"] and r["grading_correct"])
    with open(output_file, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "adapter": adapter_path,
            "phase": "fine_tuned" if adapter_path else "base_model",
            "total_images": n,
            "json_validity_pct": round(v_json/n*100, 1),
            "size_accuracy_pct": round(c_size/n*100, 1),
            "grading_accuracy_pct": round(c_grading/n*100, 1),
            "both_correct_pct": round(both/n*100, 1),
            "avg_inference_time_s": round(t_time/n, 2),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
