"""
Generate Unconditioned CoT: Gemini 2.5 Pro classifies training images freely (no answer provided).

This produces training data where the frontier model's OWN prediction (often wrong) is the label.
For granulometry, frontier models get ~30% correct, so ~70% of training data will have wrong reasoning.
"""
import os
import sys
import json
import time
import base64
from pathlib import Path
from google import genai
from google.genai import types

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# GCP config
GCP_PROJECT = "project-162f6734-044f-424a-9ad"
GCP_LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-pro"

# Tasks config
TASKS = {
    "granulometry": {
        "direct_jsonl": REPO_ROOT / "task4-fine-tuning" / "granulometry" / "training_data_direct.jsonl",
        "output_jsonl": SCRIPT_DIR / "granulometry_unconditioned_cot.jsonl",
        "classification_prompt": """Classify this concrete aggregate photograph.
Ground sampling distance (GSD) = 2.1 px/mm.
At this GSD: 8mm stone ≈ 17px, 16mm ≈ 34px, 32mm ≈ 68px.

Classification axes:
1. MAX PARTICLE SIZE: estimate the largest stone's width in pixels, divide by GSD, round to 8, 16, or 32 mm.
2. GRADING (DIN 1045 standard — describes size DISTRIBUTION, not absolute size):
   - COARSE (A): particles concentrated near max size. Gaps between stones are EMPTY. Uniform, single-layer texture.
   - MEDIUM (B): balanced mix. Gaps PARTIALLY filled by smaller particles.
   - FINE (C): wide size range. Gaps COMPLETELY filled with small particles. Dense, packed texture.

First explain your reasoning in 2-3 sentences, then respond with JSON:
{"max_particle_size_mm": <8, 16, or 32>, "grading": "<coarse, medium, or fine>"}""",
        "training_prompt": """Classify this concrete aggregate photograph.
Ground sampling distance (GSD) = 2.1 px/mm.
At this GSD: 8mm stone ≈ 17px, 16mm ≈ 34px, 32mm ≈ 68px.

Classification axes:
1. MAX PARTICLE SIZE: estimate the largest stone's width in pixels, divide by GSD, round to 8, 16, or 32 mm.
2. GRADING (DIN 1045 standard — describes size DISTRIBUTION, not absolute size):
   - COARSE (A): particles concentrated near max size. Gaps between stones are EMPTY. Uniform, single-layer texture.
   - MEDIUM (B): balanced mix. Gaps PARTIALLY filled by smaller particles.
   - FINE (C): wide size range. Gaps COMPLETELY filled with small particles. Dense, packed texture.""",
    },
    "weld": {
        "direct_jsonl": REPO_ROOT / "task4-fine-tuning" / "riawelc-weld" / "training_data_direct.jsonl",
        "output_jsonl": SCRIPT_DIR / "weld_unconditioned_cot.jsonl",
        "classification_prompt": """Classify this weld radiographic image.

This is a 227×227 pixel 8-bit grayscale radiographic (X-ray) image of a weld joint. Radiography reveals internal defects as variations in image intensity — defects appear as darker or lighter regions compared to the surrounding sound weld metal.

Possible defect classes:
1. lack_of_penetration: A dark continuous or intermittent line/band running along the weld centerline.
2. porosity: Scattered small dark circular spots within the weld area.
3. cracks: Dark, sharp, irregular jagged lines in the weld.
4. no_defect: Clean, uniform weld with no distinct dark spots, lines, or bands.

First explain your reasoning in 2-3 sentences, then respond with JSON:
{"defect_class": "<lack_of_penetration|porosity|cracks|no_defect>"}""",
        "training_prompt": """Classify this weld radiographic image.

This is a 227×227 pixel 8-bit grayscale radiographic (X-ray) image of a weld joint. Radiography reveals internal defects as variations in image intensity — defects appear as darker or lighter regions compared to the surrounding sound weld metal.

Possible defect classes:
1. lack_of_penetration: A dark continuous or intermittent line/band running along the weld centerline.
2. porosity: Scattered small dark circular spots within the weld area.
3. cracks: Dark, sharp, irregular jagged lines in the weld.
4. no_defect: Clean, uniform weld with no distinct dark spots, lines, or bands.""",
    },
}

COT_PER_IMAGE = 3


def get_unconditioned_response(client, image_path, prompt):
    """Ask Gemini 2.5 Pro to classify freely — it doesn't know the correct answer."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=512,
        ),
    )
    return response.text.strip()


def generate_unconditioned_data(task_name, config):
    """Generate unconditioned CoT training data using Gemini 2.5 Pro."""
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

    with open(str(config["direct_jsonl"])) as f:
        direct_examples = [json.loads(line) for line in f]

    base_dir = os.path.dirname(str(config["direct_jsonl"]))
    output_path = str(config["output_jsonl"])
    classification_prompt = config["classification_prompt"]
    training_prompt = config["training_prompt"]

    unconditioned_data = []
    total_images = len(direct_examples)

    print(f"\n{'='*60}")
    print(f"  {task_name.upper()} — Generating unconditioned CoT via {MODEL_ID}")
    print(f"  {total_images} images × {COT_PER_IMAGE} responses = {total_images * COT_PER_IMAGE} examples")
    print(f"{'='*60}")

    for idx, example in enumerate(direct_examples):
        img_path = None
        for content in example['messages'][0]['content']:
            if content['type'] == 'image':
                img_path = content['image']
                break

        if img_path and not os.path.isabs(img_path):
            img_path = os.path.normpath(os.path.join(base_dir, img_path))

        print(f'  [{idx+1}/{total_images}] {os.path.basename(img_path)}', end='', flush=True)

        for j in range(COT_PER_IMAGE):
            try:
                response = get_unconditioned_response(client, img_path, classification_prompt)
                unconditioned_data.append({
                    'messages': [
                        {'role': 'user', 'content': [
                            {'type': 'image', 'image': img_path},
                            {'type': 'text', 'text': training_prompt}
                        ]},
                        {'role': 'assistant', 'content': response}
                    ]
                })
                print('.', end='', flush=True)
            except Exception as e:
                print(f'X({e})', end='', flush=True)
            time.sleep(0.5)  # rate limit buffer

        # Add 1 direct example (correct JSON label from original)
        unconditioned_data.append(example)
        print(' done')

    with open(output_path, 'w') as f:
        for ex in unconditioned_data:
            f.write(json.dumps(ex) + '\n')

    print(f"\n  Saved {len(unconditioned_data)} examples to {output_path}")


if __name__ == "__main__":
    for task_name, config in TASKS.items():
        output_path = config["output_jsonl"]
        if output_path.exists():
            print(f"[SKIP] {task_name}: {output_path} already exists")
            continue
        generate_unconditioned_data(task_name, config)

    print("\nDone. Ready for training with run_unconditioned_cot.py")
