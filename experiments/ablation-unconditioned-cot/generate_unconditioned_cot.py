"""
Generate Unconditioned CoT: GPT-4.1 classifies training images freely (no answer provided).

This produces training data where GPT-4.1's OWN prediction (often wrong) is the label.
For granulometry, GPT-4.1 gets ~29.6% correct, so ~70% of training data will have wrong reasoning.
"""
import os
import sys
import json
import time
import base64
from pathlib import Path
from openai import AzureOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Azure OpenAI config
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://ether-openai.openai.azure.com/")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_API_VERSION = "2024-12-01-preview"

# Models to test (both unconditioned)
MODELS = {
    "gpt-4.1": "gpt-4.1",
    "gpt-5": "gpt-5",  # deployment name — update if different
}

# Tasks config
TASKS = {
    "granulometry": {
        "direct_jsonl": REPO_ROOT / "task4-fine-tuning" / "granulometry" / "training_data_direct.jsonl",
        "output_template": str(SCRIPT_DIR / "granulometry_unconditioned_{model}.jsonl"),
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
        "output_template": str(SCRIPT_DIR / "weld_unconditioned_{model}.jsonl"),
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

COT_PER_IMAGE = 3  # Generate 3 unconditioned responses per image (like conditioned)


def encode_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_unconditioned_response(client, image_path, prompt, model_deployment):
    """Ask a frontier model to classify freely — it doesn't know the correct answer."""
    img_b64 = encode_image(image_path)

    # Determine image type from extension
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"

    resp = client.chat.completions.create(
        model=model_deployment,
        temperature=0.7,
        max_tokens=512,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{img_b64}'}},
                {'type': 'text', 'text': prompt},
            ]
        }]
    )
    return resp.choices[0].message.content.strip()


def generate_unconditioned_data(task_name, config, model_name, model_deployment):
    """Generate unconditioned CoT training data for a task using a specific model."""
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

    # Load original direct JSONL to get image paths
    with open(str(config["direct_jsonl"])) as f:
        direct_examples = [json.loads(line) for line in f]

    base_dir = os.path.dirname(str(config["direct_jsonl"]))
    output_path = config["output_template"].format(model=model_name.replace(".", ""))
    classification_prompt = config["classification_prompt"]
    training_prompt = config["training_prompt"]

    unconditioned_data = []
    total_images = len(direct_examples)

    print(f"\n{'='*60}")
    print(f"  {task_name.upper()} — Generating unconditioned CoT via {model_name}")
    print(f"  {total_images} images × {COT_PER_IMAGE} responses = {total_images * COT_PER_IMAGE} examples")
    print(f"{'='*60}")

    for idx, example in enumerate(direct_examples):
        # Extract image path from JSONL
        img_path = None
        for content in example['messages'][0]['content']:
            if content['type'] == 'image':
                img_path = content['image']
                break

        if img_path and not os.path.isabs(img_path):
            img_path = os.path.normpath(os.path.join(base_dir, img_path))

        print(f'  [{idx+1}/{total_images}] {os.path.basename(img_path)}', end='', flush=True)

        # Generate COT_PER_IMAGE unconditioned responses
        for j in range(COT_PER_IMAGE):
            try:
                gpt_response = get_unconditioned_response(client, img_path, classification_prompt, model_deployment)

                # Training pair: image + prompt (no JSON instruction) → model's full response
                unconditioned_data.append({
                    'messages': [
                        {'role': 'user', 'content': [
                            {'type': 'image', 'image': img_path},
                            {'type': 'text', 'text': training_prompt}
                        ]},
                        {'role': 'assistant', 'content': gpt_response}
                    ]
                })
                print('.', end='', flush=True)
            except Exception as e:
                print(f'X({e})', end='', flush=True)
            time.sleep(0.3)

        # Also add 1 direct example (same as conditioned approach)
        unconditioned_data.append(example)
        print(' done')

    # Save
    with open(output_path, 'w') as f:
        for ex in unconditioned_data:
            f.write(json.dumps(ex) + '\n')

    print(f"\n  Saved {len(unconditioned_data)} examples to {output_path}")
    return unconditioned_data


if __name__ == "__main__":
    if not AZURE_API_KEY:
        print("ERROR: Set AZURE_OPENAI_KEY environment variable")
        print("  export AZURE_OPENAI_KEY='your-key-here'")
        sys.exit(1)

    for model_name, model_deployment in MODELS.items():
        print(f"\n\n{'#'*70}")
        print(f"  MODEL: {model_name}")
        print(f"{'#'*70}")

        for task_name, config in TASKS.items():
            output_path = config["output_template"].format(model=model_name.replace(".", ""))
            if os.path.exists(output_path):
                print(f"[SKIP] {task_name}/{model_name}: {output_path} already exists")
                continue
            generate_unconditioned_data(task_name, config, model_name, model_deployment)

    print("\nDone. Ready for training with run_unconditioned_cot.py")
