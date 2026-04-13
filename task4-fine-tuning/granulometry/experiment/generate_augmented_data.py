"""
SEAL-inspired data augmentation for Approach B.

Uses GPT-4.1 (Azure OpenAI) to generate multiple training variations
from each of the 18 training images.

For each image, GPT-4.1 generates 8 training examples with varied styles:
1. Direct JSON classification
2. Direct JSON with GSD in prompt
3. Chain-of-thought reasoning then JSON
4. Visual feature description concluding with classification
5. Contrastive explanation (why this grading, not others)
6. Size-focused analysis
7. Grading-focused analysis (gaps visual test)
8. Minimal prompt, JSON response

This produces ~144 training pairs from 18 images.

Usage:
    export AZURE_OPENAI_API_KEY=your-key
    python generate_augmented_data.py
"""
import os, sys, json, re, base64, random, time
from openai import AzureOpenAI

TRAIN_DIR = "../../datasets/granulometry/train"
TRAIN_MANIFEST = "../../datasets/granulometry/train_manifest.json"
OUTPUT = "training_data_augmented.jsonl"
IMAGES_PER_CLASS = 2
SEED = 42

# Azure OpenAI — GPT-4.1
AZURE_ENDPOINT = "https://ether-openai.openai.azure.com/"
DEPLOYMENT = "gpt-4.1"
API_VERSION = "2024-12-01-preview"

GRADING_DEFS = """Grading follows DIN 1045 standard curves A/B/C. It describes the SHAPE of the particle
size distribution — independent of max particle size.

COARSE (curve A / uniformly graded):
- Most particles concentrated near the maximum size
- Very few small particles present
- Large visible gaps/voids between stones (not filled by smaller material)
- Surface appears as a single layer of similarly-sized stones
- Low packing density — background visible between particles

MEDIUM (curve B / well-graded):
- Balanced mix of particle sizes from small to large
- Some smaller particles fill gaps between larger ones, but not completely
- Moderate packing density

FINE (curve C / continuously graded):
- Wide range of particle sizes present simultaneously
- Many small particles densely fill ALL gaps between larger stones
- Very high packing density — almost no visible voids
- Surface appears tightly packed, dense, and heterogeneous

KEY VISUAL TEST: Look at spaces between the largest stones.
- Gaps EMPTY → coarse (A)
- Gaps PARTIALLY filled → medium (B)
- Gaps COMPLETELY filled with smaller particles → fine (C)"""

AUGMENTATION_PROMPT = """You are helping create training data for a small vision model (Qwen2.5-VL-3B)
that classifies concrete aggregate images.

This image is class {cls} with:
- max_particle_size_mm: {size}
- grading: {grading}

{grading_defs}

GSD = 8.0 pixels per mm. At this resolution: 8mm = ~64px, 16mm = ~128px, 32mm = ~256px.

Generate exactly 8 training examples as a JSON array. Each has "prompt" and "response".
Vary the styles:

1. Direct JSON (short prompt asking to classify, JSON-only response)
2. Direct JSON with GSD and pixel hints in prompt
3. Chain-of-thought: prompt asks to think step by step, response shows reasoning about
   particle sizes and gap-filling pattern, then gives JSON
4. Visual description: prompt asks what's in the image, response describes the particles,
   their sizes, the gaps between them, and concludes with classification
5. Contrastive: response explains why it's {grading} and not the other gradings,
   specifically referencing the gaps visual test
6. Size-focused: response estimates largest particle in pixels, converts to mm using GSD
7. Grading-focused: response describes the distribution pattern and packing density
8. Minimal: very short prompt ("classify"), just JSON response

CRITICAL: All responses must include the correct classification:
max_particle_size_mm={size}, grading="{grading}"

Return ONLY the JSON array, no other text."""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_gpt41(client, image_b64, prompt):
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        max_tokens=4096,
        temperature=0.7,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return resp.choices[0].message.content


def parse_augmented(raw_text):
    raw_text = re.sub(r'```json\s*', '', raw_text)
    raw_text = re.sub(r'```\s*', '', raw_text).strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except:
                pass
    return None


def main():
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not api_key:
        print("Error: Set AZURE_OPENAI_API_KEY environment variable")
        sys.exit(1)

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=api_key,
        api_version=API_VERSION,
    )
    print(f"Azure OpenAI: {AZURE_ENDPOINT}, deployment: {DEPLOYMENT}")

    random.seed(SEED)
    with open(TRAIN_MANIFEST) as f:
        manifest = json.load(f)

    by_class = {}
    for entry in manifest:
        cls = entry["class"]
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(entry)

    selected = []
    for cls in sorted(by_class):
        random.shuffle(by_class[cls])
        selected.extend(by_class[cls][:IMAGES_PER_CLASS])

    print(f"Selected {len(selected)} training images ({IMAGES_PER_CLASS} per class)")

    all_examples = []
    for entry in selected:
        img_path = os.path.join(TRAIN_DIR, entry["image"])
        if not os.path.exists(img_path):
            print(f"  Skipping {entry['image']} (not found)")
            continue

        prompt = AUGMENTATION_PROMPT.format(
            cls=entry["class"],
            size=entry["max_particle_size_mm"],
            grading=entry["grading"],
            grading_defs=GRADING_DEFS,
        )

        print(f"  {entry['class']}: {entry['image']}...", end=" ", flush=True)
        image_b64 = encode_image(img_path)

        try:
            raw = call_gpt41(client, image_b64, prompt)
            examples = parse_augmented(raw)
            if examples and isinstance(examples, list):
                for ex in examples:
                    record = {
                        "messages": [
                            {"role": "user", "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": ex["prompt"]},
                            ]},
                            {"role": "assistant", "content": ex["response"]},
                        ]
                    }
                    all_examples.append(record)
                print(f"{len(examples)} examples")
            else:
                print("PARSE FAILED")
        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(0.5)

    with open(OUTPUT, "w") as f:
        for record in all_examples:
            f.write(json.dumps(record) + "\n")

    print(f"\nTotal: {len(all_examples)} augmented examples from {len(selected)} images")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
