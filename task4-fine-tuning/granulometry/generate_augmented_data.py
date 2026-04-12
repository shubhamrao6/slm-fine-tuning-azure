"""
SEAL-inspired data augmentation for Approach B.

Uses a frontier vision model (Claude Opus 4.6 or GPT-5) to generate
multiple training variations from each of the 18 training images.

For each image, the frontier model generates:
1. Direct classification (JSON)
2. Chain-of-thought reasoning leading to classification
3. Visual feature description (what makes this class distinct)
4. Contrastive explanation (why this class, not another)
5. Multiple prompt variations (with/without GSD, short/long)

This produces ~8 training pairs per image = ~144 total from 18 images.

Usage:
    # Using Claude (recommended)
    export ANTHROPIC_API_KEY=your-key
    python generate_augmented_data.py --provider anthropic

    # Using OpenAI
    export OPENAI_API_KEY=your-key
    python generate_augmented_data.py --provider openai
"""
import os, sys, json, base64, random, time

TRAIN_DIR = "../../datasets/granulometry/train"
TRAIN_MANIFEST = "../../datasets/granulometry/train_manifest.json"
DIRECT_JSONL = "training_data_direct.jsonl"
OUTPUT = "training_data_augmented.jsonl"
IMAGES_PER_CLASS = 2
SEED = 42

GRADING_DEFS = """Grading (DIN 1045) describes particle size distribution:
- Coarse (A): most particles similar size, close to max. Uniform texture. Few small particles.
- Medium (B): moderate mix of large and small.
- Fine (C): wide range of sizes. Small particles fill gaps between big ones. Dense, packed texture."""

AUGMENTATION_PROMPT = """You are helping create training data for a vision model that classifies concrete aggregate.

This image is class {cls} with:
- max_particle_size_mm: {size}
- grading: {grading}

{grading_defs}

GSD (ground sampling distance) = 8.0 pixels per mm at original resolution.

Generate exactly 8 training examples as a JSON array. Each example has a "prompt" (what the user asks) and "response" (what the model should answer). Vary the prompts and response styles:

1. Direct JSON classification (short prompt, JSON-only response)
2. Direct JSON with GSD mentioned in prompt
3. Chain-of-thought: prompt asks to think step by step, response shows reasoning then gives JSON
4. Visual description: prompt asks what's in the image, response describes particles and concludes with classification
5. Contrastive: prompt asks to classify, response explains why it's {grading} and not the other gradings
6. Size-focused: prompt asks about particle sizes, response estimates sizes and concludes with max size
7. Grading-focused: prompt asks about distribution, response describes the distribution pattern
8. Minimal: very short prompt ("classify"), response is just the JSON

For chain-of-thought responses, end with the JSON on its own line.
All responses must include the correct classification: max_particle_size_mm={size}, grading="{grading}".

Return ONLY the JSON array, no other text."""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_anthropic(image_b64, prompt, model="claude-sonnet-4-20250514"):
    """Call Claude API with image."""
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return msg.content[0].text


def call_openai(image_b64, prompt, model="gpt-5"):
    """Call OpenAI API with image."""
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=4096,
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
    """Extract JSON array from model response."""
    import re
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
    provider = "anthropic"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--provider" and i + 1 < len(sys.argv) - 1:
            provider = sys.argv[i + 2]

    call_fn = call_anthropic if provider == "anthropic" else call_openai
    print(f"Provider: {provider}")

    random.seed(SEED)
    with open(TRAIN_MANIFEST) as f:
        manifest = json.load(f)

    # Use same images as Approach A
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

        print(f"  Generating for {entry['class']}: {entry['image']}...", end=" ", flush=True)
        image_b64 = encode_image(img_path)

        try:
            raw = call_fn(image_b64, prompt)
            examples = parse_augmented(raw)
            if examples and isinstance(examples, list):
                # Convert to training format
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

        time.sleep(1)  # rate limiting

    # Write JSONL
    with open(OUTPUT, "w") as f:
        for record in all_examples:
            f.write(json.dumps(record) + "\n")

    print(f"\nTotal: {len(all_examples)} augmented examples from {len(selected)} images")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
