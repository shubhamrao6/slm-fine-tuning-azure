"""
Prepare training data for Approach A (Standard LoRA).
Selects 2 images per class (18 total) from the train split and creates JSONL.
"""
import os, json, random

TRAIN_DIR = "../../datasets/granulometry/train"
TRAIN_MANIFEST = "../../datasets/granulometry/train_manifest.json"
OUTPUT = "training_data_direct.jsonl"
IMAGES_PER_CLASS = 2
SEED = 42

PROMPT = (
    "Classify this concrete aggregate photo. GSD = 8.0 px/mm.\n"
    "Grading describes size distribution: coarse = uniform, mostly large; "
    "medium = moderate mix; fine = many small particles filling gaps, dense and packed.\n"
    'Respond with ONLY JSON: {"max_particle_size_mm": <8|16|32>, "grading": "<coarse|medium|fine>"}'
)

def main():
    random.seed(SEED)
    with open(TRAIN_MANIFEST) as f:
        manifest = json.load(f)

    # Group by class
    by_class = {}
    for entry in manifest:
        cls = entry["class"]
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(entry)

    selected = []
    for cls in sorted(by_class):
        entries = by_class[cls]
        random.shuffle(entries)
        picked = entries[:IMAGES_PER_CLASS]
        selected.extend(picked)
        print(f"  {cls}: {[e['image'] for e in picked]}")

    # Write JSONL
    with open(OUTPUT, "w") as f:
        for entry in selected:
            img_path = os.path.join(TRAIN_DIR, entry["image"])
            gt = json.dumps({
                "max_particle_size_mm": entry["max_particle_size_mm"],
                "grading": entry["grading"],
            })
            record = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": PROMPT},
                    ]},
                    {"role": "assistant", "content": gt},
                ]
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(selected)} examples to {OUTPUT}")
    print(f"Classes: {sorted(by_class.keys())}")


if __name__ == "__main__":
    main()
