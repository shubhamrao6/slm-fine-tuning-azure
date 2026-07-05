"""
Create 4× duplicated JSONL files for the equal-budget ablation.

Each original training example is repeated 4 times, giving the same
total example count as CoT-Aug (which has 1 direct + 3 CoT = 4 per image).
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

TASKS = {
    "granulometry": {
        "input": REPO_ROOT / "task4-fine-tuning" / "granulometry" / "training_data_direct.jsonl",
        "output": Path(__file__).resolve().parent / "granulometry_direct_4x.jsonl",
        "duplicates": 4,  # 18 × 4 = 72 (matches CoT-Aug)
    },
    "steel_surface": {
        "input": REPO_ROOT / "task4-fine-tuning" / "steel-surface" / "training_data_direct.jsonl",
        "output": Path(__file__).resolve().parent / "steel_surface_direct_4x.jsonl",
        "duplicates": 4,  # 30 × 4 = 120 (matches CoT-Aug)
    },
    "uhcs": {
        "input": REPO_ROOT / "task4-fine-tuning" / "uhcs-microstructure" / "training_data_direct.jsonl",
        "output": Path(__file__).resolve().parent / "uhcs_direct_4x.jsonl",
        "duplicates": 4,  # 30 × 4 = 120 (matches CoT-Aug)
    },
    "weld": {
        "input": REPO_ROOT / "task4-fine-tuning" / "riawelc-weld" / "training_data_direct.jsonl",
        "output": Path(__file__).resolve().parent / "weld_direct_4x.jsonl",
        "duplicates": 4,  # 24 × 4 = 96 (matches CoT-Aug)
    },
}


def create_duplicated_jsonl(task_name, config):
    """Create a JSONL file where each example is duplicated N times."""
    input_path = str(config["input"])
    output_path = str(config["output"])
    n_duplicates = config["duplicates"]

    with open(input_path) as f:
        examples = [json.loads(line) for line in f]

    with open(output_path, 'w') as f:
        for example in examples:
            for _ in range(n_duplicates):
                f.write(json.dumps(example) + '\n')

    print(f"{task_name}:")
    print(f"  Original: {len(examples)} examples")
    print(f"  Duplicated: {len(examples) * n_duplicates} examples")
    print(f"  Saved to: {output_path}")
    print()


if __name__ == "__main__":
    for task_name, config in TASKS.items():
        create_duplicated_jsonl(task_name, config)
    print("Done. Ready for training.")
