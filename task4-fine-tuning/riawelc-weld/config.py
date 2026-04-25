"""
Shared configuration for Task 4 — RIAWELC Weld Defect fine-tuning.
Used by both direct and augmented notebooks.

Prompts use the SAME detailed definitions as the task3 benchmarking config
(the ones GPT-4.1 scored 65.0% FS with).
"""
import os

# Paths
DATASET_ROOT = os.environ.get("DATASET_DIR", "../../datasets/riawelc")
TRAIN_IMAGES_DIR = os.path.join(DATASET_ROOT, "training")
TEST_IMAGES_DIR = os.path.join(DATASET_ROOT, "testing")

# Model
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SEED = 42
IMAGES_PER_CLASS = 6  # 6 per class × 4 classes = 24 training images

# Classes (4 weld defect types)
CLASSES = ["lack_of_penetration", "porosity", "cracks", "no_defect"]

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Azure OpenAI (for augmented notebook — GPT-4.1 as SEAL teacher)
AZURE_ENDPOINT = "https://ether-openai.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-4.1"
AZURE_API_VERSION = "2024-12-01-preview"


def make_prompt() -> str:
    """The ONE prompt used everywhere — training and evaluation, both approaches.
    Identical to the task3 benchmarking ZS prompt."""
    return """Classify this weld radiographic image.

This is a 227×227 pixel 8-bit grayscale radiographic (X-ray) image of a weld joint. Radiography reveals internal defects as variations in image intensity — defects appear as darker or lighter regions compared to the surrounding sound weld metal.

Possible defect classes:
1. lack_of_penetration: A dark continuous or intermittent line/band running along the weld centerline. This indicates the weld root was not fully fused. The dark line is relatively straight and follows the joint geometry.
2. porosity: Scattered small dark circular spots within the weld area. Gas pores are characteristically round or oval. They can appear as individual dots or in clusters. The key feature is the circular shape of each dark spot.
3. cracks: Dark, sharp, irregular jagged lines in the weld. Thinner and more erratic than lack of penetration. May branch or change direction. The edges are sharp and the line path is irregular.
4. no_defect: Clean, uniform weld with no distinct dark spots, lines, or bands. The radiographic image shows relatively even gray intensity across the weld region.

Respond with ONLY a JSON object:
{"defect_class": "<lack_of_penetration|porosity|cracks|no_defect>"}"""


def make_seal_prompt(defect_class: str) -> str:
    """Prompt for GPT-4.1 to generate answer-conditioned description ONLY (no JSON).
    Uses the same detailed definitions as make_prompt()."""
    class_descriptions = {
        "lack_of_penetration": "A dark continuous or intermittent line/band running along the weld centerline. This indicates the weld root was not fully fused. The dark line is relatively straight and follows the joint geometry",
        "porosity": "Scattered small dark circular spots within the weld area. Gas pores are characteristically round or oval. They can appear as individual dots or in clusters. The key feature is the circular shape of each dark spot",
        "cracks": "Dark, sharp, irregular jagged lines in the weld. Thinner and more erratic than lack of penetration. May branch or change direction. The edges are sharp and the line path is irregular",
        "no_defect": "Clean, uniform weld with no distinct dark spots, lines, or bands. The radiographic image shows relatively even gray intensity across the weld region",
    }
    desc = class_descriptions.get(defect_class, defect_class)

    return f"""Look at this weld radiographic (X-ray) image. The correct defect classification is: {defect_class}
Definition: {desc}

Key distinguishing features between similar classes:
- lack_of_penetration vs cracks: lack_of_penetration has a STRAIGHT dark line/band along the weld CENTER, while cracks are JAGGED irregular lines that may branch or change direction
- lack_of_penetration vs no_defect: lack_of_penetration has a distinct dark line along the centerline, no_defect has uniform gray with no distinct features
- porosity vs cracks: porosity has CIRCULAR dark spots (round/oval dots), cracks have LINEAR dark features (sharp jagged lines)
- porosity vs no_defect: porosity has scattered dark circular spots, no_defect has uniform intensity with no spots

Explain WHY this classification is correct based on what you see:
1. Describe the dominant dark features — what shapes do you observe? (lines, circles, jagged marks, or uniform)
2. Explain why these features match {defect_class} using the definition above
3. Explain why it is NOT the most similar class

Write 2-3 sentences justifying the classification. Be specific about what you see in THIS image.
Reply with ONLY the justification text. No JSON, no labels, just the explanation."""
