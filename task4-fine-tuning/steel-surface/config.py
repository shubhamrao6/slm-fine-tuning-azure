"""
Shared configuration for Task 4 — Steel Surface Defect fine-tuning.
Used by both direct and augmented notebooks.
"""
import os

# Paths
DATASET_ROOT = os.environ.get("DATASET_DIR", "../../datasets/neu-cls/NEU-DET")
TRAIN_IMAGES_DIR = os.path.join(DATASET_ROOT, "train", "images")
TEST_IMAGES_DIR = os.path.join(DATASET_ROOT, "validation", "images")

# Model
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SEED = 42
IMAGES_PER_CLASS = 3  # 3 per class × 6 classes = 18 training images

# Classes (folder name uses hyphen: "rolled-in_scale")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

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
    """The ONE prompt used everywhere — training and evaluation, both approaches."""
    return """Classify this steel surface defect image.

The image shows a 200×200 pixel grayscale photograph of a hot-rolled steel strip surface.

Possible defect classes:
1. crazing: network of fine surface cracks forming a web-like pattern. The cracks are thin, irregular, and multi-directional. Surface looks fragmented but relatively uniform in brightness.
2. inclusion: dark spots or streaks of foreign material embedded in the surface. Irregularly shaped dark regions on a darker, more uniform background.
3. patches: irregular lighter or darker areas with soft, blotchy boundaries. Distinct zones where surface texture or brightness changes abruptly. High contrast between zones.
4. pitted_surface: small dark holes or shallow depressions scattered across a lighter background. Pits are roughly circular and randomly distributed.
5. rolled-in_scale: oxide scale pressed into surface during rolling. Elongated dark marks or streaks aligned roughly parallel to rolling direction. Irregular edges.
6. scratches: one or more sharp linear grooves or marks, typically running in a consistent direction. Lines are sharper and more defined than crazing cracks.

Respond with ONLY a JSON object:
{"defect_class": "<crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches>"}"""


def make_seal_prompt(defect_class: str) -> str:
    """Prompt for GPT-4.1 to generate answer-conditioned description ONLY (no JSON)."""
    class_descriptions = {
        "crazing": "network of fine surface cracks forming a web-like or mosaic pattern",
        "inclusion": "dark spots or streaks of foreign material (slag, oxide) embedded in the surface",
        "patches": "irregular lighter or darker areas with soft blotchy boundaries, uneven texture regions",
        "pitted_surface": "small dark holes or shallow depressions scattered across a lighter background",
        "rolled-in_scale": "oxide scale pressed into surface during rolling, elongated dark marks parallel to rolling direction",
        "scratches": "sharp linear grooves or marks running in a consistent direction",
    }
    desc = class_descriptions.get(defect_class, defect_class)

    return f"""Look at this hot-rolled steel strip surface image. The correct defect classification is: {defect_class}
Definition: {desc}

Similar class pairs to contrast:
- crazing vs scratches (both have lines, but crazing = web-like network, scratches = few sharp directional lines)
- inclusion vs rolled-in_scale (both have dark marks, but inclusion = spots/blobs, rolled-in_scale = elongated streaks)
- patches vs pitted_surface (both have brightness variation, but patches = large blotchy zones, pitted_surface = small circular holes)

Explain WHY this classification is correct based on what you see:
1. Describe the dominant visual pattern (texture, shapes, brightness)
2. Explain why these features match {defect_class}
3. Mention why it's NOT one of the similar classes

Write 2-3 sentences justifying the classification. Be specific about what you see in THIS image.
Reply with ONLY the justification text. No JSON, no labels, just the explanation."""
