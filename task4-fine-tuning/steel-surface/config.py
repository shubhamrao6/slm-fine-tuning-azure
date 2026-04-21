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
1. CRAZING: network of fine surface cracks forming a web-like pattern. The cracks are thin, irregular, and multi-directional. Surface looks fragmented but relatively uniform in brightness.
2. INCLUSION: dark spots or streaks of foreign material embedded in the surface. Irregularly shaped dark regions on a darker, more uniform background.
3. PATCHES: irregular lighter or darker areas with soft, blotchy boundaries. Distinct zones where surface texture or brightness changes abruptly. High contrast between zones.
4. PITTED_SURFACE: small dark holes or shallow depressions scattered across a lighter background. Pits are roughly circular and randomly distributed.
5. ROLLED-IN_SCALE: oxide scale pressed into surface during rolling. Elongated dark marks or streaks aligned roughly parallel to rolling direction. Irregular edges.
6. SCRATCHES: one or more sharp linear grooves or marks, typically running in a consistent direction. Lines are sharper and more defined than crazing cracks.

Respond with ONLY a JSON object:
{"defect_class": "<crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches>"}"""


def make_seal_prompt(defect_class: str) -> str:
    """Prompt for GPT-4.1 to generate answer-conditioned CoT descriptions."""
    class_descriptions = {
        "crazing": "network of fine surface cracks forming a web-like or mosaic pattern",
        "inclusion": "dark spots or streaks of foreign material (slag, oxide) embedded in the surface",
        "patches": "irregular lighter or darker areas with soft blotchy boundaries, uneven texture regions",
        "pitted_surface": "small dark holes or shallow depressions scattered across a lighter background",
        "rolled-in_scale": "oxide scale pressed into surface during rolling, elongated dark marks parallel to rolling direction",
        "scratches": "sharp linear grooves or marks running in a consistent direction",
    }
    desc = class_descriptions.get(defect_class, defect_class)

    return f"""You are a steel surface defect analysis expert. This image shows a hot-rolled steel strip surface defect.

The CORRECT classification for this image is: {defect_class}
Definition: {desc}

Your task: Write a detailed visual analysis (3-5 sentences) explaining WHY this image shows {defect_class} based on what you actually see. Then output the JSON classification.

Requirements:
- Describe the specific visual features you observe (texture, pattern geometry, brightness, shapes)
- Explain why these features match {defect_class} and not other classes
- Use contrastive reasoning: mention why it's NOT one of the similar classes
- End with the JSON: {{"defect_class": "{defect_class}"}}

Similar class pairs to contrast:
- crazing vs scratches (both have lines, but crazing = web-like network, scratches = few sharp directional lines)
- inclusion vs rolled-in_scale (both have dark marks, but inclusion = spots/blobs, rolled-in_scale = elongated streaks)
- patches vs pitted_surface (both have brightness variation, but patches = large blotchy zones, pitted_surface = small circular holes)

Write your analysis then the JSON:"""
