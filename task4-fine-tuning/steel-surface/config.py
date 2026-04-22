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
    """The ONE prompt used everywhere — training and evaluation, both approaches.
    Uses the same detailed definitions that GPT-4.1 scored 91.1% with."""
    return """Classify this steel surface defect image.

The image shows a 200×200 pixel grayscale photograph of a hot-rolled steel strip surface.

Possible defect classes:
1. crazing: a network of fine, shallow cracks spreading across the surface in a web-like or mosaic pattern. The cracks are thin, irregular, and multi-directional. Overall texture looks fragmented but the surface is relatively uniform in brightness.
2. inclusion: dark, irregularly shaped spots or elongated streaks embedded in the steel surface. These are foreign material (slag, oxide) trapped during solidification. The background is darker and more uniform than other classes.
3. patches: large irregular regions where the surface texture or brightness changes abruptly. You see distinct lighter or darker zones with soft, blotchy boundaries. The contrast between zones is high.
4. pitted_surface: scattered small dark holes or shallow depressions across a lighter background. The pits are roughly circular and distributed somewhat randomly. The overall surface appears brighter than most other classes.
5. rolled-in_scale: oxide scale that was pressed into the surface during the hot-rolling process. Appears as elongated dark marks, streaks, or patches aligned roughly parallel to the rolling direction. The marks have irregular edges.
6. scratches: one or more linear grooves or marks on the surface, typically running in a consistent direction. The lines are sharper and more defined than crazing cracks, and usually fewer in number.

Respond with ONLY a JSON object:
{"defect_class": "<crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches>"}"""


def make_seal_prompt(defect_class: str) -> str:
    """Prompt for GPT-4.1 to generate answer-conditioned description ONLY (no JSON).
    Uses the same detailed definitions as make_prompt()."""
    class_descriptions = {
        "crazing": "a network of fine, shallow cracks spreading across the surface in a web-like or mosaic pattern. The cracks are thin, irregular, and multi-directional. Overall texture looks fragmented but the surface is relatively uniform in brightness",
        "inclusion": "dark, irregularly shaped spots or elongated streaks embedded in the steel surface. These are foreign material (slag, oxide) trapped during solidification. The background is darker and more uniform than other classes",
        "patches": "large irregular regions where the surface texture or brightness changes abruptly. Distinct lighter or darker zones with soft, blotchy boundaries. The contrast between zones is high",
        "pitted_surface": "scattered small dark holes or shallow depressions across a lighter background. The pits are roughly circular and distributed somewhat randomly. The overall surface appears brighter than most other classes",
        "rolled-in_scale": "oxide scale that was pressed into the surface during the hot-rolling process. Appears as elongated dark marks, streaks, or patches aligned roughly parallel to the rolling direction. The marks have irregular edges",
        "scratches": "one or more linear grooves or marks on the surface, typically running in a consistent direction. The lines are sharper and more defined than crazing cracks, and usually fewer in number",
    }
    desc = class_descriptions.get(defect_class, defect_class)

    return f"""Look at this hot-rolled steel strip surface image. The correct defect classification is: {defect_class}
Definition: {desc}

Key distinguishing features between similar classes:
- crazing vs scratches: crazing has a NETWORK of many fine multi-directional cracks (web-like), while scratches are FEW sharp lines in ONE direction
- inclusion vs rolled-in_scale: inclusion appears as dark SPOTS or BLOBS (foreign material trapped during solidification), while rolled-in_scale appears as ELONGATED dark marks/streaks aligned with the rolling direction
- inclusion vs scratches: inclusion has irregularly shaped dark regions on a darker uniform background, while scratches are sharp linear grooves on a lighter surface
- patches vs pitted_surface: patches are LARGE blotchy zones of brightness change, while pitted_surface has SMALL circular dark holes scattered on a brighter surface

Explain WHY this classification is correct based on what you see:
1. Describe the dominant visual pattern — what shapes, textures, brightness patterns do you observe?
2. Explain why these features match {defect_class} using the definition above
3. Explain why it is NOT the most similar class (use the distinguishing features above)

Write 2-3 sentences justifying the classification. Be specific about what you see in THIS image.
Reply with ONLY the justification text. No JSON, no labels, just the explanation."""
