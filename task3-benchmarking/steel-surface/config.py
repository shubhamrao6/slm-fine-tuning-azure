"""
Shared configuration for Steel Surface Defect benchmarking.
"""
import os

# Paths
DATASET_ROOT = os.environ.get("DATASET_DIR", "../../datasets/neu-cls/NEU-DET")
TRAIN_IMAGES_DIR = os.path.join(DATASET_ROOT, "train", "images")
TEST_IMAGES_DIR = os.path.join(DATASET_ROOT, "validation", "images")  # validation split = our test set
TEST_MANIFEST = os.path.join(DATASET_ROOT, "..", "test_manifest.json")
REF_IMAGE_PATH = "../Sample-images-in-the-NEU-CLS-dataset.png"

# Model
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ORIGINAL_RES = 200  # NEU images are 200x200
MAX_DIM = 200  # no resize needed — already small

# Classes (folder name uses hyphen: "rolled-in_scale")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_CODES = {"Cr": "crazing", "In": "inclusion", "Pa": "patches",
               "PS": "pitted_surface", "RS": "rolled-in_scale", "Sc": "scratches"}

# Azure OpenAI (frontier model)
AZURE_ENDPOINT = "https://ether-openai.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-4.1"
AZURE_API_VERSION = "2024-12-01-preview"

# Ground truth
GT_CLASSES = {
    "crazing": "crazing",
    "inclusion": "inclusion",
    "patches": "patches",
    "pitted_surface": "pitted_surface",
    "rolled-in_scale": "rolled-in_scale",
    "scratches": "scratches",
}


def make_prompt_zs() -> str:
    """Zero-shot prompt for steel surface defect classification."""
    return """Classify this steel surface defect image.

The image shows a 200×200 pixel grayscale photograph of a hot-rolled steel strip surface.

Possible defect classes:
1. CRAZING: network of fine surface cracks forming a web-like pattern
2. INCLUSION: dark spots or streaks of foreign material embedded in the surface
3. PATCHES: irregular lighter or darker areas, uneven texture regions
4. PITTED_SURFACE: small holes or depressions scattered across the surface
5. ROLLED-IN_SCALE: oxide scale pressed into surface during rolling, elongated marks parallel to rolling direction
6. SCRATCHES: linear marks or grooves on the surface, directional damage

Respond with ONLY a JSON object:
{"defect_class": "<crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches>"}"""


def make_prompt_fs_ref() -> str:
    """Few-shot reference image prompt."""
    return """First image: reference chart showing examples of all 6 steel surface defect classes from the NEU dataset.

The 6 classes are:
- CRAZING (Cr): web-like network of fine cracks
- INCLUSION (In): dark embedded foreign material spots/streaks
- PATCHES (Pa): irregular lighter/darker texture areas
- PITTED SURFACE (PS): scattered small holes/depressions
- ROLLED-IN SCALE (RS): elongated oxide marks from rolling process
- SCRATCHES (Sc): linear directional grooves/marks

Study the visual differences between each class carefully."""


def make_prompt_fs_query() -> str:
    """Few-shot query prompt."""
    return """Classify this steel surface defect image by comparing to the reference.

Which of the 6 classes does it match?
- crazing, inclusion, patches, pitted_surface, rolled-in_scale, or scratches

Respond with ONLY JSON: {"defect_class": "<crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches>"}"""
