"""
Shared configuration for Steel Surface Defect benchmarking.
"""
import os

# Paths
DATASET_ROOT = os.environ.get("DATASET_DIR", "../../datasets/neu-cls/NEU-DET")
TRAIN_IMAGES_DIR = os.path.join(DATASET_ROOT, "train", "images")
TEST_IMAGES_DIR = os.path.join(DATASET_ROOT, "validation", "images")  # validation split = our test set
TEST_MANIFEST = os.path.join(DATASET_ROOT, "..", "test_manifest.json")
REF_IMAGE_PATH = "Sample-images-in-the-NEU-CLS-dataset.png"

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
    """Few-shot reference image prompt — describes the actual grid layout."""
    return """First image: a 2×3 reference grid showing one example of each of the 6 steel surface defect classes from the NEU dataset. The layout is:

TOP ROW (left to right):
  (a) CRAZING — a network of fine, shallow cracks spreading across the surface in a web-like or mosaic pattern. The cracks are thin, irregular, and multi-directional. Overall texture looks fragmented but the surface is relatively uniform in brightness.
  (b) INCLUSION — dark, irregularly shaped spots or elongated streaks embedded in the steel surface. These are foreign material (slag, oxide) trapped during solidification. The background is darker and more uniform than other classes.
  (c) PATCHES — large irregular regions where the surface texture or brightness changes abruptly. You see distinct lighter or darker zones with soft, blotchy boundaries. The contrast between zones is high.

BOTTOM ROW (left to right):
  (d) PITTED SURFACE — scattered small dark holes or shallow depressions across a lighter background. The pits are roughly circular and distributed somewhat randomly. The overall surface appears brighter than most other classes.
  (e) ROLLED-IN SCALE — oxide scale that was pressed into the surface during the hot-rolling process. Appears as elongated dark marks, streaks, or patches aligned roughly parallel to the rolling direction. The marks have irregular edges.
  (f) SCRATCHES — one or more linear grooves or marks on the surface, typically running in a consistent direction. The lines are sharper and more defined than crazing cracks, and usually fewer in number.

Study the visual texture, pattern geometry, and brightness differences between each class carefully."""


def make_prompt_fs_query() -> str:
    """Few-shot query prompt."""
    return """Now classify this steel surface defect image by comparing it to the 6 examples in the reference grid.

Look at the dominant visual pattern:
- Fine web-like crack network → crazing
- Dark embedded spots/streaks on uniform background → inclusion
- Large blotchy lighter/darker regions → patches
- Scattered small dark holes on bright surface → pitted_surface
- Elongated dark marks aligned with rolling direction → rolled-in_scale
- Sharp linear grooves in one direction → scratches

Respond with ONLY JSON: {"defect_class": "<crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches>"}"""
