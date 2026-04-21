"""
Shared configuration for RIAWELC Weld Defect benchmarking.
"""
import os

# Paths
DATASET_ROOT = os.environ.get("DATASET_DIR", "../../datasets/riawelc")
TEST_IMAGES_DIR = os.path.join(DATASET_ROOT, "testing")
TRAIN_IMAGES_DIR = os.path.join(DATASET_ROOT, "training")
REF_IMAGE_PATH = "riawelc_reference_grid.png"

# Model
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Classes (4 weld defect types)
CLASSES = ["lack_of_penetration", "porosity", "cracks", "no_defect"]

# Azure OpenAI (frontier model)
AZURE_ENDPOINT = "https://ether-openai.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-4.1"
AZURE_API_VERSION = "2024-12-01-preview"


def make_prompt_zs() -> str:
    """Zero-shot prompt for weld defect classification."""
    return """Classify this weld radiographic image.

This is a 227×227 pixel 8-bit grayscale radiographic (X-ray) image of a weld joint. Radiography reveals internal defects as variations in image intensity — defects appear as darker or lighter regions compared to the surrounding sound weld metal.

Possible defect classes:
1. lack_of_penetration: A dark continuous or intermittent line/band running along the weld centerline. This indicates the weld root was not fully fused. The dark line is relatively straight and follows the joint geometry.
2. porosity: Scattered small dark circular spots within the weld area. Gas pores are characteristically round or oval. They can appear as individual dots or in clusters. The key feature is the circular shape of each dark spot.
3. cracks: Dark, sharp, irregular jagged lines in the weld. Thinner and more erratic than lack of penetration. May branch or change direction. The edges are sharp and the line path is irregular.
4. no_defect: Clean, uniform weld with no distinct dark spots, lines, or bands. The radiographic image shows relatively even gray intensity across the weld region.

Respond with ONLY a JSON object:
{"defect_class": "<lack_of_penetration|porosity|cracks|no_defect>"}"""


def make_prompt_fs_ref() -> str:
    """Few-shot reference image prompt — describes the 4×1 grid layout."""
    return """First image: a 4×1 reference grid showing one example of each weld defect class from radiographic images.

From LEFT to RIGHT:
  (1) lack_of_penetration — A dark continuous or intermittent line/band running along the weld centerline. This indicates the weld root was not fully fused. The dark line is relatively straight and follows the joint geometry.
  (2) porosity — Scattered small dark circular spots within the weld area. Gas pores are characteristically round or oval. They can appear as individual dots or in clusters. The key feature is the circular shape of each dark spot.
  (3) cracks — Dark, sharp, irregular jagged lines in the weld. Thinner and more erratic than lack of penetration. May branch or change direction. The edges are sharp and the line path is irregular.
  (4) no_defect — Clean, uniform weld with no distinct dark spots, lines, or bands. The radiographic image shows relatively even gray intensity across the weld region.

Focus on the shape of dark features: lines (LP), circles (porosity), jagged lines (cracks), or uniform (no defect)."""


def make_prompt_fs_query() -> str:
    """Few-shot query prompt."""
    return """Now classify this weld radiograph by comparing to the 4 reference examples.

Identify the dominant dark feature:
- Dark straight line/band along weld center → lack_of_penetration
- Scattered dark circular spots/dots → porosity
- Dark sharp jagged irregular lines → cracks
- Uniform gray, no distinct dark features → no_defect

Respond with ONLY JSON: {"defect_class": "<lack_of_penetration|porosity|cracks|no_defect>"}"""
