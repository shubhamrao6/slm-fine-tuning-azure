"""
Shared configuration for UHCS Microstructure benchmarking.
"""
import os

# Paths
DATASET_ROOT = os.environ.get("DATASET_DIR", "../../datasets/uh-carbon-steel")
IMAGES_DIR = os.path.join(DATASET_ROOT, "For Training", "Cropped")
METADATA_PATH = os.path.join(DATASET_ROOT, "new_metadata.xlsx")
TEST_MANIFEST = os.path.join(DATASET_ROOT, "test_manifest.json")
REF_IMAGE_PATH = "uhcs_reference_grid.png"

# Model
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Classes (6 microconstituent labels)
CLASSES = [
    "spheroidite",
    "network",
    "spheroidite+widmanstatten",
    "pearlite+spheroidite",
    "pearlite",
    "pearlite+widmanstatten",
]

# Azure OpenAI (frontier model)
AZURE_ENDPOINT = "https://ether-openai.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-4.1"
AZURE_API_VERSION = "2024-12-01-preview"


def make_prompt_zs(magnification: str = "unknown") -> str:
    """Zero-shot prompt for UHCS microstructure classification."""
    return f"""Classify this ultra-high carbon steel (UHCS) micrograph.

This is an optical/SEM micrograph at approximately {magnification} magnification showing the microstructure of UHCS after heat treatment.

Possible microconstituent classes:
1. spheroidite: Scattered dark round/oval cementite particles on a light ferrite matrix. The particles are isolated, roughly spherical, and uniformly distributed. Looks like "polka dots." This forms from prolonged annealing below the eutectoid temperature.
2. network: Dark continuous lines forming a connected web/mesh pattern. These are cementite films along prior austenite grain boundaries. The lines outline polygonal grain shapes. Forms during slow cooling from above A1.
3. spheroidite+widmanstatten: A mix of round spheroidized particles AND straight elongated needle/plate-like features growing inward from grain boundaries. You see both "dots" and "needles" in the same image. Indicates partial spheroidization of Widmanstatten cementite.
4. pearlite+spheroidite: Regions showing fingerprint-like lamellar striations (pearlite) alongside areas with scattered round particles (spheroidite). Two distinct textures coexist. Indicates incomplete spheroidization of pearlite.
5. pearlite: Fine parallel alternating dark/light lamellae creating a fingerprint or wood-grain pattern. Very regular, closely-spaced striations. Requires high magnification to resolve individual lamellae.
6. pearlite+widmanstatten: Lamellar pearlite colonies (fingerprint texture) in grain interiors combined with long straight needle-like plates at or near grain boundaries.

Respond with ONLY a JSON object:
{{"primary_microconstituent": "<spheroidite|network|spheroidite+widmanstatten|pearlite+spheroidite|pearlite|pearlite+widmanstatten>"}}"""


def make_prompt_fs_ref() -> str:
    """Few-shot reference image prompt — describes the 3x2 grid layout."""
    return """First image: a 3×2 reference grid showing one example of each of the 6 UHCS microstructure classes.

TOP ROW (left to right):
  (1) spheroidite — Scattered dark round/oval cementite particles on a light ferrite matrix. The particles are isolated, roughly spherical, and uniformly distributed. Looks like "polka dots." This forms from prolonged annealing below the eutectoid temperature.
  (2) network — Dark continuous lines forming a connected web/mesh pattern. These are cementite films along prior austenite grain boundaries. The lines outline polygonal grain shapes. Forms during slow cooling from above A1.
  (3) spheroidite+widmanstatten — A mix of round spheroidized particles AND straight elongated needle/plate-like features growing inward from grain boundaries. You see both "dots" and "needles" in the same image. Indicates partial spheroidization of Widmanstatten cementite.

BOTTOM ROW (left to right):
  (4) pearlite+spheroidite — Regions showing fingerprint-like lamellar striations (pearlite) alongside areas with scattered round particles (spheroidite). Two distinct textures coexist. Indicates incomplete spheroidization of pearlite.
  (5) pearlite — Fine parallel alternating dark/light lamellae creating a fingerprint or wood-grain pattern. Very regular, closely-spaced striations. Requires high magnification to resolve individual lamellae.
  (6) pearlite+widmanstatten — Lamellar pearlite colonies (fingerprint texture) in grain interiors combined with long straight needle-like plates at or near grain boundaries.

Study the differences in pattern geometry: dots vs lines vs needles vs lamellae."""


def make_prompt_fs_query(magnification: str = "unknown") -> str:
    """Few-shot query prompt."""
    return f"""Now classify this UHCS micrograph (magnification: ~{magnification}) by comparing to the 6 reference examples.

Identify the dominant pattern:
- Scattered round particles (dots) → spheroidite
- Connected dark lines outlining grains (web) → network
- Round particles + elongated plates/needles → spheroidite+widmanstatten
- Lamellar striations + round particles → pearlite+spheroidite
- Fine parallel lamellae (fingerprint) → pearlite
- Lamellae + needle plates at boundaries → pearlite+widmanstatten

Respond with ONLY JSON: {{"primary_microconstituent": "<spheroidite|network|spheroidite+widmanstatten|pearlite+spheroidite|pearlite|pearlite+widmanstatten>"}}"""
