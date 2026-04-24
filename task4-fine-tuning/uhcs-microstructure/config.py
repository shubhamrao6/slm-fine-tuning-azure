"""
Shared configuration for Task 4 — UHCS Microstructure fine-tuning.
Used by both direct and augmented notebooks.

Prompts use the SAME detailed definitions as the task3 benchmarking config
(the ones GPT-4.1 scored 71.7% FS and GPT-5 scored 80.0% FS with).
"""
import os

# Paths
DATASET_ROOT = os.environ.get("DATASET_DIR", "../../datasets/uh-carbon-steel")
IMAGES_DIR = os.path.join(DATASET_ROOT, "For Training", "Cropped")
METADATA_PATH = os.path.join(DATASET_ROOT, "new_metadata.xlsx")
TEST_MANIFEST = os.path.join(DATASET_ROOT, "test_manifest.json")
TRAIN_MANIFEST = os.path.join(DATASET_ROOT, "train_manifest.json")

# Model
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SEED = 42
IMAGES_PER_CLASS = 6  # 6 per class × 5 classes = 30 training images

# Classes (5 microconstituent labels — pearlite+widmanstatten dropped, only 5 total images)
CLASSES = [
    "spheroidite",
    "network",
    "spheroidite+widmanstatten",
    "pearlite+spheroidite",
    "pearlite",
]

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


def make_prompt(magnification: str = "unknown") -> str:
    """The ONE prompt used everywhere — training and evaluation, both approaches.
    Identical to the task3 benchmarking ZS prompt."""
    return f"""Classify this ultra-high carbon steel (UHCS) micrograph.

This is an optical/SEM micrograph at approximately {magnification} magnification showing the microstructure of UHCS after heat treatment.

Possible microconstituent classes:
1. spheroidite: Scattered dark round/oval cementite particles on a light ferrite matrix. The particles are isolated, roughly spherical, and uniformly distributed. Looks like "polka dots." This forms from prolonged annealing below the eutectoid temperature.
2. network: Dark continuous lines forming a connected web/mesh pattern. These are cementite films along prior austenite grain boundaries. The lines outline polygonal grain shapes. Forms during slow cooling from above A1.
3. spheroidite+widmanstatten: A mix of round spheroidized particles AND straight elongated needle/plate-like features growing inward from grain boundaries. You see both "dots" and "needles" in the same image. Indicates partial spheroidization of Widmanstatten cementite.
4. pearlite+spheroidite: Regions showing fingerprint-like lamellar striations (pearlite) alongside areas with scattered round particles (spheroidite). Two distinct textures coexist. Indicates incomplete spheroidization of pearlite.
5. pearlite: Fine parallel alternating dark/light lamellae creating a fingerprint or wood-grain pattern. Very regular, closely-spaced striations. Requires high magnification to resolve individual lamellae.

Respond with ONLY a JSON object:
{{"primary_microconstituent": "<spheroidite|network|spheroidite+widmanstatten|pearlite+spheroidite|pearlite>"}}"""


def make_seal_prompt(microconstituent: str, magnification: str = "unknown") -> str:
    """Prompt for GPT-4.1 to generate answer-conditioned description ONLY (no JSON).
    Uses the same detailed definitions as make_prompt()."""
    class_descriptions = {
        "spheroidite": "Scattered dark round/oval cementite particles on a light ferrite matrix. The particles are isolated, roughly spherical, and uniformly distributed. Looks like polka dots. Forms from prolonged annealing below the eutectoid temperature",
        "network": "Dark continuous lines forming a connected web/mesh pattern. These are cementite films along prior austenite grain boundaries. The lines outline polygonal grain shapes. Forms during slow cooling from above A1",
        "spheroidite+widmanstatten": "A mix of round spheroidized particles AND straight elongated needle/plate-like features growing inward from grain boundaries. Both dots and needles visible. Indicates partial spheroidization of Widmanstatten cementite",
        "pearlite+spheroidite": "Regions showing fingerprint-like lamellar striations (pearlite) alongside areas with scattered round particles (spheroidite). Two distinct textures coexist. Indicates incomplete spheroidization of pearlite",
        "pearlite": "Fine parallel alternating dark/light lamellae creating a fingerprint or wood-grain pattern. Very regular, closely-spaced striations. Requires high magnification to resolve individual lamellae",
    }
    desc = class_descriptions.get(microconstituent, microconstituent)

    return f"""Look at this UHCS micrograph (magnification: ~{magnification}). The correct microconstituent classification is: {microconstituent}
Definition: {desc}

Key distinguishing features between similar classes:
- spheroidite vs network: spheroidite has ISOLATED round particles (dots), network has CONNECTED dark lines (web outlining grains)
- spheroidite vs spheroidite+widmanstatten: pure spheroidite has ONLY round particles, spheroidite+widmanstatten also has straight NEEDLE/PLATE features from grain boundaries
- pearlite vs pearlite+spheroidite: pure pearlite has ONLY lamellar striations (fingerprint), pearlite+spheroidite also has scattered round particles between lamellar regions
- spheroidite+widmanstatten vs pearlite+spheroidite: spheroidite+widmanstatten has dots+needles, pearlite+spheroidite has lamellae+dots — the key difference is NEEDLES vs LAMELLAE

Explain WHY this classification is correct based on what you see:
1. Describe the dominant visual pattern — what shapes and textures do you observe?
2. Explain why these features match {microconstituent} using the definition above
3. Explain why it is NOT the most similar class

Write 2-3 sentences justifying the classification. Be specific about what you see in THIS image.
Reply with ONLY the justification text. No JSON, no labels, just the explanation."""
