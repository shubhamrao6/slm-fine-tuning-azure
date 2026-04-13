"""
Shared configuration for Task 4 — used by both direct and augmented notebooks.
"""

# Paths
TRAIN_DIR = "../../datasets/granulometry/train"
TEST_DIR = "../../datasets/granulometry/test"
TRAIN_MANIFEST = "../../datasets/granulometry/train_manifest.json"
TEST_MANIFEST = "../../datasets/granulometry/test_manifest.json"

# Model
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ORIGINAL_GSD = 8.0
MAX_DIM = 800  # consistent for train and eval
SEED = 42

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Ground truth
GT = {
    "A8":  (8, "coarse"),  "A16": (16, "coarse"), "A32": (32, "coarse"),
    "B8":  (8, "medium"),  "B16": (16, "medium"), "B32": (32, "medium"),
    "C8":  (8, "fine"),    "C16": (16, "fine"),   "C32": (32, "fine"),
}

# Azure OpenAI (for augmented notebook)
AZURE_ENDPOINT = "https://ether-openai.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-4.1"
AZURE_API_VERSION = "2024-12-01-preview"


def make_prompt(gsd: float) -> str:
    """The ONE prompt used everywhere — training and evaluation, both approaches."""
    return f"""Classify this concrete aggregate photograph.
Ground sampling distance (GSD) = {gsd:.1f} px/mm.
At this GSD: 8mm stone ≈ {8*gsd:.0f}px, 16mm ≈ {16*gsd:.0f}px, 32mm ≈ {32*gsd:.0f}px.

Classification axes:
1. MAX PARTICLE SIZE: estimate the largest stone's width in pixels, divide by GSD, round to 8, 16, or 32 mm.
2. GRADING (DIN 1045 standard — describes size DISTRIBUTION, not absolute size):
   - COARSE (A): particles concentrated near max size. Gaps between stones are EMPTY. Uniform, single-layer texture.
   - MEDIUM (B): balanced mix. Gaps PARTIALLY filled by smaller particles.
   - FINE (C): wide size range. Gaps COMPLETELY filled with small particles. Dense, packed texture.

Respond with JSON: {{"max_particle_size_mm": <8, 16, or 32>, "grading": "<coarse, medium, or fine>"}}"""
