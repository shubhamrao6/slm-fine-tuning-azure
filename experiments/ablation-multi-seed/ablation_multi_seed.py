"""
Ablation 5: Statistical Significance via Multiple Seeds
=========================================================
Proves that CoT-augmented improvement over Direct LoRA is consistent
and not a lucky seed result. Runs the same experiments with 5 different
random seeds and reports mean ± std.

What changes per seed:
- torch.manual_seed() → LoRA adapter weight initialization, training data
  shuffle order per epoch, dropout mask during training.

What stays fixed:
- Training images (same JSONL files already cached)
- CoT descriptions (already generated and cached)
- Prompts (identical)
- Hyperparameters (LR=2e-5, epochs=40, r=16, alpha=32)
- Test set (same images per task)
- Evaluation (temp=0.1, same parsing logic)

Seeds: [42, 123, 456, 789, 1024]
Seed 42 = existing result. 4 new runs per task per approach.

Tasks: granulometry, steel_surface, uhcs, weld (4 tasks × 2 approaches × 5 seeds = 40 runs)
Total: 32 NEW training runs (seed 42 already done for all tasks)

Run on GCP L4 24GB: ~28 hours total (~$28)
"""

import os
import sys
import json
import re
import gc
import time
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 1024]

# Existing seed-42 results (from task4-fine-tuning)
SEED42_RESULTS = {
    "granulometry": {"direct": 71.3, "augmented": 79.6},
    "steel_surface": {"direct": 63.1, "augmented": 66.7},
    "uhcs": {"direct": 67.5, "augmented": 68.4},
    "weld": {"direct": 73.3, "augmented": 75.8},
}

# Paths relative to THIS script's location
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TASK4_DIR = REPO_ROOT / "task4-fine-tuning"

TASKS = {
    "granulometry": {
        "direct_jsonl": TASK4_DIR / "granulometry" / "training_data_direct.jsonl",
        "augmented_jsonl": TASK4_DIR / "granulometry" / "training_data_augmented.jsonl",
        "test_manifest": REPO_ROOT / "datasets" / "granulometry" / "test_manifest.json",
        "test_dir": REPO_ROOT / "datasets" / "granulometry" / "test",
        "eval_type": "granulometry",  # special: both_correct metric
        "test_images": 108,
    },
    "steel_surface": {
        "direct_jsonl": TASK4_DIR / "steel-surface" / "training_data_direct.jsonl",
        "augmented_jsonl": TASK4_DIR / "steel-surface" / "training_data_augmented.jsonl",
        "test_dir": REPO_ROOT / "datasets" / "neu-cls" / "NEU-DET" / "validation" / "images",
        "eval_type": "classification",  # standard accuracy
        "classes": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
        "test_images": 360,
    },
    "uhcs": {
        "direct_jsonl": TASK4_DIR / "uhcs-microstructure" / "training_data_direct.jsonl",
        "augmented_jsonl": TASK4_DIR / "uhcs-microstructure" / "training_data_augmented.jsonl",
        "test_manifest": REPO_ROOT / "datasets" / "uh-carbon-steel" / "test_manifest.json",
        "eval_type": "classification",  # standard accuracy
        "test_images": 117,
    },
    "weld": {
        "direct_jsonl": TASK4_DIR / "riawelc-weld" / "training_data_direct.jsonl",
        "augmented_jsonl": TASK4_DIR / "riawelc-weld" / "training_data_augmented.jsonl",
        "test_dir": REPO_ROOT / "datasets" / "riawelc" / "testing",
        "eval_type": "classification",  # standard accuracy
        "classes": ["lack_of_penetration", "porosity", "cracks", "no_defect"],
        "test_images": 240,
    },
}

# Model
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
EPOCHS = 40
LR = 2e-5
GRAD_ACCUM = 4

# Evaluation
EVAL_TEMPERATURE = 0.1
EVAL_MAX_TOKENS = 256

# Results
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SEED MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def set_all_seeds(seed):
    """Set random seeds. We only set Python random and numpy for any data-loading
    reproducibility. We do NOT set torch seeds — this lets each run get a naturally
    different LoRA initialization, which is what we're measuring."""
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_response(raw):
    """Parse JSON from model response, handling markdown fences and CoT prefixes."""
    if not raw:
        return None
    raw = raw.replace('<', '').replace('>', '')
    cleaned = re.sub(r'```json\s*', '', raw)
    cleaned = re.sub(r'```\s*', '', cleaned).strip()

    # Try direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Find JSON object in response (re.DOTALL to match across newlines)
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: regex for individual fields
    sm = re.search(r'"max_particle_size_mm"\s*:\s*(\d+)', cleaned)
    gm = re.search(r'"grading"\s*:\s*"(\w+)"', cleaned)
    if sm and gm:
        return {'max_particle_size_mm': int(sm.group(1)), 'grading': gm.group(1)}

    dm = re.search(r'"primary_microconstituent"\s*:\s*"([\w+]+)"', cleaned)
    if dm:
        return {'primary_microconstituent': dm.group(1)}

    dm2 = re.search(r'"defect_class"\s*:\s*"([\w]+)"', cleaned)
    if dm2:
        return {'defect_class': dm2.group(1)}

    return None


def evaluate_granulometry_response(parsed, gt_entry):
    """Evaluate granulometry: returns (size_correct, grading_correct, both_correct)."""
    if parsed is None:
        return False, False, False

    gt_size = gt_entry["max_particle_size_mm"]
    gt_grading = gt_entry["grading"]

    pred_size = parsed.get("max_particle_size_mm")
    if isinstance(pred_size, str):
        pred_size = int(pred_size) if pred_size.isdigit() else None
    pred_grading = parsed.get("grading", "").lower().strip()

    size_ok = (pred_size == gt_size)
    grading_ok = (pred_grading == gt_grading)
    return size_ok, grading_ok, (size_ok and grading_ok)


def evaluate_classification_response(parsed, gt_class, key="defect_class"):
    """Evaluate classification: returns True/False."""
    if parsed is None:
        return False
    pred = parsed.get(key, "").lower().strip()
    return pred == gt_class.lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class LoRADataset(torch.utils.data.Dataset):
    """Universal dataset for all 4 tasks — reads JSONL training data."""

    def __init__(self, jsonl_path, processor):
        with open(jsonl_path) as f:
            self.data = [json.loads(line) for line in f]
        self.processor = processor
        self.base_dir = os.path.dirname(jsonl_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        msgs = entry['messages']

        # Extract image path and text from user message
        img_path = None
        user_text = ""
        for content in msgs[0]['content']:
            if content['type'] == 'image':
                img_path = content['image']
            elif content['type'] == 'text':
                user_text = content['text']

        # Resolve relative image path
        if img_path and not os.path.isabs(img_path):
            img_path = os.path.normpath(os.path.join(self.base_dir, img_path))

        # Get assistant response
        assistant_text = msgs[1]['content']
        if not isinstance(assistant_text, str):
            assistant_text = json.dumps(assistant_text)

        # Load image
        image = Image.open(img_path).convert('RGB') if img_path else None

        # Build chat and tokenize
        chat = [
            {'role': 'user', 'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': user_text}
            ]},
            {'role': 'assistant', 'content': [
                {'type': 'text', 'text': assistant_text}
            ]}
        ]
        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[image], return_tensors='pt', padding=True)

        input_ids = inputs['input_ids'].squeeze(0)
        labels = input_ids.clone()

        # Mask everything except assistant tokens
        ast_tokens = self.processor.tokenizer.encode(assistant_text, add_special_tokens=False)
        if len(ast_tokens) < len(labels):
            labels[:-len(ast_tokens)] = -100

        if image:
            image.close()

        return {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels,
            'pixel_values': inputs.get('pixel_values', None),
            'image_grid_thw': inputs.get('image_grid_thw', None),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_lora(base_model, processor, jsonl_path, output_dir, seed):
    """Train a LoRA adapter with the given seed."""
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    # Set seed BEFORE LoRA initialization (this is the key point of the ablation)
    set_all_seeds(seed)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, task_type='CAUSAL_LM', bias='none'
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    dataset = LoRADataset(jsonl_path, processor)
    print(f'  Training: {len(dataset)} examples, {EPOCHS} epochs, lr={LR}, seed={seed}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = max(len(dataset) * EPOCHS // GRAD_ACCUM, 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    model.train()
    losses = []

    for epoch in range(EPOCHS):
        # Iterate sequentially (same as original notebooks — no shuffling)
        epoch_loss = 0
        for step in range(len(dataset)):
            batch = dataset[step]
            ids = batch['input_ids'].unsqueeze(0).to(model.device)
            mask = batch['attention_mask'].unsqueeze(0).to(model.device)
            lab = batch['labels'].unsqueeze(0).to(model.device)

            kw = {'input_ids': ids, 'attention_mask': mask, 'labels': lab}
            if batch.get('pixel_values') is not None:
                kw['pixel_values'] = batch['pixel_values'].to(model.device)
            if batch.get('image_grid_thw') is not None:
                kw['image_grid_thw'] = batch['image_grid_thw'].to(model.device)

            out = model(**kw)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            del ids, mask, lab, out, loss
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'    Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f} — lr: {scheduler.get_last_lr()[0]:.2e}')

    # Save adapter
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f'  Adapter saved to {output_dir}')

    # Cleanup
    model.unload()
    del model, optimizer, scheduler, dataset
    gc.collect()
    torch.cuda.empty_cache()

    return losses


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def load_test_set_granulometry(task_config):
    """Load granulometry test manifest."""
    with open(task_config["test_manifest"]) as f:
        manifest = json.load(f)
    test_items = []
    for entry in manifest:
        img_path = os.path.join(str(task_config["test_dir"]), entry["image"])
        test_items.append({
            "image_path": img_path,
            "class": entry["class"],
            "gt": entry,  # has max_particle_size_mm and grading
        })
    return test_items


def load_test_set_steel(task_config):
    """Load steel surface test set from directory structure."""
    test_items = []
    test_dir = str(task_config["test_dir"])
    for cls in task_config["classes"]:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"  WARNING: {cls_dir} not found")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                test_items.append({
                    "image_path": os.path.join(cls_dir, fname),
                    "class": cls,
                })
    return test_items


def load_test_set_uhcs(task_config):
    """Load UHCS test manifest."""
    with open(task_config["test_manifest"]) as f:
        manifest = json.load(f)
    test_items = []
    for entry in manifest:
        test_items.append({
            "image_path": entry["image_path"] if "image_path" in entry else entry.get("image", ""),
            "class": entry["class"],
            "magnification": entry.get("magnification", "unknown"),
        })
    return test_items


def load_test_set_weld(task_config):
    """Load weld test set from directory structure — sample 60 per class (same as notebook)."""
    import random as _random
    _random.seed(42)  # Use fixed seed for deterministic test set
    SAMPLE_PER_CLASS = 60
    test_items = []
    test_dir = str(task_config["test_dir"])
    for cls in task_config["classes"]:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"  WARNING: {cls_dir} not found")
            continue
        images = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith('.png')])
        if len(images) > SAMPLE_PER_CLASS:
            images = _random.sample(images, SAMPLE_PER_CLASS)
        for fname in images:
            test_items.append({
                "image_path": os.path.join(cls_dir, fname),
                "class": cls,
            })
    _random.shuffle(test_items)
    return test_items


def get_eval_prompt_from_jsonl(jsonl_path):
    """Extract the evaluation prompt from the first direct JSONL entry."""
    with open(jsonl_path) as f:
        first = json.loads(f.readline())
    for content in first['messages'][0]['content']:
        if content['type'] == 'text':
            return content['text']
    return ""


def get_granulometry_eval_prompt(image):
    """Compute the granulometry prompt with dynamic GSD based on image size."""
    ORIGINAL_GSD = 8.0
    MAX_DIM = 800
    scale = min(MAX_DIM / max(image.size), 1.0)
    gsd = ORIGINAL_GSD * scale
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


def evaluate_model(model, processor, task_name, task_config):
    """Evaluate a trained model on the task's test set. Returns accuracy (%)."""

    # Load test set
    if task_name == "granulometry":
        test_items = load_test_set_granulometry(task_config)
    elif task_name == "steel_surface":
        test_items = load_test_set_steel(task_config)
    elif task_name == "uhcs":
        test_items = load_test_set_uhcs(task_config)
    elif task_name == "weld":
        test_items = load_test_set_weld(task_config)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Get evaluation prompt from direct JSONL (same prompt used for eval)
    # For granulometry/UHCS this will be overridden per-image
    eval_prompt = get_eval_prompt_from_jsonl(str(task_config["direct_jsonl"]))

    # Task-specific max_new_tokens
    max_tokens = 128 if task_name == "granulometry" else EVAL_MAX_TOKENS

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, item in enumerate(test_items):
            img_path = item["image_path"]
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert('RGB')

            # Build prompt — task-specific
            if task_name == "granulometry":
                # Granulometry uses dynamic GSD based on image size
                prompt = get_granulometry_eval_prompt(image)
            elif task_name == "uhcs" and "magnification" in item:
                # UHCS has magnification in prompt — rebuild per image
                mag = item.get("magnification", "unknown")
                # Import make_prompt from config or use the JSONL prompt structure
                # The UHCS JSONL stores each image's unique prompt, so we use the generic one
                # But since we extract from first JSONL entry, it has a specific magnification
                # Instead, use the make_prompt pattern:
                prompt = f"""Classify this ultra-high carbon steel (UHCS) micrograph.

This is an optical/SEM micrograph at approximately {mag} magnification showing the microstructure of UHCS after heat treatment.

Possible microconstituent classes:
1. spheroidite: Scattered dark round/oval cementite particles on a light ferrite matrix. The particles are isolated, roughly spherical, and uniformly distributed. Looks like "polka dots." This forms from prolonged annealing below the eutectoid temperature.
2. network: Dark continuous lines forming a connected web/mesh pattern. These are cementite films along prior austenite grain boundaries. The lines outline polygonal grain shapes. Forms during slow cooling from above A1.
3. spheroidite+widmanstatten: A mix of round spheroidized particles AND straight elongated needle/plate-like features growing inward from grain boundaries. You see both "dots" and "needles" in the same image. Indicates partial spheroidization of Widmanstatten cementite.
4. pearlite+spheroidite: Regions showing fingerprint-like lamellar striations (pearlite) alongside areas with scattered round particles (spheroidite). Two distinct textures coexist. Indicates incomplete spheroidization of pearlite.
5. pearlite: Fine parallel alternating dark/light lamellae creating a fingerprint or wood-grain pattern. Very regular, closely-spaced striations. Requires high magnification to resolve individual lamellae.

Respond with ONLY a JSON object:
{{"primary_microconstituent": "<spheroidite|network|spheroidite+widmanstatten|pearlite+spheroidite|pearlite>"}}"""
            else:
                prompt = eval_prompt

            # Build message
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]

            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors='pt', padding=True).to(model.device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=EVAL_TEMPERATURE,
                do_sample=True,
            )
            raw = processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0].strip()

            image.close()
            del inputs, output_ids
            torch.cuda.empty_cache()

            # Parse and evaluate
            parsed = parse_json_response(raw)
            gt_class = item["class"]

            if task_name == "granulometry":
                _, _, is_correct = evaluate_granulometry_response(parsed, item["gt"])
            else:
                # Determine the key based on task
                if task_name == "uhcs":
                    key = "primary_microconstituent"
                else:
                    key = "defect_class"
                is_correct = evaluate_classification_response(parsed, gt_class, key=key)

            if is_correct:
                correct += 1
            total += 1

            if (i + 1) % 50 == 0:
                print(f'    Eval: {i+1}/{len(test_items)} — acc so far: {correct}/{total} ({correct/total*100:.1f}%)')

    accuracy = (correct / total * 100) if total > 0 else 0.0
    print(f'  Evaluation: {correct}/{total} ({accuracy:.1f}%)')
    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAIN + EVALUATE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(task_name, approach, seed):
    """
    Full pipeline: load model → train LoRA → evaluate → cleanup.
    Returns accuracy (float, percentage).
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    task_config = TASKS[task_name]
    jsonl_path = str(task_config["direct_jsonl"] if approach == "direct" else task_config["augmented_jsonl"])

    print(f'\n  Loading training data from: {jsonl_path}')
    with open(jsonl_path) as f:
        n_examples = sum(1 for _ in f)
    print(f'  Training examples: {n_examples}')

    # Load base model
    print(f'  Loading {MODEL_ID}...')
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=512*28*28)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto'
    )
    base_model.enable_input_require_grads()
    print(f'  Model loaded in {time.time()-t0:.1f}s')

    # Train
    adapter_dir = str(RESULTS_DIR / f"{task_name}_{approach}_seed{seed}_adapter")
    losses = train_lora(base_model, processor, jsonl_path, adapter_dir, seed)

    # Load adapter for evaluation
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    # Evaluate
    accuracy = evaluate_model(model, processor, task_name, task_config)

    # Cleanup
    del model, base_model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f'  GPU memory freed.')

    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation():
    """Run all seeds for all tasks and approaches."""
    all_results = {}

    for task_name in TASKS:
        all_results[task_name] = {"direct": {}, "augmented": {}}

        for approach in ["direct", "augmented"]:
            for seed in SEEDS:
                run_id = f"{task_name}_{approach}_seed{seed}"
                result_file = RESULTS_DIR / f"{run_id}.json"

                # Skip if already completed
                if result_file.exists():
                    with open(result_file) as f:
                        existing = json.load(f)
                    accuracy = existing["accuracy"]
                    all_results[task_name][approach][seed] = accuracy
                    print(f'\n[SKIP] {run_id} — already done: {accuracy:.1f}%')
                    continue

                print(f'\n{"="*60}')
                print(f'  Task: {task_name} | Approach: {approach} | Seed: {seed}')
                print(f'{"="*60}')

                t_start = time.time()
                accuracy = train_and_evaluate(task_name, approach, seed)
                elapsed = time.time() - t_start

                all_results[task_name][approach][seed] = accuracy

                # Save individual result
                result = {
                    "task": task_name,
                    "approach": approach,
                    "seed": seed,
                    "accuracy": accuracy,
                    "elapsed_min": round(elapsed / 60, 1),
                }
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)

                print(f'  → Accuracy: {accuracy:.1f}% (took {elapsed/60:.1f} min)')

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print(f'\n\n{"="*70}')
    print("ABLATION 5 RESULTS: MEAN ± STD ACROSS 5 SEEDS")
    print(f'{"="*70}')

    summary = {}
    for task_name in TASKS:
        print(f'\n{task_name.upper()}:')
        summary[task_name] = {}

        for approach in ["direct", "augmented"]:
            accs = [all_results[task_name][approach].get(s) for s in SEEDS]
            accs = [a for a in accs if a is not None]

            if accs:
                mean = np.mean(accs)
                std = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
                print(f'  {approach:12s}: {mean:.1f} ± {std:.1f}%  (seeds: {[f"{a:.1f}" for a in accs]})')
                summary[task_name][approach] = {"mean": round(mean, 1), "std": round(std, 1), "runs": accs}
            else:
                print(f'  {approach:12s}: NO RESULTS')

        # Paired difference (CoT-Aug minus Direct)
        direct_accs = [all_results[task_name]["direct"].get(s) for s in SEEDS]
        aug_accs = [all_results[task_name]["augmented"].get(s) for s in SEEDS]
        paired = [(a - d) for d, a in zip(direct_accs, aug_accs) if d is not None and a is not None]

        if paired:
            delta_mean = np.mean(paired)
            delta_std = np.std(paired, ddof=1) if len(paired) > 1 else 0.0
            print(f'  {"DELTA":12s}: +{delta_mean:.1f} ± {delta_std:.1f}pp  (CoT-Aug minus Direct)')

            # Simple paired t-test
            if len(paired) >= 3:
                from scipy import stats
                t_stat, p_value = stats.ttest_1samp(paired, 0)
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f'  {"":12s}  p={p_value:.4f} ({sig}) — paired t-test')

            summary[task_name]["delta"] = {"mean": round(delta_mean, 1), "std": round(delta_std, 1), "p_value": round(p_value, 4) if len(paired) >= 3 else None}

    # Save summary
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f'\n\nAll results saved to {RESULTS_DIR}/')


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running a single task via CLI: python ablation_multi_seed.py granulometry
    if len(sys.argv) > 1:
        task_filter = sys.argv[1]
        if task_filter in TASKS:
            TASKS_TO_RUN = {task_filter: TASKS[task_filter]}
            # Monkey-patch for single-task run
            original_tasks = dict(TASKS)
            for k in list(TASKS.keys()):
                if k != task_filter:
                    del TASKS[k]
        else:
            print(f"Unknown task: {task_filter}. Choose from: {list(TASKS.keys())}")
            sys.exit(1)

    run_ablation()
