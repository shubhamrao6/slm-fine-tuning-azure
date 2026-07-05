"""
Unconditioned CoT Ablation: Train with GPT-4.1's free classifications (often wrong).

Uses the SAME training and evaluation code as the multi-seed ablation.
The only difference is the training JSONL (unconditioned instead of conditioned).
"""
import os
import sys
import json
import re
import gc
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TASK4_DIR = REPO_ROOT / "task4-fine-tuning"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EPOCHS = 40
LR = 2e-5
GRAD_ACCUM = 4
EVAL_TEMPERATURE = 0.1

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TASKS = {
    "granulometry": {
        "unconditioned_jsonl": SCRIPT_DIR / "granulometry_unconditioned_cot.jsonl",
        "direct_jsonl": TASK4_DIR / "granulometry" / "training_data_direct.jsonl",
        "test_manifest": REPO_ROOT / "datasets" / "granulometry" / "test_manifest.json",
        "test_dir": REPO_ROOT / "datasets" / "granulometry" / "test",
        "eval_type": "granulometry",
    },
    "weld": {
        "unconditioned_jsonl": SCRIPT_DIR / "weld_unconditioned_cot.jsonl",
        "direct_jsonl": TASK4_DIR / "riawelc-weld" / "training_data_direct.jsonl",
        "test_dir": REPO_ROOT / "datasets" / "riawelc" / "testing",
        "classes": ["lack_of_penetration", "porosity", "cracks", "no_defect"],
        "eval_type": "classification",
    },
}

# Import shared code
sys.path.insert(0, str(SCRIPT_DIR.parent / "ablation-multi-seed"))
from ablation_multi_seed import (
    LoRADataset, parse_json_response, evaluate_granulometry_response,
    evaluate_classification_response, get_granulometry_eval_prompt,
    get_eval_prompt_from_jsonl, load_test_set_weld,
)


def train_and_evaluate_unconditioned(task_name, task_config):
    """Train with unconditioned CoT data and evaluate."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import get_cosine_schedule_with_warmup

    jsonl_path = str(task_config["unconditioned_jsonl"])
    print(f'\n  Loading training data from: {jsonl_path}')
    with open(jsonl_path) as f:
        n_examples = sum(1 for _ in f)
    print(f'  Training examples: {n_examples}')

    # Load model
    print(f'  Loading {MODEL_ID}...')
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=512*28*28)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto',
        max_memory={0: '6GiB', 1: '15GiB'} if torch.cuda.device_count() > 1 else None
    )
    base_model.enable_input_require_grads()
    print(f'  Model loaded in {time.time()-t0:.1f}s')

    # Train (identical loop to multi-seed)
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, task_type='CAUSAL_LM', bias='none'
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    dataset = LoRADataset(jsonl_path, processor)
    print(f'  Training: {len(dataset)} examples, {EPOCHS} epochs, lr={LR}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = max(len(dataset) * EPOCHS // GRAD_ACCUM, 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    model.train()
    for epoch in range(EPOCHS):
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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'    Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f} — lr: {scheduler.get_last_lr()[0]:.2e}')

    # Save adapter
    adapter_dir = str(RESULTS_DIR / f"{task_name}_unconditioned_cot_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    print(f'  Adapter saved to {adapter_dir}')
    model.unload()
    del model, optimizer, scheduler, dataset
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate (same logic as multi-seed)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    if task_name == "granulometry":
        with open(str(task_config["test_manifest"])) as f:
            manifest = json.load(f)
        test_dir = str(task_config["test_dir"])
        correct = 0
        total = 0
        with torch.no_grad():
            for i, entry in enumerate(manifest):
                img_path = os.path.join(test_dir, entry['image'])
                if not os.path.exists(img_path):
                    continue
                image = Image.open(img_path).convert('RGB')
                prompt = get_granulometry_eval_prompt(image)
                msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors='pt', padding=True).to(model.device)
                out_ids = model.generate(**inputs, max_new_tokens=128, temperature=EVAL_TEMPERATURE, do_sample=True)
                raw = processor.batch_decode(out_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
                del inputs, out_ids
                image.close()
                torch.cuda.empty_cache()
                parsed = parse_json_response(raw)
                _, _, is_correct = evaluate_granulometry_response(parsed, entry)
                if is_correct:
                    correct += 1
                total += 1
        accuracy = (correct / total * 100) if total > 0 else 0.0
    else:
        test_items = load_test_set_weld(task_config)
        eval_prompt = get_eval_prompt_from_jsonl(str(task_config["direct_jsonl"]))
        correct = 0
        total = 0
        with torch.no_grad():
            for i, item in enumerate(test_items):
                img_path = item["image_path"]
                if not os.path.exists(img_path):
                    continue
                image = Image.open(img_path).convert('RGB')
                msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': eval_prompt}]}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors='pt', padding=True).to(model.device)
                out_ids = model.generate(**inputs, max_new_tokens=256, temperature=EVAL_TEMPERATURE, do_sample=True)
                raw = processor.batch_decode(out_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
                del inputs, out_ids
                image.close()
                torch.cuda.empty_cache()
                parsed = parse_json_response(raw)
                is_correct = evaluate_classification_response(parsed, item["class"], key="defect_class")
                if is_correct:
                    correct += 1
                total += 1
        accuracy = (correct / total * 100) if total > 0 else 0.0

    print(f'  Result: {correct}/{total} ({accuracy:.1f}%)')
    del model, base_model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return accuracy


if __name__ == "__main__":
    results = {}
    for task_name, task_config in TASKS.items():
        if not task_config["unconditioned_jsonl"].exists():
            print(f"ERROR: {task_config['unconditioned_jsonl']} not found. Run generate_unconditioned_cot.py first.")
            continue

        result_file = RESULTS_DIR / f"{task_name}_unconditioned_cot.json"
        if result_file.exists():
            with open(result_file) as f:
                existing = json.load(f)
            print(f'[SKIP] {task_name}: {existing["accuracy"]:.1f}%')
            results[task_name] = existing["accuracy"]
            continue

        print(f'\n{"="*60}')
        print(f'  {task_name.upper()} | Unconditioned CoT (Gemini 2.5 Pro)')
        print(f'{"="*60}')

        t_start = time.time()
        accuracy = train_and_evaluate_unconditioned(task_name, task_config)
        elapsed = time.time() - t_start

        results[task_name] = accuracy
        with open(result_file, 'w') as f:
            json.dump({"task": task_name, "model": "gemini-2.5-pro", "approach": "unconditioned_cot",
                       "accuracy": accuracy, "elapsed_min": round(elapsed / 60, 1)}, f, indent=2)
        print(f'  → {accuracy:.1f}% (took {elapsed/60:.1f} min)')

    print(f'\n\n{"="*60}')
    print("UNCONDITIONED COT RESULTS (Gemini 2.5 Pro)")
    print(f'{"="*60}')
    for task_name, acc in results.items():
        print(f'  {task_name}: {acc:.1f}%')
