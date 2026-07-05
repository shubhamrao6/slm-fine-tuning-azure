"""
Exact reproduction of the granulometry direct training notebook.
Copied directly from train_direct.ipynb — NO changes.
"""
import os, json, re, time, torch, gc, random
import numpy as np
from PIL import Image
from collections import defaultdict, Counter

# --- From config.py (paths relative to repo root) ---
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_DIR = str(REPO_ROOT / "datasets/granulometry/train")
TEST_DIR = str(REPO_ROOT / "datasets/granulometry/test")
TRAIN_MANIFEST = str(REPO_ROOT / "datasets/granulometry/train_manifest.json")
TEST_MANIFEST = str(REPO_ROOT / "datasets/granulometry/test_manifest.json")
TRAINING_JSONL = str(REPO_ROOT / "task4-fine-tuning/granulometry/training_data_direct.jsonl")
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ORIGINAL_GSD = 8.0
MAX_DIM = 800
SEED = 42
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
GT = {
    "A8":  (8, "coarse"),  "A16": (16, "coarse"), "A32": (32, "coarse"),
    "B8":  (8, "medium"),  "B16": (16, "medium"), "B32": (32, "medium"),
    "C8":  (8, "fine"),    "C16": (16, "fine"),   "C32": (32, "fine"),
}

def make_prompt(gsd: float) -> str:
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

# --- Setup ---
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)} — {torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB')

# --- Step 3: Load Model + LoRA (exact copy from notebook) ---
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset

processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=512*28*28)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map='auto', torch_dtype=torch.bfloat16)
base_model.enable_input_require_grads()

lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS, task_type='CAUSAL_LM', bias='none')

print(f'Model loaded (BF16). Processor max_pixels=512*28*28')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.1f} GB')

# --- Step 4: Dataset + Training (exact copy from notebook) ---
class GranulometryDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        with open(jsonl_path) as f:
            self.data = [json.loads(line) for line in f]
        self.processor = processor
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        entry = self.data[idx]
        msgs = entry['messages']
        img_path = next((c['image'] for c in msgs[0]['content'] if c['type']=='image'), None)
        user_text = next((c['text'] for c in msgs[0]['content'] if c['type']=='text'), '')
        assistant_text = msgs[1]['content']
        if not isinstance(assistant_text, str): assistant_text = json.dumps(assistant_text)
        image = Image.open(img_path).convert('RGB') if img_path else None
        chat = [{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':user_text}]},
                {'role':'assistant','content':[{'type':'text','text':assistant_text}]}]
        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[image], return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].squeeze(0)
        labels = input_ids.clone()
        ast_tokens = self.processor.tokenizer.encode(assistant_text, add_special_tokens=False)
        if len(ast_tokens) < len(labels): labels[:-len(ast_tokens)] = -100
        if image: image.close()
        return {'input_ids': input_ids, 'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': labels, 'pixel_values': inputs.get('pixel_values', None),
                'image_grid_thw': inputs.get('image_grid_thw', None)}

EPOCHS = 40
LR = 2e-5
GRAD_ACCUM = 4

def train_lora(base_model, data_path, output_dir, epochs=EPOCHS, lr=LR):
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    dataset = GranulometryDataset(data_path, processor)
    print(f'Training: {len(dataset)} examples, {epochs} epochs, lr={lr}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(len(dataset) * epochs // GRAD_ACCUM, 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(dataset)):
            batch = dataset[i]
            ids = batch['input_ids'].unsqueeze(0).to(model.device)
            mask = batch['attention_mask'].unsqueeze(0).to(model.device)
            lab = batch['labels'].unsqueeze(0).to(model.device)
            kw = {'input_ids':ids,'attention_mask':mask,'labels':lab}
            if batch.get('pixel_values') is not None: kw['pixel_values']=batch['pixel_values'].to(model.device)
            if batch.get('image_grid_thw') is not None: kw['image_grid_thw']=batch['image_grid_thw'].to(model.device)
            out = model(**kw)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM
            if (i+1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            del ids, mask, lab, out, loss; torch.cuda.empty_cache()
        avg = epoch_loss / len(dataset)
        losses.append(avg)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1}/{epochs} — loss: {avg:.4f} — lr: {scheduler.get_last_lr()[0]:.2e}')
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f'Adapter saved to {output_dir}/')
    model.unload()
    del model, optimizer, scheduler, dataset
    gc.collect(); torch.cuda.empty_cache()
    print(f'Memory freed. GPU 0: {torch.cuda.memory_allocated(0)/1e9:.1f} GB')
    return losses

# --- Run training ---
losses = train_lora(base_model, TRAINING_JSONL, 'repro_lora_direct', epochs=EPOCHS, lr=LR)

# --- Step 5: Evaluate (exact copy from notebook) ---
def parse_response(raw):
    if not raw: return None
    raw = raw.replace('<','').replace('>','')
    raw = re.sub(r'```json\s*','',raw); raw = re.sub(r'```\s*','',raw).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict): return obj
    except: pass
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    sm = re.search(r'"max_particle_size_mm"\s*:\s*(\d+)', raw)
    gm = re.search(r'"grading"\s*:\s*"(\w+)"', raw)
    if sm and gm: return {'max_particle_size_mm': int(sm.group(1)), 'grading': gm.group(1)}
    return None

def run_eval(model, processor, manifest, max_dim=MAX_DIM):
    model.eval()
    results=[]; cs=0; cg=0; vj=0; tt=0
    for i, entry in enumerate(manifest):
        img_path = os.path.join(TEST_DIR, entry['image'])
        if not os.path.exists(img_path): continue
        image = Image.open(img_path).convert('RGB')
        scale = min(max_dim/max(image.size), 1.0)
        gsd = ORIGINAL_GSD * scale
        prompt = make_prompt(gsd)
        msgs = [{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors='pt', padding=True).to(model.device)
        t=time.time()
        with torch.no_grad(): ids = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=True)
        elapsed=time.time()-t; tt+=elapsed
        out = processor.batch_decode(ids[:,inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        del inputs, ids; image.close(); torch.cuda.empty_cache()
        parsed = parse_response(out)
        gs=entry['max_particle_size_mm']; gg=entry['grading']; so=False; go=False
        if parsed:
            vj+=1; ps=parsed.get('max_particle_size_mm')
            if isinstance(ps,str): ps=int(ps) if ps.isdigit() else None
            if ps==gs: so=True; cs+=1
            if parsed.get('grading','').lower()==gg: go=True; cg+=1
        results.append({'image':entry['image'],'class':entry['class'],'gt_size':gs,'gt_grading':gg,
            'predicted':parsed,'raw':out,'size_correct':so,'grading_correct':go,'valid_json':parsed is not None,'time_s':round(elapsed,2)})
        if (i+1)%20==0:
            n=i+1; print(f'  [{n}/{len(manifest)}] Size:{cs}/{n}({cs/n*100:.0f}%) Grade:{cg}/{n}({cg/n*100:.0f}%)')
    return results, cs, cg, vj, tt

with open(TEST_MANIFEST) as f: test_manifest = json.load(f)
print(f'Test set: {len(test_manifest)} images')

# Load adapter and evaluate
model = PeftModel.from_pretrained(base_model, 'repro_lora_direct')
r_full, cs, cg, vj, tt = run_eval(model, processor, test_manifest)
n=len(r_full); both=sum(1 for x in r_full if x['size_correct'] and x['grading_correct'])
print(f'\nJSON: {vj}/{n} ({vj/n*100:.0f}%) | Size: {cs}/{n} ({cs/n*100:.1f}%) | Grading: {cg}/{n} ({cg/n*100:.1f}%) | Both: {both}/{n} ({both/n*100:.1f}%)')
