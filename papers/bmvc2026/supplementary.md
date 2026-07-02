# Supplementary Material

**Paper:** Answer-Conditioned Chain-of-Thought Distillation for Few-Shot Industrial Vision with Small VLMs

---

## A. Classification Prompts

The following prompts are used identically for training and evaluation. Each task has one prompt that contains image context, class definitions, and output format instructions.

### A.1 Concrete Aggregate Grading (Granulometry)

```
Classify this concrete aggregate photograph.
Ground sampling distance (GSD) = {gsd} px/mm.
At this GSD: 8mm stone ≈ {8*gsd}px, 16mm ≈ {16*gsd}px, 32mm ≈ {32*gsd}px.

Classification axes:
1. MAX PARTICLE SIZE: estimate the largest stone's width in pixels, divide by GSD, round to 8, 16, or 32 mm.
2. GRADING (DIN 1045 standard — describes size DISTRIBUTION, not absolute size):
   - COARSE (A): particles concentrated near max size. Gaps between stones are EMPTY. Uniform, single-layer texture.
   - MEDIUM (B): balanced mix. Gaps PARTIALLY filled by smaller particles.
   - FINE (C): wide size range. Gaps COMPLETELY filled with small particles. Dense, packed texture.

Respond with JSON: {"max_particle_size_mm": <8, 16, or 32>, "grading": "<coarse, medium, or fine>"}
```

Note: GSD is computed dynamically based on image resize factor. Training uses GSD = 2.1 px/mm (images resized to 800px max from original ~3000px at 8.0 px/mm).

### A.2 Steel Surface Defect Detection (NEU-CLS)

```
Classify this steel surface defect image.

The image shows a 200×200 pixel grayscale photograph of a hot-rolled steel strip surface.

Possible defect classes:
1. crazing: a network of fine, shallow cracks spreading across the surface in a web-like or mosaic pattern. The cracks are thin, irregular, and multi-directional. Overall texture looks fragmented but the surface is relatively uniform in brightness.
2. inclusion: dark, irregularly shaped spots or elongated streaks embedded in the steel surface. These are foreign material (slag, oxide) trapped during solidification. The background is darker and more uniform than other classes.
3. patches: large irregular regions where the surface texture or brightness changes abruptly. You see distinct lighter or darker zones with soft, blotchy boundaries. The contrast between zones is high.
4. pitted_surface: scattered small dark holes or shallow depressions across a lighter background. The pits are roughly circular and distributed somewhat randomly. The overall surface appears brighter than most other classes.
5. rolled-in_scale: oxide scale that was pressed into the surface during the hot-rolling process. Appears as elongated dark marks, streaks, or patches aligned roughly parallel to the rolling direction. The marks have irregular edges.
6. scratches: one or more linear grooves or marks on the surface, typically running in a consistent direction. The lines are sharper and more defined than crazing cracks, and usually fewer in number.

Respond with ONLY a JSON object:
{"defect_class": "<crazing|inclusion|patches|pitted_surface|rolled-in_scale|scratches>"}
```

### A.3 UHCS Microstructure Classification

```
Classify this ultra-high carbon steel (UHCS) micrograph.

This is an optical/SEM micrograph at approximately {magnification} magnification showing the microstructure of UHCS after heat treatment.

Possible microconstituent classes:
1. spheroidite: Scattered dark round/oval cementite particles on a light ferrite matrix. The particles are isolated, roughly spherical, and uniformly distributed. Looks like "polka dots." This forms from prolonged annealing below the eutectoid temperature.
2. network: Dark continuous lines forming a connected web/mesh pattern. These are cementite films along prior austenite grain boundaries. The lines outline polygonal grain shapes. Forms during slow cooling from above A1.
3. spheroidite+widmanstatten: A mix of round spheroidized particles AND straight elongated needle/plate-like features growing inward from grain boundaries. You see both "dots" and "needles" in the same image. Indicates partial spheroidization of Widmanstatten cementite.
4. pearlite+spheroidite: Regions showing fingerprint-like lamellar striations (pearlite) alongside areas with scattered round particles (spheroidite). Two distinct textures coexist. Indicates incomplete spheroidization of pearlite.
5. pearlite: Fine parallel alternating dark/light lamellae creating a fingerprint or wood-grain pattern. Very regular, closely-spaced striations. Requires high magnification to resolve individual lamellae.

Respond with ONLY a JSON object:
{"primary_microconstituent": "<spheroidite|network|spheroidite+widmanstatten|pearlite+spheroidite|pearlite>"}
```

Note: Magnification is included from dataset metadata where available, otherwise "unknown".

### A.4 Weld Defect Classification (RIAWELC)

```
Classify this weld radiographic image.

This is a 227×227 pixel 8-bit grayscale radiographic (X-ray) image of a weld joint. Radiography reveals internal defects as variations in image intensity — defects appear as darker or lighter regions compared to the surrounding sound weld metal.

Possible defect classes:
1. lack_of_penetration: A dark continuous or intermittent line/band running along the weld centerline. This indicates the weld root was not fully fused. The dark line is relatively straight and follows the joint geometry.
2. porosity: Scattered small dark circular spots within the weld area. Gas pores are characteristically round or oval. They can appear as individual dots or in clusters. The key feature is the circular shape of each dark spot.
3. cracks: Dark, sharp, irregular jagged lines in the weld. Thinner and more erratic than lack of penetration. May branch or change direction. The edges are sharp and the line path is irregular.
4. no_defect: Clean, uniform weld with no distinct dark spots, lines, or bands. The radiographic image shows relatively even gray intensity across the weld region.

Respond with ONLY a JSON object:
{"defect_class": "<lack_of_penetration|porosity|cracks|no_defect>"}
```

---

## B. CoT Generation Prompts (sent to GPT-4.1)

These prompts are sent to the frontier model along with the training image and the correct label. The model generates only the justification text (no JSON). The last 2 lines of the classification prompt ("Respond with..." and the JSON template) are stripped for CoT training pairs.

### B.1 Steel Surface CoT Generation Prompt (example: crazing)

```
Look at this hot-rolled steel strip surface image. The correct defect classification is: crazing
Definition: a network of fine, shallow cracks spreading across the surface in a web-like or mosaic pattern. The cracks are thin, irregular, and multi-directional. Overall texture looks fragmented but the surface is relatively uniform in brightness

Key distinguishing features between similar classes:
- crazing vs scratches: crazing has a NETWORK of many fine multi-directional cracks (web-like), while scratches are FEW sharp lines in ONE direction
- inclusion vs rolled-in_scale: inclusion appears as dark SPOTS or BLOBS (foreign material trapped during solidification), while rolled-in_scale appears as ELONGATED dark marks/streaks aligned with the rolling direction
- inclusion vs scratches: inclusion has irregularly shaped dark regions on a darker uniform background, while scratches are sharp linear grooves on a lighter surface
- patches vs pitted_surface: patches are LARGE blotchy zones of brightness change, while pitted_surface has SMALL circular dark holes scattered on a brighter surface

Explain WHY this classification is correct based on what you see:
1. Describe the dominant visual pattern — what shapes, textures, brightness patterns do you observe?
2. Explain why these features match crazing using the definition above
3. Explain why it is NOT the most similar class (use the distinguishing features above)

Write 2-3 sentences justifying the classification. Be specific about what you see in THIS image.
Reply with ONLY the justification text. No JSON, no labels, just the explanation.
```

### B.2 Weld Defect CoT Generation Prompt (example: porosity)

```
Look at this weld radiographic (X-ray) image. The correct defect classification is: porosity
Definition: Scattered small dark circular spots within the weld area. Gas pores are characteristically round or oval. They can appear as individual dots or in clusters. The key feature is the circular shape of each dark spot

Key distinguishing features between similar classes:
- lack_of_penetration vs cracks: lack_of_penetration has a STRAIGHT dark line/band along the weld CENTER, while cracks are JAGGED irregular lines that may branch or change direction
- lack_of_penetration vs no_defect: lack_of_penetration has a distinct dark line along the centerline, no_defect has uniform gray with no distinct features
- porosity vs cracks: porosity has CIRCULAR dark spots (round/oval dots), cracks have LINEAR dark features (sharp jagged lines)
- porosity vs no_defect: porosity has scattered dark circular spots, no_defect has uniform intensity with no spots

Explain WHY this classification is correct based on what you see:
1. Describe the dominant dark features — what shapes do you observe? (lines, circles, jagged marks, or uniform)
2. Explain why these features match porosity using the definition above
3. Explain why it is NOT the most similar class

Write 2-3 sentences justifying the classification. Be specific about what you see in THIS image.
Reply with ONLY the justification text. No JSON, no labels, just the explanation.
```

---

## C. Sample CoT Descriptions (GPT-4.1 outputs)

These are actual outputs from GPT-4.1 used in training data, generated at temperature 0.7.

### C.1 Granulometry — Class A16 (16mm, coarse)

> The largest stones in this image appear to be around 16 mm, fitting the max_particle_size_mm: 16 classification. The gaps between the largest stones are mostly empty, with very few smaller particles filling the spaces, which is characteristic of a coarse grading. The overall texture looks uniform and not densely packed with mixed sizes, supporting the classification of coarse aggregate according to DIN 1045.

### C.2 Steel Surface — Class: crazing

> The image shows a surface covered with a fine, web-like network of shallow cracks that are thin, irregular, and multi-directional, creating a fragmented yet relatively uniform texture and brightness. These visual cues match the definition of crazing, as the cracks form a mosaic pattern across the steel strip. It is not scratches, because there are no few sharp lines in a single direction; instead, the cracks form an interconnected network characteristic of crazing.

### C.3 UHCS Microstructure — Class: spheroidite

> The dominant visual pattern in this image is a field of scattered, isolated, round to oval-shaped particles distributed uniformly across a lighter matrix, resembling polka dots. These features match the definition of spheroidite, where cementite appears as discrete, spheroidal particles embedded in a ferrite matrix. There are no continuous dark lines (which would indicate a network structure), no lamellar fingerprint-like striations (as in pearlite), and no straight needle or plate-like features growing from boundaries (as in spheroidite+widmanstatten).

### C.4 Weld Defects — Class: lack_of_penetration

> The image shows a straight, continuous dark band running along the centerline of the weld, which matches the definition of lack_of_penetration. This feature is distinct because it is uniform and follows the geometry of the joint, rather than being jagged or irregular. It is not a crack, as there are no jagged or branching lines, and it is not porosity, since there are no circular dark spots; the uniform gray outside the centerline also rules out other defect types.

---

## D. Sample Training Data (JSONL format)

### D.1 Direct LoRA training pair (weld defects)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "../../datasets/riawelc/training/lack_of_penetration/RRT-13R_Img1_A80_S4_[16][12].png"},
        {"type": "text", "text": "Classify this weld radiographic image.\n\nThis is a 227×227 pixel 8-bit grayscale radiographic (X-ray) image of a weld joint...\n\nRespond with ONLY a JSON object:\n{\"defect_class\": \"<lack_of_penetration|porosity|cracks|no_defect>\"}"}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"defect_class\": \"lack_of_penetration\"}"
    }
  ]
}
```

### D.2 CoT-augmented training pair (weld defects, same image)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "../../datasets/riawelc/training/lack_of_penetration/RRT-13R_Img1_A80_S4_[16][12].png"},
        {"type": "text", "text": "Classify this weld radiographic image.\n\nThis is a 227×227 pixel 8-bit grayscale radiographic (X-ray) image of a weld joint...\n4. no_defect: Clean, uniform weld with no distinct dark spots, lines, or bands. The radiographic image shows relatively even gray intensity across the weld region."}
      ]
    },
    {
      "role": "assistant",
      "content": "The image shows a straight, continuous dark band running along the centerline of the weld, which matches the definition of lack_of_penetration. This feature is distinct because it is uniform and follows the geometry of the joint, rather than being jagged or irregular. It is not a crack, as there are no jagged or branching lines, and it is not porosity, since there are no circular dark spots; the uniform gray outside the centerline also rules out other defect types.\n{\"defect_class\": \"lack_of_penetration\"}"
    }
  ]
}
```

Note: In the CoT pair, the user prompt ends after the class definitions (the "Respond with ONLY a JSON object" line is stripped). The assistant response contains the justification text followed by a newline and the correct JSON, which is appended programmatically.

---

## E. Training Code (Core Functions)

### E.1 Dataset Class

```python
class TrainingDataset(Dataset):
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
        if not isinstance(assistant_text, str):
            assistant_text = json.dumps(assistant_text)
        
        image = Image.open(img_path).convert('RGB') if img_path else None
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
        
        # Mask everything except assistant response
        ast_tokens = self.processor.tokenizer.encode(assistant_text, add_special_tokens=False)
        if len(ast_tokens) < len(labels):
            labels[:-len(ast_tokens)] = -100
        
        if image: image.close()
        return {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels,
            'pixel_values': inputs.get('pixel_values', None),
            'image_grid_thw': inputs.get('image_grid_thw', None)
        }
```

### E.2 Training Loop

```python
EPOCHS = 40
LR = 2e-5
GRAD_ACCUM = 4

def train_lora(base_model, data_path, output_dir):
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    dataset = TrainingDataset(data_path, processor)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = max(len(dataset) * EPOCHS // GRAD_ACCUM, 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i in range(len(dataset)):
            batch = dataset[i]
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
            
            if (i+1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            del ids, mask, lab, out, loss
            torch.cuda.empty_cache()
    
    model.save_pretrained(output_dir)
    return model
```

### E.3 LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type='CAUSAL_LM',
    bias='none'
)
# Trainable params: 37,152,768 / 3,791,775,744 total (0.98%)
```


---

## F. Training Convergence Data

All models use cosine learning rate scheduling with 10% warmup. Direct LoRA models learn shorter JSON-only outputs and start at lower loss. Augmented models learn longer CoT descriptions and start with higher initial loss but converge smoothly.

### F.1 Granulometry

| Epoch | Direct Loss | Augmented Loss |
|-------|-------------|----------------|
| 1 | 0.1859 | 1.3712 |
| 5 | 0.1553 | 0.4422 |
| 10 | 0.1308 | 0.3514 (final*) |
| 15 | 0.1131 | — |
| 20 | 0.0801 | — |
| 25 | 0.0469 | — |
| 30 | 0.0303 | — |
| 35 | 0.0225 | — |
| 40 | 0.0202 | — |

*Granulometry augmented was trained for 10 epochs only (72 examples × 10 = 720 gradient steps), while Direct ran for 40 epochs (18 examples × 40 = 720 gradient steps). Both see approximately the same total gradient steps.

### F.2 Steel Surface (Augmented)

| Epoch | Loss | Elapsed (s) |
|-------|------|-------------|
| 1 | 1.5525 | 264 |
| 6 | 0.6530 | 1,574 |
| 12 | 0.2992 | 3,149 |
| 18 | 0.1593 | 4,725 |
| 24 | 0.0756 | 6,300 |
| 30 | 0.0366 | 7,867 |
| 36 | 0.0227 | 9,434 |
| 39 | 0.0216 | 10,218 |

Total training time: 170 minutes on 2×V100 (120 examples, 40 epochs).

### F.3 UHCS Microstructure (Augmented)

| Epoch | Loss | Elapsed (s) |
|-------|------|-------------|
| 1 | 1.4193 | 360 |
| 5 | 0.6019 | 1,792 |
| 10 | 0.3135 | 3,587 |
| 15 | 0.1629 | 5,383 |
| 20 | 0.1175 | 7,186 |
| 25 | 0.0569 | 9,000 |
| 30 | 0.0299 | 10,791 |
| 35 | 0.0177 | 12,582 |
| 40 | 0.0160 | 14,374 |

Total training time: 240 minutes on 2×V100 (120 examples, 40 epochs).

### F.4 Weld Defects

| Epoch | Direct Loss | Augmented Loss |
|-------|-------------|----------------|
| 1 | 0.1739 | 1.3514 |
| 5 | 0.1028 | 0.6513 |
| 10 | 0.0563 | 0.3409 |
| 15 | 0.0235 | 0.1872 |
| 20 | 0.0062 | 0.1129 |
| 25 | 0.0004 | 0.0624 |
| 30 | 0.0002 | 0.0345 |
| 35 | 0.0002 | 0.0237 |
| 40 | 0.0002 | 0.0218 |

Direct converges to near-zero loss by epoch 25 (24 short JSON examples fully memorized). Augmented converges smoothly to 0.022 (96 longer CoT+JSON examples provide more diversity).

### F.5 Convergence Observations

1. **Direct models** start with low loss (0.17–0.19) since targets are short JSON strings (10–30 tokens). They converge to near-zero (0.0002–0.02) by epoch 25–30.
2. **Augmented models** start with higher loss (1.35–1.55) because targets include 50–150 token descriptions plus JSON. They converge to 0.016–0.035 by epoch 40.
3. All models follow smooth cosine decay patterns with no training instabilities.
4. The weld Direct model reaches 0.0002 loss, indicating complete memorization of 24 JSON outputs — this is expected and motivates the augmented approach's diversity.

---

## G. Granulometry Per-Class Breakdown (9 Classes)

The granulometry task classifies along two axes (size × grading = 9 classes). Table below shows per-class accuracy for both approaches.

| Class | N | Direct (Size/Grading/Both) | CoT-Aug (Size/Grading/Both) |
|-------|---|----------------------------|------------------------------|
| A8 (8mm, coarse) | 12 | 100% / 100% / 100% | 100% / 83% / 83% |
| A16 (16mm, coarse) | 12 | 100% / 92% / 92% | 100% / 92% / 92% |
| A32 (32mm, coarse) | 12 | 100% / 100% / 100% | 92% / 100% / 92% |
| B8 (8mm, medium) | 12 | 100% / 67% / 67% | 100% / 58% / 58% |
| B16 (16mm, medium) | 12 | 100% / 67% / 67% | 100% / 83% / 83% |
| B32 (32mm, medium) | 12 | 92% / 67% / 58% | 100% / 100% / 100% |
| C8 (8mm, fine) | 12 | 100% / 100% / 100% | 100% / 100% / 100% |
| C16 (16mm, fine) | 12 | 58% / 92% / 50% | 67% / 92% / 67% |
| C32 (32mm, fine) | 12 | 58% / 25% / 8% | 67% / 67% / 42% |
| **Overall** | **108** | **89.8% / 78.7% / 71.3%** | **91.7% / 86.1% / 79.6%** |

**Key observations:**
- Size classification is easier (90–92% overall) than grading (79–86%). Both methods achieve near-perfect size accuracy for 8mm and 16mm classes.
- The largest improvement from CoT augmentation is on grading (+7.4pp), particularly for medium and coarse distinctions (B16: 67%→83%, B32: 67%→100%).
- C32 (32mm, fine) is the hardest class for both methods (8%→42% "both correct"). Large stones with fine grading is an unusual combination that may be underrepresented.
- Coarse grading (A classes) is nearly perfect for both methods. The CoT descriptions explicitly teach "gaps are EMPTY" which is a clear, unambiguous visual feature.

---

## H. UHCS Dataset Note

The UHCS dataset contains 6 microconstituent classes in total. We train and evaluate on 5 classes: spheroidite, network, spheroidite+widmanstatten, pearlite+spheroidite, and pearlite. The 6th class (pearlite+widmanstatten) was excluded because only 5 total images exist in the dataset, providing insufficient samples for both training (minimum 6 per class) and testing. The 3 pearlite+widmanstatten images that appear in the test pool are excluded from evaluation. All reported accuracy metrics are computed on the 117-image test set spanning the 5 trained classes.

---

## I. Evaluation Code

### I.1 Inference and JSON Extraction

```python
def evaluate_single(model, processor, image_path, prompt, max_new_tokens=256):
    """Run inference on a single test image and extract JSON from response."""
    image = Image.open(image_path).convert('RGB')
    
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
        )
    
    # Decode only the generated tokens
    generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def extract_json(response_text):
    """Extract JSON from model response, handling both direct and CoT formats."""
    # Try direct JSON parse first
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Look for JSON within text (CoT format: description followed by JSON)
    import re
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, response_text)
    
    # Try matches from last to first (JSON typically at end)
    for match in reversed(matches):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None
```

### I.2 Batch Evaluation Loop

```python
def evaluate_all(model, processor, test_manifest, prompt_fn):
    """Evaluate model on full test set, return per-image results."""
    results = []
    
    for item in test_manifest:
        image_path = item['path']
        ground_truth = item['label']
        
        response = evaluate_single(model, processor, image_path, prompt_fn(item))
        predicted = extract_json(response)
        
        results.append({
            'image': image_path,
            'class': ground_truth,
            'predicted': predicted,
            'raw': response,
            'correct': predicted == ground_truth if predicted else False,
            'valid_json': predicted is not None,
        })
    
    accuracy = sum(r['correct'] for r in results) / len(results) * 100
    json_validity = sum(r['valid_json'] for r in results) / len(results) * 100
    
    return results, accuracy, json_validity
```

---

## J. Model and Hardware Details

| Component | Specification |
|-----------|--------------|
| Base model | Qwen2.5-VL-3B-Instruct |
| Model parameters | 3,791,775,744 total |
| LoRA trainable parameters | 37,152,768 (0.98% of total) |
| Training precision | BF16 (no quantization) |
| Training hardware | 2× NVIDIA Tesla V100 16GB PCIe |
| Memory allocation | GPU0: 6 GiB, GPU1: 15 GiB |
| Gradient checkpointing | Enabled |
| Timing hardware | 1× NVIDIA L4 24GB (for single-GPU benchmarks) |
| Frontier model | GPT-4.1 (Azure AI Foundry) |
| CoT generation temperature | 0.7 |
| CoT max tokens | 512 |
| Evaluation temperature | 0.1 |
| Evaluation max tokens | 256 |
| Random seed | 42 (for training image selection and model init) |
