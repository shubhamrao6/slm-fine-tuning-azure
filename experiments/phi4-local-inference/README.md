# Task 2: Run Phi-4-Multimodal Locally

## Your PC Specs

| Component | Value |
|-----------|-------|
| Laptop | MSI Katana GF66 12UD |
| OS | Windows 11 Home |
| CPU | Intel 12th Gen (i7-12700H) |
| RAM | 16 GB |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop — 4 GB VRAM |
| CUDA | 11.8 |
| Driver | 522.06 |

## Phi-4-Multimodal Hardware Requirements

### Before Quantization (Original FP16 Model)

| | Value |
|---|---|
| Model size on disk | ~11.2 GB |
| VRAM needed | ~12 GB |
| RAM needed | 16 GB+ |
| Runs on your PC (4 GB VRAM)? | No — not even close |

### After Quantization (INT4)

Quantization shrinks the model by ~3x and makes it ~2x faster, with only ~2-3% quality loss.

| | Value |
|---|---|
| Model size on disk | ~3.5 GB |
| VRAM needed | ~4-5 GB |
| RAM needed | 8 GB+ |
| Runs on your PC (4 GB VRAM)? | Barely — model fits but may OOM when processing images (vision encoder + KV cache need extra VRAM on top of the 3.5 GB model) |
| CPU-only mode (no VRAM needed) | Yes, uses ~8 GB RAM, but slow (~2-5 tok/s) |

## What Can You Run Locally?

### Phi-4-multimodal (5.6B, multimodal) — Tight fit
- INT4 quantized model is ~3.5 GB, your GPU has 4 GB
- Leaves almost no room for vision encoder, KV cache, and image processing
- May load but likely crashes when processing images
- CPU-only mode works but expect 10-20 seconds per response

### Phi-4-mini (3.8B, text-only) — Comfortable fit ✓
- INT4 quantized is ~2 GB, fits easily in 4 GB VRAM
- Fast inference, room for KV cache
- No image/audio support though

### Recommended approach for your hardware
- Use Phi-4-mini locally for text tasks (fast, fits in VRAM)
- Use the serverless API for multimodal tasks (images, audio)
- Use an Azure GPU VM (NC4as_T4_v3, 16 GB VRAM, $0.53/hr) when you need local-like control with more power

## Inference Options

### Option A: ONNX Runtime with INT4 (Best for Windows)

Microsoft publishes an official INT4 ONNX model for local use.

```bash
pip install onnxruntime-genai torch transformers

# Download the ONNX model (~3.5 GB)
git lfs install
git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct-onnx
```

```python
import onnxruntime_genai as og

model = og.Model("./Phi-4-multimodal-instruct-onnx")
tokenizer = og.Tokenizer(model)
params = og.GeneratorParams(model)
params.set_search_options(max_length=2048)

prompt = "<|user|>\nDescribe the solar system<|end|>\n<|assistant|>\n"
input_tokens = tokenizer.encode(prompt)
params.input_ids = input_tokens

generator = og.Generator(model, params)
output_tokens = []
while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()
    output_tokens.append(generator.get_next_tokens()[0])

print(tokenizer.decode(output_tokens))
```

### Option B: Hugging Face Transformers (FP16 — needs 12 GB+ VRAM)

Won't run on your GPU but included for reference (use on Azure VM or better hardware).

```bash
pip install torch transformers accelerate flash-attn soundfile pillow
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

model_id = "microsoft/Phi-4-multimodal-instruct"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# Image + Text inference
image = Image.open("photo.jpg")
prompt = "<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"

inputs = processor(prompt, images=[image], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
print(processor.batch_decode(output, skip_special_tokens=True)[0])
```

### Option C: Ollama (when supported)

Ollama does not natively support Phi-4-multimodal yet. Check `ollama.com/library` for updates.

For Phi-4-mini (text-only), which works on your hardware:
```bash
ollama run phi4-mini
```

## Expected Performance on Your PC (RTX 3050 Ti, 4 GB)

All speeds below are after INT4 quantization (the only viable option for local GPU inference).
FP16 (unquantized) won't fit on any GPU with less than 12 GB VRAM.

| Model | Mode | Speed | Feasible? |
|-------|------|-------|-----------|
| Phi-4-multimodal (INT4) | GPU | ~15-25 tok/s | Risky (OOM on images) |
| Phi-4-multimodal (INT4) | CPU | ~2-5 tok/s | Yes, slow |
| Phi-4-mini (INT4) | GPU | ~30-50 tok/s | Yes, comfortable |

## Expected Performance on Better Hardware

All figures are after INT4 quantization unless marked FP16.

| GPU | VRAM | INT4 (quantized) | FP16 (original, unquantized) |
|-----|------|-----------------|------------------------------|
| RTX 3050 Ti (yours) | 4 GB | ~15-25 tok/s (risky) | Won't fit |
| RTX 3060 | 12 GB | ~40-60 tok/s | ~20-30 tok/s |
| RTX 4060 | 8 GB | ~60-80 tok/s | Won't fit |
| RTX 4070 | 12 GB | ~80-100 tok/s | ~30-40 tok/s |
| RTX 4090 | 24 GB | ~150+ tok/s | ~60-80 tok/s |
| Azure T4 (NC4as_T4_v3) | 16 GB | ~40-60 tok/s | ~20-30 tok/s |

## Sub-Second Response — Is It Possible?

For a short response (~80 tokens, like a JSON detection output).
All figures are after INT4 quantization. FP16 would be roughly half the speed.

| Hardware | INT4 Speed | Time for 80 tokens | Sub-second? |
|----------|-----------|-------------------|-------------|
| Your RTX 3050 Ti | ~15-25 tok/s | 3-5 sec | No |
| RTX 3060 | ~40-60 tok/s | 1.3-2 sec | No |
| RTX 4060/4070 | ~80-100 tok/s | 0.8-1 sec | Borderline |
| RTX 4090 | ~150+ tok/s | ~0.5 sec | Yes |

Note: Multimodal adds ~200-500ms for image encoding on top of token generation. This overhead exists regardless of quantization.

## Video Frame Extraction (for multimodal)

Phi-4-multimodal doesn't process video natively. Extract frames first:

```python
import cv2
from PIL import Image

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames
```
