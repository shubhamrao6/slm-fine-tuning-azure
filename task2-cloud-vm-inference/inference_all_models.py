"""
Task 2: Run inference on all 4 models and compare outputs.

Usage:
    python inference_all_models.py <image_path>
    python inference_all_models.py <image_path> --prompt "Count the objects"
"""
import sys
import time
import torch
from PIL import Image


def load_florence2():
    from transformers import AutoModelForCausalLM, AutoProcessor
    print("Loading Florence-2-large...")
    t = time.time()
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large-ft",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")
    print(f"  Loaded in {time.time()-t:.1f}s")
    return model, processor


def load_qwen(size="3B"):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model_id = f"Qwen/Qwen2.5-VL-{size}-Instruct"
    print(f"Loading Qwen2.5-VL-{size}...")
    t = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Loaded in {time.time()-t:.1f}s")
    return model, processor


def load_phi4mm():
    from transformers import AutoModelForCausalLM, AutoProcessor
    print("Loading Phi-4-multimodal...")
    t = time.time()
    processor = AutoProcessor.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        _attn_implementation="eager",
    )
    print(f"  Loaded in {time.time()-t:.1f}s")
    return model, processor


# --- Inference functions ---

def infer_florence2(model, processor, image, task="<OD>"):
    """Florence-2 uses task prompts like <OD>, <CAPTION>, <DENSE_REGION_CAPTION>."""
    inputs = processor(text=task, images=image, return_tensors="pt").to("cuda", torch.float16)
    t = time.time()
    ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, num_beams=3)
    elapsed = time.time() - t
    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(text, task=task, image_size=(image.width, image.height))
    return parsed, elapsed


def infer_qwen(model, processor, image, prompt="Describe this image."):
    """Qwen2.5-VL uses chat format with image tokens."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
    t = time.time()
    ids = model.generate(**inputs, max_new_tokens=256)
    elapsed = time.time() - t
    output = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return output, elapsed


def infer_phi4mm(model, processor, image, prompt="Describe this image."):
    """Phi-4-multimodal uses special image tokens."""
    full_prompt = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
    inputs = processor(full_prompt, images=[image], return_tensors="pt")
    # Filter None values (audio fields) and move to cuda
    generate_kwargs = {k: v.to("cuda") for k, v in inputs.items() if v is not None}
    generate_kwargs["max_new_tokens"] = 256
    generate_kwargs["num_logits_to_keep"] = 1
    t = time.time()
    ids = model.generate(**generate_kwargs)
    elapsed = time.time() - t
    output = processor.batch_decode(ids, skip_special_tokens=True)[0]
    if "<|assistant|>" in output:
        output = output.split("<|assistant|>")[-1].strip()
    return output, elapsed


# --- Main ---

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference_all_models.py <image_path> [--prompt 'your prompt']")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = "Describe what you see in this image."
    for i, arg in enumerate(sys.argv[2:]):
        if arg == "--prompt" and i + 1 < len(sys.argv) - 2:
            prompt = sys.argv[i + 3]

    image = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path} ({image.width}x{image.height})")
    print(f"Prompt: {prompt}\n")
    print("=" * 70)

    # Florence-2 (uses task prompts, not free-form text)
    f2_model, f2_proc = load_florence2()
    for task in ["<OD>", "<CAPTION>", "<DENSE_REGION_CAPTION>"]:
        result, elapsed = infer_florence2(f2_model, f2_proc, image, task)
        print(f"\n[Florence-2] Task: {task} ({elapsed:.2f}s)")
        print(f"  {result}")
    del f2_model, f2_proc
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)

    # Qwen2.5-VL-3B
    q3_model, q3_proc = load_qwen("3B")
    result, elapsed = infer_qwen(q3_model, q3_proc, image, prompt)
    print(f"\n[Qwen2.5-VL-3B] ({elapsed:.2f}s)")
    print(f"  {result}")
    del q3_model, q3_proc
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)

    # Qwen2.5-VL-7B
    q7_model, q7_proc = load_qwen("7B")
    result, elapsed = infer_qwen(q7_model, q7_proc, image, prompt)
    print(f"\n[Qwen2.5-VL-7B] ({elapsed:.2f}s)")
    print(f"  {result}")
    del q7_model, q7_proc
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)

    # Phi-4-multimodal
    phi_model, phi_proc = load_phi4mm()
    result, elapsed = infer_phi4mm(phi_model, phi_proc, image, prompt)
    print(f"\n[Phi-4-multimodal] ({elapsed:.2f}s)")
    print(f"  {result}")
    del phi_model, phi_proc
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
