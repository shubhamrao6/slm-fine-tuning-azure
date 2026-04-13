"""
LoRA fine-tuning of Qwen2.5-VL-3B for granulometry classification.
Uses full BF16 precision (not QLoRA) for maximum quality.

Usage:
    # Approach A: Standard LoRA (18 direct examples)
    python fine_tune.py --data training_data_direct.jsonl --output lora_direct

    # Approach B: SEAL-inspired (augmented examples)
    python fine_tune.py --data training_data_augmented.jsonl --output lora_augmented

    # Custom settings
    python fine_tune.py --data training_data_direct.jsonl --output lora_direct --epochs 30 --lr 1e-5
"""
import os, sys, json, torch
from PIL import Image

# --- Config ---
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_EPOCHS = 20
DEFAULT_LR = 2e-5
DEFAULT_BATCH = 1
DEFAULT_GRAD_ACCUM = 4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    data_path = "training_data_direct.jsonl"
    output_dir = "lora_direct"
    epochs = DEFAULT_EPOCHS
    lr = DEFAULT_LR

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--data" and i + 1 < len(args): data_path = args[i + 1]
        if arg == "--output" and i + 1 < len(args): output_dir = args[i + 1]
        if arg == "--epochs" and i + 1 < len(args): epochs = int(args[i + 1])
        if arg == "--lr" and i + 1 < len(args): lr = float(args[i + 1])

    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}, LR: {lr}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}")

    with open(data_path) as f:
        train_data = [json.loads(line) for line in f]
    print(f"Training examples: {len(train_data)}")

    # Load model in full BF16 precision (not quantized)
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    print("Loading model (BF16 full precision)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.enable_input_require_grads()  # required for LoRA training

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.1f} GB")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=DEFAULT_BATCH,
        gradient_accumulation_steps=DEFAULT_GRAD_ACCUM,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
    )

    print(f"\nConfig ready:")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"  Effective batch size: {DEFAULT_BATCH * DEFAULT_GRAD_ACCUM}")
    print(f"  Total steps: ~{len(train_data) * epochs // (DEFAULT_BATCH * DEFAULT_GRAD_ACCUM)}")

    # TODO: Implement the training loop.
    # The multimodal data collator for Qwen2.5-VL requires specific handling
    # of image tokens. Two options:
    #
    # Option 1: Use trl.SFTTrainer with a custom data collator
    # Option 2: Manual training loop with processor + model.forward()
    #
    # The exact implementation depends on library versions on the VM.
    # Run: pip list | grep -E "trl|peft|transformers"
    # See: https://connectaman.hashnode.dev/fine-tuning-the-qwen25-7b-vl-instruct-model
    print(f"\n[TODO] Training loop — see README for references")
    print(f"Model and LoRA config ready. Output will be saved to {output_dir}/")


if __name__ == "__main__":
    main()
