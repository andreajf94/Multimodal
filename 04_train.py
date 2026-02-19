#!/usr/bin/env python3
"""
04_train.py

QLoRA fine-tuning of Qwen2-VL-7B-Instruct using Unsloth.
Run this on a GPU machine (A100 40GB+ recommended).

Setup on GPU machine:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes
    pip install qwen-vl-utils

Usage:
    python 04_train.py                          # Default settings
    python 04_train.py --epochs 2 --lr 1e-4     # Custom
    python 04_train.py --resume output/checkpoints/checkpoint-500  # Resume

Time: ~4-8 hours on A100 40GB for 1000-2000 examples, 3 epochs
Cost: ~$5-10 on Vast.ai/RunPod
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL-7B for systems design")
    parser.add_argument("--model", default="unsloth/Qwen2-VL-7B-Instruct",
                        help="Base model (Unsloth version for 2x speed)")
    parser.add_argument("--data", default="data/train.jsonl",
                        help="Training data file")
    parser.add_argument("--output", default="output/checkpoints",
                        help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (2e-4 is standard for LoRA)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size (reduce if OOM)")
    parser.add_argument("--grad-accum", type=int, default=16,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (32 for budget, 64-128 for quality)")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="Max sequence length")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Load model with Unsloth ───
    print("=" * 60)
    print("STEP 1: Loading model with Unsloth (4-bit quantized)")
    print("=" * 60)

    from unsloth import FastVisionModel
    import torch

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=True,              # QLoRA: 4-bit base model
        use_gradient_checkpointing="unsloth",  # Saves 60% memory
    )

    # ─── Step 2: Add LoRA adapters ───
    print("=" * 60)
    print("STEP 2: Adding LoRA adapters")
    print("=" * 60)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,     # Keep vision encoder frozen (budget mode)
        finetune_language_layers=True,     # Train language model
        finetune_attention_modules=True,   # Train attention
        finetune_mlp_modules=True,         # Train MLP layers
        r=args.lora_rank,                  # LoRA rank
        lora_alpha=args.lora_rank * 2,     # Standard: 2x rank
        lora_dropout=0.05,
        bias="none",
        random_state=42,
        use_rslora=True,                   # Rank-stabilized LoRA (better convergence)
    )

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ─── Step 3: Load and prepare dataset ───
    print("=" * 60)
    print("STEP 3: Loading training data")
    print("=" * 60)

    from datasets import load_dataset

    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"Training examples: {len(dataset)}")

    # Convert ShareGPT format to the chat template format
    from unsloth.chat_templates import get_chat_template, standardize_sharegpt

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen2-vl",
    )

    dataset = standardize_sharegpt(dataset)

    def format_example(example):
        convos = example["conversations"]
        texts = tokenizer.apply_chat_template(
            convos,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": texts}

    dataset = dataset.map(format_example, batched=False)

    # ─── Step 4: Training ───
    print("=" * 60)
    print(f"STEP 4: Training ({args.epochs} epochs)")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA rank: {args.lora_rank}")
    print("=" * 60)

    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.output,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            fp16=False,
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,          # Keep only last 3 checkpoints (disk space)
            max_seq_length=args.max_seq_len,
            dataset_text_field="text",
            packing=True,                # Pack short examples together (efficiency)
            seed=42,
            report_to="none",            # Set to "wandb" if you want logging
        ),
    )

    # Print training estimate
    total_steps = (len(dataset) * args.epochs) // (args.batch_size * args.grad_accum)
    print(f"  Estimated total steps: {total_steps}")
    print(f"  Estimated save checkpoints: {total_steps // 200}")
    print()

    # Train
    if args.resume:
        print(f"  Resuming from {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ─── Step 5: Save final adapter ───
    print("=" * 60)
    print("STEP 5: Saving final adapter")
    print("=" * 60)

    final_dir = Path(args.output) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Adapter saved to {final_dir}")

    # Print stats
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Training examples: {len(dataset)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Final adapter: {final_dir}")
    print(f"\nNext step: python 05_merge_and_export.py")


if __name__ == "__main__":
    main()
