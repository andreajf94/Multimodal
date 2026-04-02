#!/usr/bin/env python3
"""SFT warmup for RepoDesign using Tinker SDK.

Trains the model on (prompt, teacher_plan) pairs so it learns the JSON format
before GRPO fine-tuning. This dramatically improves format compliance.

Usage:
    python scripts/train_sft_tinker.py data/commit_pairs_production -v
    python scripts/train_sft_tinker.py data/commit_pairs_production --epochs 1 --lr 1e-4

After SFT, use the checkpoint as starting point for GRPO:
    python scripts/train_grpo_tinker.py data/commit_pairs_production --checkpoint output/sft_training/final
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData

import torch
import wandb

from repodesign.training.reward import parse_diff_files
from repodesign.training.vl_renderer import Qwen3VLRenderer, load_diagram_images
from repodesign.training.data_gen import summarize_repo_ir_for_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class SFTConfig:
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    lora_rank: int = 64
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_ratio: float = 0.03
    batch_size: int = 4
    num_epochs: int = 1
    save_every: int = 20
    log_path: str = "output/sft_training"
    use_diagrams: bool = True
    max_grad_norm: float = 1.0
    max_samples: int | None = None


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, config: SFTConfig) -> float:
    warmup_steps = int(total_steps * config.warmup_ratio)
    if step < warmup_steps:
        return config.learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine_decay


# ---------------------------------------------------------------------------
# Prompt building (same as GRPO but returns prompt + target separately)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior software architect. Given a codebase analysis and a feature specification, generate a detailed implementation plan as JSON.

Return a JSON object with these fields:
{
  "architecture_decisions": [{"dimension": "...", "recommendation": "...", "rationale": "...", "alternatives_considered": [...], "files_affected": [...]}],
  "tickets": [{"id": "T-001", "title": "...", "description": "...", "files_to_modify": [...], "files_to_create": [...], "estimated_effort": "small|medium|large", "dependencies": [...]}],
  "implementation_summary": "2-3 paragraph explanation of the overall approach, key patterns used, and how the changes integrate with the existing codebase"
}

IMPORTANT:
- files_to_modify MUST reference existing file paths from the codebase file manifest
- files_to_create should be new paths that don't exist yet
- Generate 3-8 architecture decisions and 4-10 actionable tickets
- implementation_summary should explain the overall approach, key patterns used, and how changes integrate with existing code"""


def build_sft_example(
    repo_ir_summary: str,
    spec: dict,
    file_manifest: list[str],
    teacher_plan: dict,
    renderer: Qwen3VLRenderer,
    diagram_images: list[bytes] | None = None,
) -> types.Datum | None:
    """Build a Tinker SFT Datum from (prompt, teacher_plan) pair.

    Returns a Datum with the prompt as prefix (masked) and teacher plan as target.
    """
    manifest_str = "\n".join(file_manifest[:500])

    user_text = f"""## Codebase Analysis
{repo_ir_summary}

## File Manifest (all files in the repository)
{manifest_str}

## Feature Specification
Project: {spec.get('project_name', 'Unknown')}
Feature: {spec.get('feature_name', 'Unknown')}
Description: {spec.get('description', '')}

Requirements:
{chr(10).join(f'- {r}' for r in spec.get('functional_requirements', []))}

{f"Scale: {spec['scale_tier']}" + chr(10) if spec.get('scale_tier') else ''}Generate an implementation plan following the JSON schema above. Reference real file paths from the manifest for files_to_modify."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    # Build the prompt (input up to assistant turn)
    prompt = renderer.build_generation_prompt(messages, diagram_images=diagram_images)

    # Encode the teacher plan as the target completion
    target_text = json.dumps(teacher_plan, indent=2)
    target_tokens = renderer.tokenizer.encode(target_text)

    if len(target_tokens) < 2:
        return None

    # Truncate if total sequence would exceed model's max (32768)
    MAX_SEQ_LEN = 32768
    total_len = prompt.length + len(target_tokens)
    if total_len > MAX_SEQ_LEN:
        available = MAX_SEQ_LEN - prompt.length - 1
        if available < 50:
            return None  # Prompt alone is too long
        target_tokens = target_tokens[:available]

    # Build full sequence: prompt + target
    ob_len = prompt.length - 1
    model_input = prompt.append(types.EncodedTextChunk(tokens=target_tokens[:-1]))

    # For SFT: mask the prompt tokens (0), only compute loss on target tokens
    full_target = [0] * ob_len + target_tokens
    # Uniform logprobs (not used in SFT but required by Datum format)
    logprobs = [0.0] * model_input.length
    # All advantages = 1.0 for SFT (uniform weighting on target tokens)
    advantages = [0.0] * ob_len + [1.0] * (model_input.length - ob_len)

    assert model_input.length == len(full_target) == len(logprobs) == len(advantages)

    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(full_target)),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs)),
            "advantages": TensorData.from_torch(torch.tensor(advantages)),
        },
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_examples(repo_irs_dir: str, use_diagrams: bool = True) -> list[dict]:
    """Load all training examples."""
    examples = []
    base = Path(repo_irs_dir)

    for repo_dir in sorted(base.iterdir()):
        if not repo_dir.is_dir():
            continue

        repo_ir_path = repo_dir / "repo_ir.json"
        spec_path = repo_dir / "spec.json"
        plan_path = repo_dir / "teacher_plan.json"

        if not all(p.exists() for p in [repo_ir_path, spec_path, plan_path]):
            continue

        try:
            with open(repo_ir_path, encoding="utf-8") as f:
                repo_ir = json.load(f)
            with open(spec_path, encoding="utf-8") as f:
                spec = json.load(f)
            with open(plan_path, encoding="utf-8") as f:
                teacher_plan = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {repo_dir.name}: {e}")
            continue

        diagram_images = []
        if use_diagrams:
            diagram_images = load_diagram_images(repo_ir, str(repo_dir))

        examples.append({
            "repo_name": repo_dir.name,
            "repo_ir": repo_ir,
            "repo_ir_summary": summarize_repo_ir_for_prompt(repo_ir),
            "spec": spec,
            "teacher_plan": teacher_plan,
            "file_manifest": repo_ir.get("file_manifest", []),
            "diagram_images": diagram_images,
        })

    return examples


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: SFTConfig, repo_irs_dir: str):
    """Run SFT training loop."""
    logger.info(f"Loading training examples from {repo_irs_dir}...")
    examples = load_examples(repo_irs_dir, use_diagrams=config.use_diagrams)
    if not examples:
        print("ERROR: No complete training examples found.")
        sys.exit(1)

    if config.max_samples is not None:
        examples = examples[:config.max_samples]

    random.seed(42)
    random.shuffle(examples)
    print(f"Loaded {len(examples)} SFT examples")

    # Setup Tinker
    logger.info(f"Connecting to Tinker with model {config.model_name}...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    tokenizer = training_client.get_tokenizer()
    renderer = Qwen3VLRenderer(tokenizer)

    n_batches = max(1, len(examples) // config.batch_size)
    total_steps = n_batches * config.num_epochs

    # W&B logging
    wandb.init(
        project="repodesign-grpo",
        name="sft-warmup",
        config={
            "stage": "sft",
            "model_name": config.model_name,
            "lora_rank": config.lora_rank,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "n_examples": len(examples),
            "total_steps": total_steps,
        },
    )

    log_path = Path(config.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    metrics_file = open(log_path / "metrics.jsonl", "a")

    print(f"\nSFT config:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  LR: {config.learning_rate} -> {config.min_learning_rate} (cosine)")
    print(f"  Epochs: {config.num_epochs}")
    print()

    global_step = 0

    for epoch in range(config.num_epochs):
        print(f"\n{'='*60}")
        print(f"SFT Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*60}")

        random.shuffle(examples)

        for batch_idx in range(n_batches):
            t_start = time.time()
            current_lr = get_lr(global_step, total_steps, config)

            adam_params = types.AdamParams(
                learning_rate=current_lr,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )

            start = batch_idx * config.batch_size
            end = min(start + config.batch_size, len(examples))
            batch = examples[start:end]

            datums = []
            for ex in batch:
                try:
                    datum = build_sft_example(
                        repo_ir_summary=ex["repo_ir_summary"],
                        spec=ex["spec"],
                        file_manifest=ex["file_manifest"],
                        teacher_plan=ex["teacher_plan"],
                        renderer=renderer,
                        diagram_images=ex["diagram_images"] if config.use_diagrams else None,
                    )
                    if datum is not None:
                        datums.append(datum)
                except Exception as e:
                    logger.warning(f"Failed to build SFT example for {ex.get('repo_name', '?')}: {e}")

            if not datums:
                global_step += 1
                continue

            # Forward-backward + optimizer step
            try:
                fwd_bwd_future = training_client.forward_backward(
                    datums,
                    loss_fn="importance_sampling",
                )
                optim_future = training_client.optim_step(adam_params)
                fwd_bwd_result = fwd_bwd_future.result()
                optim_result = optim_future.result()

                metrics = {
                    "progress/epoch": epoch,
                    "progress/global_step": global_step,
                    "optim/lr": current_lr,
                    "training/n_datums": len(datums),
                    "time/total": time.time() - t_start,
                }
                if optim_result.metrics:
                    metrics.update(optim_result.metrics)
                if "loss" in metrics:
                    metrics["train/loss"] = metrics["loss"]

                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()
                wandb.log(metrics, step=global_step)

                loss_str = f"loss={metrics.get('train/loss', '?'):.4f}" if 'train/loss' in metrics else ""
                print(f"  Step {global_step}: {loss_str} lr={current_lr:.2e} "
                      f"datums={len(datums)} time={metrics['time/total']:.1f}s")

            except Exception as e:
                logger.error(f"SFT step failed at batch {batch_idx}: {e}")
                global_step += 1
                continue

            # Checkpoint
            if config.save_every > 0 and global_step % config.save_every == 0 and global_step > 0:
                try:
                    from tinker_cookbook import checkpoint_utils
                    checkpoint_utils.save_checkpoint(
                        training_client=training_client,
                        name=f"sft_{global_step:06d}",
                        log_path=config.log_path,
                        kind="state",
                        loop_state={"step": global_step},
                    )
                    logger.info(f"  Saved SFT checkpoint at step {global_step}")
                except Exception as e:
                    logger.warning(f"  Checkpoint save failed: {e}")

            global_step += 1

    # Save final checkpoint
    try:
        from tinker_cookbook import checkpoint_utils
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="sft_final",
            log_path=config.log_path,
            kind="both",
            loop_state={"step": global_step},
        )
        print(f"\nSFT complete! Final checkpoint saved to {config.log_path}")
    except Exception as e:
        logger.warning(f"Final checkpoint save failed: {e}")

    metrics_file.close()
    wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SFT warmup for RepoDesign via Tinker")
    parser.add_argument("repo_irs_dir", help="Directory with per-repo training data")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Instruct", help="Base model")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Examples per batch")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--save-every", type=int, default=20, help="Save checkpoint every N steps")
    parser.add_argument("--log-path", default="output/sft_training", help="Log/checkpoint directory")
    parser.add_argument("--no-diagrams", action="store_true", help="Disable diagram images")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training examples")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = SFTConfig()
    config.model_name = args.model
    config.lora_rank = args.lora_rank
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.save_every = args.save_every
    config.log_path = args.log_path
    config.use_diagrams = not args.no_diagrams
    config.max_samples = args.max_samples

    train(config, args.repo_irs_dir)


if __name__ == "__main__":
    main()
