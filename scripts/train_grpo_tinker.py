#!/usr/bin/env python3
"""GRPO training loop for RepoDesign using Tinker SDK.

Adapted from:
  - tinker-cookbook/recipes/rl_loop.py (Tinker GRPO pattern)
  - deepmind_tunix/general_reasoning (curriculum + reward design)

Usage:
    python scripts/train_grpo_tinker.py data/repo_irs --group-size 4 -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

from repodesign.training.reward import compute_rewards
from repodesign.training.data_gen import summarize_repo_ir_for_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    lora_rank: int = 64
    learning_rate: float = 1e-5
    batch_size: int = 4          # prompts per batch
    group_size: int = 4          # completions per prompt (G in GRPO)
    max_tokens: int = 4096       # max generation length
    num_epochs: int = 1
    save_every: int = 5          # save checkpoint every N batches
    log_path: str = "output/grpo_training"
    use_llm_judge: bool = True   # use DeepSeek as judge (slower but better signal)
    use_diagrams: bool = True    # feed diagram images to VLM


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior software architect. Given a codebase analysis and a feature specification, generate a detailed implementation plan as JSON.

Return a JSON object with these fields:
{
  "architecture_decisions": [{"dimension": "...", "recommendation": "...", "rationale": "...", "alternatives_considered": [...], "files_affected": [...]}],
  "tickets": [{"id": "T-001", "title": "...", "description": "...", "files_to_modify": [...], "files_to_create": [...], "estimated_effort": "small|medium|large", "dependencies": [...]}],
  "technology_choices": [{"category": "...", "choice": "...", "rationale": "..."}]
}

IMPORTANT:
- files_to_modify must reference REAL file paths from the codebase
- files_to_create should follow the project's existing conventions
- Generate 3-8 architecture decisions and 4-10 actionable tickets"""


def build_prompt(
    repo_ir_summary: str,
    spec: dict,
    tokenizer,
    renderer,
    diagram_images: list[bytes] | None = None,
) -> types.ModelInput:
    """Build a Tinker ModelInput prompt from RepoIR + Spec + optional diagrams."""
    # Build the user message content
    user_text = f"""## Codebase Analysis
{repo_ir_summary}

## Feature Specification
Project: {spec.get('project_name', 'Unknown')}
Feature: {spec.get('feature_name', 'Unknown')}
Description: {spec.get('description', '')}

Requirements:
{chr(10).join(f'- {r}' for r in spec.get('functional_requirements', []))}

Scale: {spec.get('scale_tier', 'startup')}

Generate a detailed implementation plan as JSON."""

    # Build conversation
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if diagram_images:
        # Multimodal: include diagram images in the user message
        content_parts = []
        for img_data in diagram_images:
            content_parts.append({"type": "image", "image": img_data})
        content_parts.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": content_parts})
    else:
        messages.append({"role": "user", "content": user_text})

    return renderer.build_generation_prompt(messages)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_examples(repo_irs_dir: str, use_diagrams: bool = True) -> list[dict]:
    """Load all training examples (RepoIR + Spec + teacher plan) from output dir."""
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
            with open(repo_ir_path) as f:
                repo_ir = json.load(f)
            with open(spec_path) as f:
                spec = json.load(f)
            with open(plan_path) as f:
                teacher_plan = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {repo_dir.name}: {e}")
            continue

        # Load diagram images if available
        diagram_images = []
        if use_diagrams:
            for dp in repo_ir.get("diagram_paths", []):
                img_path = repo_dir / dp
                if img_path.exists() and img_path.stat().st_size < 5_000_000:  # skip >5MB
                    try:
                        diagram_images.append(img_path.read_bytes())
                    except Exception:
                        pass

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
# Metrics logging
# ---------------------------------------------------------------------------

class MetricsLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.metrics_file = open(self.log_path / "metrics.jsonl", "a")

    def log(self, metrics: dict, step: int):
        metrics["step"] = step
        self.metrics_file.write(json.dumps(metrics) + "\n")
        self.metrics_file.flush()

        # Print summary
        reward_total = metrics.get("reward/total", 0)
        rgs_mean = metrics.get("reward/rgs_mean", 0)
        fmt_mean = metrics.get("reward/format_mean", 0)
        t = metrics.get("time/total", 0)
        print(f"  Step {step}: reward={reward_total:.3f} rgs={rgs_mean:.3f} fmt={fmt_mean:.3f} time={t:.1f}s")

    def close(self):
        self.metrics_file.close()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: Config, repo_irs_dir: str):
    """Run GRPO training loop."""
    # Load data
    logger.info(f"Loading training examples from {repo_irs_dir}...")
    examples = load_training_examples(repo_irs_dir, use_diagrams=config.use_diagrams)
    if not examples:
        print("ERROR: No complete training examples found. Run generate_training_data.py first.")
        sys.exit(1)
    print(f"Loaded {len(examples)} training examples")

    # Setup Tinker
    logger.info(f"Connecting to Tinker with model {config.model_name}...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    tokenizer = training_client.get_tokenizer()

    # Get appropriate renderer for Qwen3-VL
    # NOTE: tinker-cookbook model_info doesn't have VL models registered,
    # so we use Qwen3Renderer directly (same chat template as text Qwen3).
    from tinker_cookbook.renderers import Qwen3Renderer
    renderer = Qwen3Renderer(tokenizer)
    logger.info(f"Using renderer: Qwen3Renderer")

    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    # Metrics
    ml_logger = MetricsLogger(config.log_path)

    # Calculate batches
    n_batches = len(examples) // config.batch_size
    if n_batches == 0:
        n_batches = 1
        config.batch_size = len(examples)

    print(f"\nTraining config:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Group size: {config.group_size}")
    print(f"  Batches per epoch: {n_batches}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  LLM judge: {config.use_llm_judge}")
    print(f"  Diagrams: {config.use_diagrams}")
    print()

    global_step = 0

    for epoch in range(config.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*60}")

        for batch_idx in range(n_batches):
            t_start = time.time()
            metrics: dict[str, float] = {
                "progress/epoch": epoch,
                "progress/batch": batch_idx,
                "progress/global_step": global_step,
                "optim/lr": config.learning_rate,
            }

            # Get batch
            start = batch_idx * config.batch_size
            end = min(start + config.batch_size, len(examples))
            batch = examples[start:end]

            print(f"\n[Batch {batch_idx + 1}/{n_batches}] Processing {len(batch)} examples...")

            # Create sampling client from current weights
            sampling_client = training_client.save_weights_and_get_sampling_client()

            datums_D: list[types.Datum] = []
            all_rewards: list[float] = []
            all_rgs: list[float] = []
            all_fmt: list[float] = []

            for ex in batch:
                # Build prompt
                prompt = build_prompt(
                    repo_ir_summary=ex["repo_ir_summary"],
                    spec=ex["spec"],
                    tokenizer=tokenizer,
                    renderer=renderer,
                    diagram_images=ex["diagram_images"] if config.use_diagrams else None,
                )

                # Generate G completions
                sample_result = sampling_client.sample(
                    prompt=prompt,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                ).result()

                # Collect completions and their tokens/logprobs
                completions: list[str] = []
                sampled_tokens_G: list[list[int]] = []
                logprobs_G: list[list[float]] = []

                for seq in sample_result.sequences:
                    sampled_tokens_G.append(seq.tokens)
                    logprobs_G.append(seq.logprobs)
                    parsed_msg, _ = renderer.parse_response(seq.tokens)
                    content = parsed_msg.get("content", "")
                    completions.append(content)

                # Compute rewards for all completions
                reward_results = compute_rewards(
                    completions=completions,
                    file_manifests=[ex["file_manifest"]] * len(completions),
                    specs=[ex["spec"]] * len(completions),
                    teacher_plans=[ex["teacher_plan"]] * len(completions),
                    use_llm_judge=config.use_llm_judge,
                )

                rewards_G = [r["total"] for r in reward_results]
                mean_reward = sum(rewards_G) / len(rewards_G)
                advantages_G = [r - mean_reward for r in rewards_G]

                # Track metrics
                all_rewards.append(mean_reward)
                all_rgs.extend(r["rgs_score"] for r in reward_results)
                all_fmt.extend(r["format_compliance"] for r in reward_results)

                # Skip if all advantages are zero (no signal)
                if all(a == 0.0 for a in advantages_G):
                    continue

                # Build training datums
                for tokens, logprobs, advantage in zip(
                    sampled_tokens_G, logprobs_G, advantages_G
                ):
                    ob_len = prompt.length - 1
                    model_input = prompt.append(types.EncodedTextChunk(tokens=tokens[:-1]))
                    target_tokens = [0] * ob_len + tokens
                    padded_logprobs = [0.0] * ob_len + logprobs
                    padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)

                    assert (
                        model_input.length
                        == len(target_tokens)
                        == len(padded_logprobs)
                        == len(padded_advantages)
                    )

                    datum = types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                        },
                    )
                    datums_D.append(datum)

            # Training step
            if datums_D:
                fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
                optim_future = training_client.optim_step(adam_params)
                fwd_bwd_future.result()
                optim_result = optim_future.result()
                if optim_result.metrics:
                    metrics.update(optim_result.metrics)

            # Log metrics
            metrics["time/total"] = time.time() - t_start
            metrics["reward/total"] = sum(all_rewards) / len(all_rewards) if all_rewards else 0
            metrics["reward/rgs_mean"] = sum(all_rgs) / len(all_rgs) if all_rgs else 0
            metrics["reward/format_mean"] = sum(all_fmt) / len(all_fmt) if all_fmt else 0
            metrics["training/n_datums"] = len(datums_D)
            ml_logger.log(metrics, step=global_step)

            # Save checkpoint
            if config.save_every > 0 and global_step % config.save_every == 0 and global_step > 0:
                from tinker_cookbook import checkpoint_utils
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{global_step:06d}",
                    log_path=config.log_path,
                    kind="state",
                    loop_state={"batch": global_step},
                )
                logger.info(f"  Saved checkpoint at step {global_step}")

            global_step += 1

    # Save final checkpoint
    from tinker_cookbook import checkpoint_utils
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": global_step},
    )
    ml_logger.close()
    print(f"\nTraining complete! Final checkpoint saved to {config.log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO training for RepoDesign via Tinker")
    parser.add_argument("repo_irs_dir", help="Directory with per-repo training data")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Instruct", help="Base model")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Prompts per batch")
    parser.add_argument("--group-size", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max generation tokens")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N batches")
    parser.add_argument("--log-path", default="output/grpo_training", help="Log/checkpoint directory")
    parser.add_argument("--no-llm-judge", action="store_true", help="Disable LLM-as-judge (faster)")
    parser.add_argument("--no-diagrams", action="store_true", help="Disable diagram images")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = Config()
    config.model_name = args.model
    config.lora_rank = args.lora_rank
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.group_size = args.group_size
    config.max_tokens = args.max_tokens
    config.num_epochs = args.epochs
    config.save_every = args.save_every
    config.log_path = args.log_path
    config.use_llm_judge = not args.no_llm_judge
    config.use_diagrams = not args.no_diagrams

    train(config, args.repo_irs_dir)


if __name__ == "__main__":
    main()
