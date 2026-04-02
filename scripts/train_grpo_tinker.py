#!/usr/bin/env python3
"""GRPO training loop for RepoDesign using Tinker SDK.

Adapted from:
  - tinker-cookbook/recipes/rl_loop.py (Tinker GRPO pattern)
  - deepmind_tunix/general_reasoning (curriculum + reward design)

Usage:
    python scripts/train_grpo_tinker.py data/commit_pairs_production --group-size 8 -v
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

from repodesign.training.reward import compute_rewards, _normalize_path, parse_diff_files, init_tfidf_corpus
from repodesign.training.vl_renderer import Qwen3VLRenderer, load_diagram_images
from repodesign.training.data_gen import summarize_repo_ir_for_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    lora_rank: int = 64
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6   # cosine decay target
    warmup_ratio: float = 0.05        # fraction of steps for warmup
    batch_size: int = 4               # prompts per batch
    group_size: int = 4               # completions per prompt (G in GRPO)
    max_tokens: int = 8192            # max generation length
    num_epochs: int = 1
    save_every: int = 5               # save checkpoint every N batches
    log_path: str = "output/grpo_training"
    use_diagrams: bool = True         # feed diagram images to VLM
    max_samples: int | None = None    # limit training examples (None = use all)
    eval_every: int = 10              # run eval every N steps
    max_grad_norm: float = 1.0        # gradient clipping
    advantage_clip: float = 2.0       # advantage clipping
    kl_beta: float = 0.1              # KL penalty coefficient
    eval_fraction: float = 0.2        # fraction of data for eval (~2 per repo)
    num_steps: int | None = None      # explicit step count (overrides epochs)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, config: Config) -> float:
    """Cosine learning rate schedule with linear warmup."""
    warmup_steps = int(total_steps * config.warmup_ratio)
    if step < warmup_steps:
        # Linear warmup
        return config.learning_rate * (step + 1) / warmup_steps
    # Cosine decay
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine_decay


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior software architect. Given a codebase analysis and a feature specification, generate a detailed implementation plan as JSON.

Return a JSON object with these fields:
{
  "architecture_decisions": [{"dimension": "...", "recommendation": "...", "rationale": "...", "alternatives_considered": [...], "files_affected": [...]}],
  "tickets": [{"id": "T-001", "title": "...", "description": "...", "files_to_modify": [...], "files_to_create": [...], "estimated_effort": "small|medium|large", "dependencies": [...]}],
  "implementation_summary": "2-3 paragraph explanation of the overall approach, key patterns used, and how the changes integrate with the existing codebase"
}

IMPORTANT:
- files_to_modify and files_to_create must reference REAL file paths from the codebase
- files_affected in architecture_decisions should also use real paths from the diff
- Generate 3-8 architecture decisions and 4-10 actionable tickets
- implementation_summary should explain the overall approach, key patterns used, and how changes integrate with existing code"""


def build_prompt(
    repo_ir_summary: str,
    spec: dict,
    file_manifest: list[str],
    diff_files: dict,
    renderer: Qwen3VLRenderer,
    diagram_images: list[bytes] | None = None,
) -> types.ModelInput:
    """Build a Tinker ModelInput prompt from RepoIR + Spec + optional diagrams.

    Constructs a candidate file list: ground truth modified/created files + ~50
    distractors from the manifest. The model selects relevant files from this list.
    """
    import random
    manifest_str = "\n".join(file_manifest[:500])

    # Build candidate list: actual modified/created files + 50 distractors
    actual_files = set(diff_files.get("modified", [])) | set(diff_files.get("created", []))
    available_distractors = [f for f in file_manifest if f not in actual_files]
    num_distractors = min(50, len(available_distractors))
    distractors = random.sample(available_distractors, num_distractors) if available_distractors else []

    candidate_files = sorted(list(actual_files) + distractors)
    candidate_str = "\n".join(candidate_files)

    user_text = f"""## Codebase Analysis
{repo_ir_summary}

## File Manifest (for context - all files in repo)
{manifest_str}

## Feature Specification
Project: {spec.get('project_name', 'Unknown')}
Feature: {spec.get('feature_name', 'Unknown')}
Description: {spec.get('description', '')}

Requirements:
{chr(10).join(f'- {r}' for r in spec.get('functional_requirements', []))}

{f"Scale: {spec['scale_tier']}" + chr(10) if spec.get('scale_tier') else ''}## CRITICAL INSTRUCTION - File Selection Constraint
You MUST ONLY select files from the "Candidate Files" list below for files_to_modify and files_to_create.
Do NOT invent new file paths. Do NOT reference files from the full manifest above.
ONLY use files from this candidate list:

## Candidate Files (ONLY modify/create files from this list)
{candidate_str}

Generate an implementation plan following the JSON schema. Remember: ALL file references must come from the Candidate Files list above."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    return renderer.build_generation_prompt(messages, diagram_images=diagram_images)


# ---------------------------------------------------------------------------
# Data loading with train/eval split
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
        diff_path = repo_dir / "ground_truth_diff.txt"

        if not all(p.exists() for p in [repo_ir_path, spec_path, plan_path, diff_path]):
            continue

        try:
            with open(repo_ir_path, encoding="utf-8") as f:
                repo_ir = json.load(f)
            with open(spec_path, encoding="utf-8") as f:
                spec = json.load(f)
            with open(plan_path, encoding="utf-8") as f:
                teacher_plan = json.load(f)
            with open(diff_path, encoding="utf-8") as f:
                diff_text = f.read()
            diff_files = parse_diff_files(diff_text)
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
            "diff_files": diff_files,
            "diagram_images": diagram_images,
        })

    return examples


def split_train_eval(examples: list[dict], eval_fraction: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split examples into train/eval, stratified by repo (~2 per repo for eval)."""
    rng = random.Random(seed)

    # Group by repo prefix (e.g., "astro" from "astro_pr15779")
    repo_groups: dict[str, list[dict]] = {}
    for ex in examples:
        repo = ex["repo_name"].rsplit("_pr", 1)[0] if "_pr" in ex["repo_name"] else ex["repo_name"]
        repo_groups.setdefault(repo, []).append(ex)

    train, eval_set = [], []
    for repo, group in repo_groups.items():
        rng.shuffle(group)
        n_eval = max(1, int(len(group) * eval_fraction))
        eval_set.extend(group[:n_eval])
        train.extend(group[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_set)
    return train, eval_set


# ---------------------------------------------------------------------------
# Curriculum sampler
# ---------------------------------------------------------------------------

class StratifiedSampler:
    """Curriculum sampler: stratified by difficulty + EMA-based reweighting.

    Difficulty proxy: number of files in the ground-truth diff (modified + created).
    More files = harder (model must identify more specific paths from candidate list).

    Curriculum schedule:
      - Early steps: 60% easy / 30% medium / 10% hard
      - By step 50+:  30% easy / 30% medium / 40% hard

    Within each bin, examples in the "learning zone" (not too easy, not stuck)
    are sampled more often via EMA-based weights.
    """

    EMA_ALPHA = 0.3   # recency weight for per-example reward EMA
    MASTERED_THRESHOLD = 4.5   # EMA above this → mastered, reduce exposure
    STUCK_THRESHOLD    = 0.5   # EMA below this → stuck, give brief reprieve

    def __init__(self, examples: list[dict], seed: int = 42):
        self.rng = random.Random(seed)
        self.ema_rewards: dict[str, float | None] = {}

        # Annotate each example with a difficulty score
        annotated = []
        for ex in examples:
            n_files = (len(ex["diff_files"].get("modified", []))
                       + len(ex["diff_files"].get("created", [])))
            annotated.append((n_files, ex))
            self.ema_rewards[ex["repo_name"]] = None

        # Sort and split into tertiles
        annotated.sort(key=lambda x: x[0])
        n = len(annotated)
        t1, t2 = n // 3, 2 * n // 3
        self.easy   = [ex for _, ex in annotated[:t1]]
        self.medium = [ex for _, ex in annotated[t1:t2]]
        self.hard   = [ex for _, ex in annotated[t2:]]

        easy_max   = annotated[t1 - 1][0] if t1 > 0 else 0
        med_max    = annotated[t2 - 1][0] if t2 > t1 else 0
        hard_min   = annotated[t2][0]     if t2 < n  else 0

        logger.info(
            f"StratifiedSampler: {len(self.easy)} easy (≤{easy_max} files), "
            f"{len(self.medium)} medium ({easy_max+1}–{med_max}), "
            f"{len(self.hard)} hard (≥{hard_min})"
        )

    # ------------------------------------------------------------------
    def update(self, repo_name: str, reward: float) -> None:
        """Update per-example EMA after observing a reward."""
        prev = self.ema_rewards.get(repo_name)
        if prev is None:
            self.ema_rewards[repo_name] = reward
        else:
            self.ema_rewards[repo_name] = self.EMA_ALPHA * reward + (1 - self.EMA_ALPHA) * prev

    # ------------------------------------------------------------------
    def sample_batch(self, batch_size: int, step: int = 0) -> list[dict]:
        """Return a batch with a difficulty distribution that shifts over time."""
        # Progressive curriculum: easy fraction decreases, hard increases
        progress    = min(step / 50.0, 1.0)   # saturates at step 50
        easy_frac   = 0.60 - 0.30 * progress  # 0.60 → 0.30
        medium_frac = 0.30                     # constant
        hard_frac   = 0.10 + 0.30 * progress  # 0.10 → 0.40

        total = easy_frac + medium_frac + hard_frac
        n_easy   = round(batch_size * easy_frac   / total)
        n_medium = round(batch_size * medium_frac / total)
        n_hard   = batch_size - n_easy - n_medium

        batch: list[dict] = []
        seen: set[str] = set()

        for pool, n in [(self.easy, n_easy), (self.medium, n_medium), (self.hard, n_hard)]:
            if not pool or n <= 0:
                continue
            weights = self._weights(pool)
            candidates = self.rng.choices(pool, weights=weights, k=min(n * 3, len(pool)))
            for ex in candidates:
                if ex["repo_name"] not in seen and len(batch) < batch_size:
                    batch.append(ex)
                    seen.add(ex["repo_name"])
                if sum(1 for b in batch if b in pool) >= n:
                    break

        # Fill any remaining slots from the full pool
        all_examples = self.easy + self.medium + self.hard
        while len(batch) < batch_size:
            ex = self.rng.choice(all_examples)
            if ex["repo_name"] not in seen:
                batch.append(ex)
                seen.add(ex["repo_name"])

        self.rng.shuffle(batch)
        return batch[:batch_size]

    # ------------------------------------------------------------------
    def _weights(self, pool: list[dict]) -> list[float]:
        """Sampling weights within a bin: prioritise the learning zone."""
        weights = []
        for ex in pool:
            ema = self.ema_rewards.get(ex["repo_name"])
            if ema is None:
                w = 1.0                        # unseen — neutral
            elif ema >= self.MASTERED_THRESHOLD:
                w = 0.5                        # mastered — reduce exposure
            elif ema <= self.STUCK_THRESHOLD:
                w = 0.5                        # stuck — brief reprieve
            else:
                w = 1.5                        # learning zone — prioritise
            weights.append(w)
        return weights

    # ------------------------------------------------------------------
    def difficulty_stats(self, step: int) -> dict:
        """Log-friendly snapshot of current curriculum state."""
        progress  = min(step / 50.0, 1.0)
        seen_emas = [v for v in self.ema_rewards.values() if v is not None]
        return {
            "curriculum/step":         step,
            "curriculum/progress":     progress,
            "curriculum/easy_frac":    round(0.60 - 0.30 * progress, 2),
            "curriculum/hard_frac":    round(0.10 + 0.30 * progress, 2),
            "curriculum/n_seen":       len(seen_emas),
            "curriculum/mean_ema":     sum(seen_emas) / len(seen_emas) if seen_emas else 0.0,
        }


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------

class MetricsLogger:
    def __init__(self, log_path: str, config: "Config"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.metrics_file = open(self.log_path / "metrics.jsonl", "a")

        wandb.init(
            project="repodesign-grpo",
            config={
                "model_name": config.model_name,
                "lora_rank": config.lora_rank,
                "learning_rate": config.learning_rate,
                "min_learning_rate": config.min_learning_rate,
                "warmup_ratio": config.warmup_ratio,
                "batch_size": config.batch_size,
                "group_size": config.group_size,
                "max_tokens": config.max_tokens,
                "num_epochs": config.num_epochs,
                "use_diagrams": config.use_diagrams,
                "kl_beta": config.kl_beta,
                "max_grad_norm": config.max_grad_norm,
                "advantage_clip": config.advantage_clip,
                "eval_fraction": config.eval_fraction,
            },
        )

    def log(self, metrics: dict, step: int):
        metrics["step"] = step
        self.metrics_file.write(json.dumps(metrics) + "\n")
        self.metrics_file.flush()
        wandb.log(metrics, step=step)

        reward_total = metrics.get("reward/total", 0)
        existing_acc = metrics.get("reward/existing_file_accuracy", 0)
        created_acc = metrics.get("reward/created_file_accuracy", 0)
        fmt_rate = metrics.get("reward/format_compliance_rate", 0)
        sem_sim = metrics.get("reward/semantic_similarity", 0)
        t = metrics.get("time/total", 0)
        lr = metrics.get("optim/lr", 0)
        prefix = "EVAL" if "eval/" in str(metrics.keys()) else "Step"
        print(f"  {prefix} {step}: reward={reward_total:.3f} exist_f1={existing_acc:.3f} "
              f"created_f1={created_acc:.3f} fmt_rate={fmt_rate:.2f} sem={sem_sim:.3f} "
              f"lr={lr:.2e} time={t:.1f}s")

    def close(self):
        self.metrics_file.close()
        wandb.finish()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    eval_examples: list[dict],
    sampling_client,
    renderer: Qwen3VLRenderer,
    config: Config,
    step: int,
    ml_logger: MetricsLogger,
):
    """Run evaluation on held-out examples (no gradient updates)."""
    if not eval_examples:
        return

    all_rewards = []
    all_metrics: dict[str, list[float]] = {
        "format_compliance": [], "existing_file_accuracy": [],
        "created_file_accuracy": [], "semantic_similarity": [],
        "structural_quality": [],
    }

    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    # Eval on a subset to keep it fast
    eval_subset = eval_examples[:min(8, len(eval_examples))]

    for ex in eval_subset:
        try:
            prompt = build_prompt(
                repo_ir_summary=ex["repo_ir_summary"],
                spec=ex["spec"],
                file_manifest=ex["file_manifest"],
                diff_files=ex["diff_files"],
                renderer=renderer,
                diagram_images=ex["diagram_images"] if config.use_diagrams else None,
            )

            # Single greedy completion for eval
            sample_result = sampling_client.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params,
            ).result()

            parsed_msg, _ = renderer.parse_response(sample_result.sequences[0].tokens)
            completion = parsed_msg.get("content", "")

            reward_results = compute_rewards(
                completions=[completion],
                file_manifests=[ex["file_manifest"]],
                specs=[ex["spec"]],
                teacher_plans=[ex["teacher_plan"]],
                diff_files=[ex["diff_files"]],
            )

            r = reward_results[0]
            all_rewards.append(r["total"])
            for key in all_metrics:
                all_metrics[key].append(r.get(key, 0.0))

        except Exception as e:
            logger.warning(f"Eval failed for {ex.get('repo_name', '?')}: {e}")

    if all_rewards:
        metrics = {
            "eval/reward_total": sum(all_rewards) / len(all_rewards),
            "eval/format_compliance_rate": sum(1 for v in all_metrics["format_compliance"] if v > 0) / len(all_metrics["format_compliance"]),
        }
        for key, vals in all_metrics.items():
            metrics[f"eval/{key}"] = sum(vals) / len(vals) if vals else 0
        ml_logger.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: Config, repo_irs_dir: str):
    """Run GRPO training loop."""
    logger.info(f"Loading training examples from {repo_irs_dir}...")
    all_examples = load_training_examples(repo_irs_dir, use_diagrams=config.use_diagrams)
    if not all_examples:
        print("ERROR: No complete training examples found.")
        sys.exit(1)

    # Train/eval split
    train_examples, eval_examples = split_train_eval(all_examples, eval_fraction=config.eval_fraction)
    print(f"Data: {len(train_examples)} train, {len(eval_examples)} eval (from {len(all_examples)} total)")

    # Apply max_samples
    if config.max_samples is not None:
        train_examples = train_examples[:config.max_samples]
        print(f"  Truncated to {len(train_examples)} train examples")

    # Curriculum sampler
    sampler = StratifiedSampler(train_examples, seed=42)

    # Pre-fit TF-IDF on full teacher corpus so IDF weights are meaningful
    init_tfidf_corpus([ex["teacher_plan"] for ex in examples])

    # Setup Tinker
    logger.info(f"Connecting to Tinker with model {config.model_name}...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    tokenizer = training_client.get_tokenizer()
    renderer = Qwen3VLRenderer(tokenizer)
    logger.info(f"Using renderer: Qwen3VLRenderer (multimodal={config.use_diagrams})")

    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    # Total steps: explicit num_steps takes priority, else derive from epochs
    n_batches_per_epoch = max(1, len(train_examples) // config.batch_size)
    total_steps = config.num_steps if config.num_steps is not None else n_batches_per_epoch * config.num_epochs

    # Metrics
    ml_logger = MetricsLogger(config.log_path, config)

    print(f"\nTraining config:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Group size: {config.group_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  LR: {config.learning_rate} -> {config.min_learning_rate} (cosine)")
    print(f"  Warmup: {config.warmup_ratio:.0%} ({int(total_steps * config.warmup_ratio)} steps)")
    print(f"  KL beta: {config.kl_beta}")
    print(f"  Grad clip: {config.max_grad_norm}")
    print(f"  Advantage clip: {config.advantage_clip}")
    print(f"  Curriculum: stratified + EMA reweighting")
    print(f"  Eval every: {config.eval_every} steps ({len(eval_examples)} eval examples)")
    print(f"  Diagrams: {config.use_diagrams}")
    print()

    global_step = 0

    print(f"\n{'='*60}")
    print(f"Training ({total_steps} steps, curriculum enabled)")
    print(f"{'='*60}")

    for step_idx in range(total_steps):
            t_start = time.time()

            # Compute current learning rate
            current_lr = get_lr(global_step, total_steps, config)
            adam_params = types.AdamParams(
                learning_rate=current_lr,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )

            metrics: dict[str, float] = {
                "progress/global_step": global_step,
                "optim/lr": current_lr,
            }
            metrics.update(sampler.difficulty_stats(global_step))

            # Get batch from curriculum sampler
            batch = sampler.sample_batch(config.batch_size, step=global_step)

            print(f"\n[Step {step_idx + 1}/{total_steps}] Processing {len(batch)} examples "
                  f"(lr={current_lr:.2e})...")

            # Create sampling client from current weights
            try:
                sampling_client = training_client.save_weights_and_get_sampling_client()
            except Exception as e:
                logger.error(f"Failed to create sampling client: {e}")
                global_step += 1
                continue

            datums_D: list[types.Datum] = []
            all_rewards: list[float] = []
            all_existing_acc: list[float] = []
            all_created_acc: list[float] = []
            all_fmt_exact: list[float] = []
            all_fmt_partial: list[float] = []
            all_sem_sim: list[float] = []
            all_struct: list[float] = []
            all_nonempty: list[float] = []
            n_format_pass = 0
            n_total_completions = 0

            for ex in batch:
                try:
                    prompt = build_prompt(
                        repo_ir_summary=ex["repo_ir_summary"],
                        spec=ex["spec"],
                        file_manifest=ex["file_manifest"],
                        diff_files=ex["diff_files"],
                        renderer=renderer,
                        diagram_images=ex["diagram_images"] if config.use_diagrams else None,
                    )

                    # Generate G completions
                    sample_result = sampling_client.sample(
                        prompt=prompt,
                        num_samples=config.group_size,
                        sampling_params=sampling_params,
                    ).result()

                    completions: list[str] = []
                    sampled_tokens_G: list[list[int]] = []
                    logprobs_G: list[list[float]] = []

                    for seq in sample_result.sequences:
                        sampled_tokens_G.append(seq.tokens)
                        logprobs_G.append(seq.logprobs)
                        parsed_msg, _ = renderer.parse_response(seq.tokens)
                        content = parsed_msg.get("content", "")
                        completions.append(content)

                    n_total_completions += len(completions)

                    # Compute rewards
                    reward_results = compute_rewards(
                        completions=completions,
                        file_manifests=[ex["file_manifest"]] * len(completions),
                        specs=[ex["spec"]] * len(completions),
                        teacher_plans=[ex["teacher_plan"]] * len(completions),
                        diff_files=[ex["diff_files"]] * len(completions),
                    )

                    rewards_G = [r["total"] for r in reward_results]
                    mean_reward = sum(rewards_G) / len(rewards_G)
                    advantages_G = [r - mean_reward for r in rewards_G]

                    # Clip advantages
                    advantages_G = [
                        max(-config.advantage_clip, min(config.advantage_clip, a))
                        for a in advantages_G
                    ]

                    # Update curriculum EMA for this example
                    sampler.update(ex["repo_name"], mean_reward)

                    # Track metrics
                    all_rewards.append(mean_reward)
                    for r in reward_results:
                        all_existing_acc.append(r["existing_file_accuracy"])
                        all_created_acc.append(r["created_file_accuracy"])
                        all_fmt_exact.append(r["format_compliance"])
                        all_fmt_partial.append(r["format_partial"])
                        all_sem_sim.append(r["semantic_similarity"])
                        all_struct.append(r["structural_quality"])
                        all_nonempty.append(r.get("nonempty_bonus", 0.0))
                        if r["format_compliance"] > 0:
                            n_format_pass += 1

                    # Log best completion for inspection
                    best_idx = max(range(len(rewards_G)), key=lambda i: rewards_G[i])
                    completion_log = {
                        "step": global_step,
                        "repo": ex.get("repo_name", "unknown"),
                        "completion": completions[best_idx],
                        "rewards": reward_results[best_idx],
                    }
                    completion_log_path = Path(config.log_path) / "completions.jsonl"
                    completion_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(completion_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(completion_log) + "\n")

                    # Skip if all advantages are zero
                    if all(a == 0.0 for a in advantages_G):
                        continue

                    # Build training datums
                    for tokens, logprobs, advantage in zip(
                        sampled_tokens_G, logprobs_G, advantages_G
                    ):
                        if len(tokens) < 2:
                            continue
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

                except Exception as e:
                    logger.error(f"Error processing {ex.get('repo_name', '?')}: {e}")
                    continue

            # Training step with crash handling
            if datums_D:
                try:
                    fwd_bwd_future = training_client.forward_backward(
                        datums_D,
                        loss_fn="importance_sampling",
                    )
                    optim_future = training_client.optim_step(adam_params)
                    fwd_bwd_future.result()
                    optim_result = optim_future.result()
                    if optim_result.metrics:
                        metrics.update(optim_result.metrics)
                except Exception as e:
                    logger.error(f"Training step failed at step {global_step}: {e}")
                    global_step += 1
                    continue

            # Log metrics
            metrics["time/total"] = time.time() - t_start
            metrics["reward/total"] = sum(all_rewards) / len(all_rewards) if all_rewards else 0
            metrics["train/reward"] = metrics["reward/total"]

            metrics["reward/existing_file_accuracy"] = sum(all_existing_acc) / len(all_existing_acc) if all_existing_acc else 0
            metrics["reward/created_file_accuracy"] = sum(all_created_acc) / len(all_created_acc) if all_created_acc else 0
            metrics["reward/format_compliance"] = sum(all_fmt_exact) / len(all_fmt_exact) if all_fmt_exact else 0
            metrics["reward/format_partial"] = sum(all_fmt_partial) / len(all_fmt_partial) if all_fmt_partial else 0
            metrics["reward/semantic_similarity"] = sum(all_sem_sim) / len(all_sem_sim) if all_sem_sim else 0
            metrics["reward/structural_quality"] = sum(all_struct) / len(all_struct) if all_struct else 0
            metrics["reward/nonempty_bonus"] = sum(all_nonempty) / len(all_nonempty) if all_nonempty else 0

            # Format compliance rate (key health metric)
            metrics["reward/format_compliance_rate"] = n_format_pass / n_total_completions if n_total_completions > 0 else 0

            metrics["training/n_datums"] = len(datums_D)
            metrics["training/n_completions"] = n_total_completions

            if "loss" in metrics:
                metrics["train/loss"] = metrics["loss"]

            ml_logger.log(metrics, step=global_step)

            # Eval
            if config.eval_every > 0 and global_step % config.eval_every == 0:
                print(f"  Running eval...")
                try:
                    eval_sampling_client = training_client.save_weights_and_get_sampling_client()
                    run_eval(eval_examples, eval_sampling_client, renderer, config, global_step, ml_logger)
                except Exception as e:
                    logger.warning(f"Eval failed at step {global_step}: {e}")

            # Checkpoint
            if config.save_every > 0 and global_step % config.save_every == 0 and global_step > 0:
                try:
                    from tinker_cookbook import checkpoint_utils
                    checkpoint_utils.save_checkpoint(
                        training_client=training_client,
                        name=f"{global_step:06d}",
                        log_path=config.log_path,
                        kind="state",
                        loop_state={"batch": global_step},
                    )
                    logger.info(f"  Saved checkpoint at step {global_step}")
                except Exception as e:
                    logger.warning(f"  Checkpoint save failed at step {global_step}: {e}")

            global_step += 1

    # Final checkpoint
    try:
        from tinker_cookbook import checkpoint_utils
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"batch": global_step},
        )
    except Exception as e:
        logger.warning(f"Final checkpoint save failed: {e}")
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
    parser.add_argument("--group-size", type=int, default=4, help="Completions per prompt (G)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max generation tokens")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N batches")
    parser.add_argument("--eval-every", type=int, default=10, help="Eval every N steps")
    parser.add_argument("--log-path", default="output/grpo_training", help="Log/checkpoint directory")
    parser.add_argument("--no-diagrams", action="store_true", help="Disable diagram images")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training examples")
    parser.add_argument("--kl-beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--num-steps", type=int, default=None, help="Explicit step count (overrides --epochs)")
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
    config.eval_every = args.eval_every
    config.log_path = args.log_path
    config.use_diagrams = not args.no_diagrams
    config.max_samples = args.max_samples
    config.kl_beta = args.kl_beta
    config.max_grad_norm = args.max_grad_norm
    config.num_steps = args.num_steps

    train(config, args.repo_irs_dir)


if __name__ == "__main__":
    main()
