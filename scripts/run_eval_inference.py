#!/usr/bin/env python3
"""Run inference on eval examples using the trained model via Tinker.

Usage:
    python scripts/run_eval_inference.py data/eval --output output/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import types

from repodesign.training.reward import compute_rewards, parse_diff_files
from repodesign.training.vl_renderer import Qwen3VLRenderer, load_diagram_images
from repodesign.training.data_gen import summarize_repo_ir_for_prompt

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior software architect. Given a codebase analysis and a feature specification, generate a detailed implementation plan as JSON.

Return a JSON object with these fields:
{
  "architecture_decisions": [{"dimension": "...", "recommendation": "...", "rationale": "...", "alternatives_considered": [...], "files_affected": [...]}],
  "tickets": [{"id": "T-001", "title": "...", "description": "...", "files_to_modify": [...], "files_to_create": [...], "estimated_effort": "small|medium|large", "dependencies": [...]}],
  "implementation_summary": "2-3 paragraph explanation of the overall approach, key patterns used, and how the changes integrate with the existing codebase"
}

IMPORTANT:
- files_to_modify and files_to_create must reference REAL file paths from the codebase
- Generate 3-8 architecture decisions and 4-10 actionable tickets
- implementation_summary should explain the overall approach"""


def build_eval_prompt(
    repo_ir_summary: str,
    spec: dict,
    file_manifest: list[str],
    diff_files: dict,
    renderer: Qwen3VLRenderer,
    diagram_images: list[bytes] | None = None,
) -> types.ModelInput:
    """Build prompt for eval — includes candidate file list like training."""
    import random
    rng = random.Random(42)  # deterministic for eval

    manifest_str = "\n".join(file_manifest[:500])

    actual_files = set(diff_files.get("modified", [])) | set(diff_files.get("created", []))
    available_distractors = [f for f in file_manifest if f not in actual_files]
    num_distractors = min(50, len(available_distractors))
    distractors = rng.sample(available_distractors, num_distractors) if available_distractors else []
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

{f"Scale: {spec['scale_tier']}" + chr(10) if spec.get('scale_tier') else ''}## Candidate Files (select from this list for files_to_modify and files_to_create)
{candidate_str}

Generate an implementation plan following the JSON schema. ALL file references must come from the Candidate Files list above."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    return renderer.build_generation_prompt(messages, diagram_images=diagram_images)


def main():
    parser = argparse.ArgumentParser(description="Run eval inference via Tinker")
    parser.add_argument("eval_dir", help="Directory with eval examples")
    parser.add_argument("--output", "-o", default="output/eval_results.json")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Instruct")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--num-samples", type=int, default=4, help="Completions per example")
    parser.add_argument("--checkpoint-dir", default=None, help="Load trained checkpoint from this dir")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load eval examples
    eval_dir = Path(args.eval_dir)
    examples = []
    for repo_dir in sorted(eval_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        required = ["repo_ir.json", "spec.json", "teacher_plan.json", "ground_truth_diff.txt"]
        if not all((repo_dir / f).exists() for f in required):
            print(f"Skipping {repo_dir.name} (missing files)")
            continue

        repo_ir = json.loads((repo_dir / "repo_ir.json").read_text())
        spec = json.loads((repo_dir / "spec.json").read_text())
        teacher_plan = json.loads((repo_dir / "teacher_plan.json").read_text())
        diff_files = parse_diff_files((repo_dir / "ground_truth_diff.txt").read_text())

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

    print(f"Loaded {len(examples)} eval examples")
    if not examples:
        print("ERROR: No eval examples found")
        sys.exit(1)

    # Connect to Tinker
    print(f"Connecting to Tinker ({args.model})...")
    service_client = tinker.ServiceClient()

    if args.checkpoint_dir:
        # Load trained checkpoint
        from tinker_cookbook.checkpoint_utils import get_last_checkpoint
        ckpt = get_last_checkpoint(args.checkpoint_dir)
        print(f"Loading checkpoint: {ckpt['name']} (batch {ckpt['batch']})")
        training_client = service_client.create_training_client_from_state(
            path=ckpt["state_path"],
        )
        tokenizer = training_client.get_tokenizer()
        renderer = Qwen3VLRenderer(tokenizer)
        # Use sampler weights if available, else save current weights
        if ckpt.get("sampler_path"):
            sampling_client = service_client.create_sampling_client(
                model_path=ckpt["sampler_path"],
            )
        else:
            sampling_client = training_client.save_weights_and_get_sampling_client()
    else:
        # Base model (no training)
        training_client = service_client.create_lora_training_client(
            base_model=args.model,
            rank=args.lora_rank,
        )
        tokenizer = training_client.get_tokenizer()
        renderer = Qwen3VLRenderer(tokenizer)
        sampling_client = training_client.save_weights_and_get_sampling_client()
    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    # Run inference
    results = []
    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex['repo_name']}...")
        t_start = time.time()

        try:
            prompt = build_eval_prompt(
                repo_ir_summary=ex["repo_ir_summary"],
                spec=ex["spec"],
                file_manifest=ex["file_manifest"],
                diff_files=ex["diff_files"],
                renderer=renderer,
                diagram_images=ex["diagram_images"],
            )

            sample_result = sampling_client.sample(
                prompt=prompt,
                num_samples=args.num_samples,
                sampling_params=sampling_params,
            ).result()

            completions = []
            for seq in sample_result.sequences:
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                completions.append(parsed_msg.get("content", ""))

            # Score all completions
            reward_results = compute_rewards(
                completions=completions,
                file_manifests=[ex["file_manifest"]] * len(completions),
                specs=[ex["spec"]] * len(completions),
                teacher_plans=[ex["teacher_plan"]] * len(completions),
                diff_files=[ex["diff_files"]] * len(completions),
            )

            # Pick best completion
            best_idx = max(range(len(reward_results)), key=lambda j: reward_results[j]["total"])
            best = reward_results[best_idx]

            result = {
                "repo_name": ex["repo_name"],
                "spec": ex["spec"],
                "ground_truth": {
                    "modified": ex["diff_files"].get("modified", []),
                    "created": ex["diff_files"].get("created", []),
                },
                "best_completion": completions[best_idx],
                "best_rewards": best,
                "all_rewards": reward_results,
                "time_seconds": time.time() - t_start,
            }
            results.append(result)

            print(f"  Reward: {best['total']:.3f} | fmt={best['format_compliance']:.1f} "
                  f"exist_f1={best['existing_file_accuracy']:.3f} "
                  f"sem={best['semantic_similarity']:.3f} "
                  f"time={result['time_seconds']:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "repo_name": ex["repo_name"],
                "error": str(e),
            })

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    scored = [r for r in results if "best_rewards" in r]
    if scored:
        avg_reward = sum(r["best_rewards"]["total"] for r in scored) / len(scored)
        avg_exist = sum(r["best_rewards"]["existing_file_accuracy"] for r in scored) / len(scored)
        avg_fmt = sum(1 for r in scored if r["best_rewards"]["format_compliance"] > 0) / len(scored)
        print(f"\n{'='*60}")
        print(f"EVAL SUMMARY ({len(scored)} examples)")
        print(f"  Avg reward:        {avg_reward:.3f}")
        print(f"  Avg exist_f1:      {avg_exist:.3f}")
        print(f"  Format pass rate:  {avg_fmt:.0%}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
