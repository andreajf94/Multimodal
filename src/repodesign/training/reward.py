"""Reward functions for GRPO training of the RepoDesign model.

Adapted from deepmind_tunix/general_reasoning reward patterns,
specialized for implementation plan evaluation.

Components:
  1. format_compliance  (max 0.5)  — valid JSON with required fields
  2. format_partial     (max 0.25) — partial credit for near-valid structure
  3. rgs_score          (max 3.0)  — Repo Grounding Score against file manifest
  4. llm_judge          (max 3.0)  — DeepSeek Chat evaluates plan quality vs teacher
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# Required top-level keys in a valid plan
REQUIRED_PLAN_KEYS = {"architecture_decisions", "tickets", "technology_choices"}
REQUIRED_TICKET_KEYS = {"id", "title", "description"}
REQUIRED_DECISION_KEYS = {"dimension", "recommendation", "rationale"}


# ---------------------------------------------------------------------------
# 1. Format compliance (exact match)
# ---------------------------------------------------------------------------

def format_compliance(completions: list[str]) -> list[float]:
    """Score 0.5 if the completion is valid JSON with all required plan fields."""
    scores = []
    for text in completions:
        try:
            plan = _parse_plan_json(text)
            if plan is None:
                scores.append(0.0)
                continue
            # Check required keys
            if not REQUIRED_PLAN_KEYS.issubset(plan.keys()):
                scores.append(0.0)
                continue
            # Check that lists are non-empty
            if not plan.get("architecture_decisions") or not plan.get("tickets"):
                scores.append(0.0)
                continue
            # Check ticket structure
            first_ticket = plan["tickets"][0]
            if not REQUIRED_TICKET_KEYS.issubset(first_ticket.keys()):
                scores.append(0.0)
                continue
            scores.append(0.5)
        except Exception:
            scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# 2. Format partial (approximate match)
# ---------------------------------------------------------------------------

def format_partial(completions: list[str]) -> list[float]:
    """Partial credit for near-valid JSON structure."""
    scores = []
    for text in completions:
        score = 0.0
        # Credit for having JSON-like structure
        if "{" in text and "}" in text:
            score += 0.05
        # Credit for key field names being present
        for key in ["architecture_decisions", "tickets", "technology_choices",
                     "files_to_modify", "files_to_create", "recommendation"]:
            if key in text:
                score += 0.025
        # Cap at 0.25
        scores.append(min(score, 0.25))
    return scores


# ---------------------------------------------------------------------------
# 3. RGS score (Repo Grounding Score against file manifest)
# ---------------------------------------------------------------------------

def rgs_score(completions: list[str], file_manifests: list[list[str]]) -> list[float]:
    """Score based on fraction of referenced file paths that exist in manifest.

    Max score: 3.0 (scaled from 0-1 RGS ratio).
    """
    scores = []
    for text, manifest in zip(completions, file_manifests):
        plan = _parse_plan_json(text)
        if plan is None:
            scores.append(0.0)
            continue

        manifest_set = set(manifest)
        referenced = _extract_file_paths(plan)

        if not referenced:
            scores.append(0.0)
            continue

        valid = sum(1 for p in referenced if _normalize_path(p) in manifest_set)
        ratio = valid / len(referenced)
        scores.append(ratio * 3.0)
    return scores


# ---------------------------------------------------------------------------
# 4. LLM-as-judge (DeepSeek Chat)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a code review judge. Evaluate an implementation plan for a software feature.

Score the plan from 0 to 10 on these criteria:
1. **Coherence** (0-3): Are decisions logically connected? Do tickets follow from decisions?
2. **Specificity** (0-3): Are file paths specific? Are descriptions actionable?
3. **Completeness** (0-2): Does the plan cover the feature requirements?
4. **Appropriateness** (0-2): Are technology choices suitable for the project's scale?

Also compare to the reference plan if provided.

Return ONLY a JSON object: {"score": <0-10>, "reasoning": "<brief explanation>"}"""


def llm_judge(
    completions: list[str],
    specs: list[dict],
    teacher_plans: list[dict],
) -> list[float]:
    """Use DeepSeek Chat as judge. Returns scores in [0, 3.0]."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY not set, returning 0 for LLM judge scores")
        return [0.0] * len(completions)

    import openai
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    scores = []
    for text, spec, teacher in zip(completions, specs, teacher_plans):
        try:
            user_prompt = f"""## Feature Specification
{json.dumps(spec, indent=2)}

## Generated Plan (to evaluate)
{text[:4000]}

## Reference Plan (from expert)
{json.dumps(teacher, indent=2)[:4000]}

Score the generated plan."""

            response = client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=512,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            result_text = response.choices[0].message.content or ""

            # Parse score
            try:
                result = json.loads(result_text)
                raw_score = float(result.get("score", 0))
            except (json.JSONDecodeError, ValueError):
                # Try regex fallback
                match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', result_text)
                raw_score = float(match.group(1)) if match else 0.0

            # Scale from 0-10 to 0-3
            scores.append(min(raw_score / 10.0 * 3.0, 3.0))

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            scores.append(0.0)

    return scores


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

def compute_rewards(
    completions: list[str],
    file_manifests: list[list[str]],
    specs: list[dict],
    teacher_plans: list[dict],
    use_llm_judge: bool = True,
) -> list[dict]:
    """Compute all reward components for a batch of completions.

    Returns list of dicts with per-component scores and total.
    """
    fmt_exact = format_compliance(completions)
    fmt_partial = format_partial(completions)
    rgs = rgs_score(completions, file_manifests)

    if use_llm_judge:
        judge = llm_judge(completions, specs, teacher_plans)
    else:
        judge = [0.0] * len(completions)

    results = []
    for i in range(len(completions)):
        total = fmt_exact[i] + fmt_partial[i] + rgs[i] + judge[i]
        results.append({
            "format_compliance": fmt_exact[i],
            "format_partial": fmt_partial[i],
            "rgs_score": rgs[i],
            "llm_judge": judge[i],
            "total": total,
        })
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_plan_json(text: str) -> dict | None:
    """Try to parse JSON from model output (handles markdown blocks)."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _extract_file_paths(plan: dict) -> list[str]:
    """Extract all file paths referenced in a plan dict."""
    paths: set[str] = set()
    for decision in plan.get("architecture_decisions", []):
        paths.update(decision.get("files_affected", []))
    for ticket in plan.get("tickets", []):
        paths.update(ticket.get("files_to_modify", []))
        paths.update(ticket.get("files_to_create", []))
    return list(paths)


def _normalize_path(path: str) -> str:
    """Normalize a file path for comparison."""
    p = path.strip().replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("/"):
        p = p[1:]
    return p
