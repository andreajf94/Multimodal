"""Training data generation: create Specs and teacher plans from RepoIRs using DeepSeek.

Pipeline:
  1. Load RepoIR JSON
  2. Generate a synthetic Spec (feature request) via DeepSeek Reasoner
  3. Generate a teacher reference ImplementationPlan via DeepSeek Reasoner
  4. Save as a training example (prompt + reference) for GRPO
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeepSeek client helper
# ---------------------------------------------------------------------------

def _get_deepseek_client(model: str = "deepseek-reasoner"):
    """Create an OpenAI-compatible client pointing at DeepSeek."""
    import openai

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com"), model


def _call_deepseek(system_prompt: str, user_prompt: str, model: str = "deepseek-reasoner", max_tokens: int = 4096) -> str:
    """Call DeepSeek and return the text response."""
    client, model_name = _get_deepseek_client(model)
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response (handles markdown code blocks)."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not extract JSON from response:\n{text[:500]}")


# ---------------------------------------------------------------------------
# Spec generation
# ---------------------------------------------------------------------------

SPEC_SYSTEM_PROMPT = """You are a product manager. Given a structured analysis of an existing codebase (its architecture, dependencies, API routes, data models, infrastructure), generate a REALISTIC feature request that a team working on this project might actually receive.

Return a JSON object with exactly these fields:
{
  "project_name": "string - name of the project",
  "feature_name": "string - short name for the requested feature",
  "description": "string - 2-3 sentence description of what to build",
  "functional_requirements": ["list of 3-6 specific, actionable requirements"],
  "scale_tier": "hobby" | "startup" | "growth" | "enterprise"
}

Guidelines:
- The feature should be REALISTIC for this codebase — something that builds on existing infrastructure
- Reference actual technologies and patterns you see in the codebase
- Requirements should be specific enough to generate file paths
- Scale tier should match the existing project's maturity level
- Return ONLY the JSON object, no markdown formatting or explanation"""


def generate_synthetic_spec(repo_ir_summary: str, repo_name: str) -> dict:
    """Generate a synthetic feature Spec from a RepoIR summary.

    Args:
        repo_ir_summary: Condensed string representation of the RepoIR.
        repo_name: Name of the repository.

    Returns:
        Dict with spec fields.
    """
    user_prompt = f"Generate a realistic feature request for this project:\n\n{repo_ir_summary}"
    raw = _call_deepseek(SPEC_SYSTEM_PROMPT, user_prompt, model="deepseek-reasoner")
    spec = _extract_json(raw)
    spec.setdefault("project_name", repo_name)
    return spec


# ---------------------------------------------------------------------------
# Teacher plan generation
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = """You are a senior software architect. Given a codebase analysis (RepoIR) and a feature specification, generate a detailed implementation plan.

The plan MUST reference REAL file paths from the codebase. You will be evaluated on whether the file paths you reference actually exist.

Return a JSON object with exactly these fields:
{
  "architecture_decisions": [
    {
      "dimension": "string (e.g., api_design, data_modeling, caching, security, deployment)",
      "recommendation": "string - what to do",
      "rationale": "string - why",
      "alternatives_considered": ["list of alternatives"],
      "files_affected": ["list of REAL file paths from the codebase"]
    }
  ],
  "tickets": [
    {
      "id": "T-001",
      "title": "string - ticket title",
      "description": "string - what to implement",
      "files_to_modify": ["list of EXISTING file paths to change"],
      "files_to_create": ["list of NEW file paths to create"],
      "estimated_effort": "small" | "medium" | "large",
      "dependencies": ["list of ticket IDs this depends on"]
    }
  ],
  "technology_choices": [
    {
      "category": "string (e.g., database, framework, library)",
      "choice": "string - specific technology",
      "rationale": "string - why this choice"
    }
  ]
}

Guidelines:
- files_to_modify MUST be paths that exist in the provided file manifest
- files_to_create should follow the project's existing directory structure conventions
- Generate 3-8 architecture decisions and 4-10 tickets
- Be specific and actionable — each ticket should be implementable by a developer
- Return ONLY the JSON object"""


def generate_teacher_plan(repo_ir_summary: str, spec: dict, file_manifest: list[str]) -> dict:
    """Generate a teacher reference implementation plan.

    Args:
        repo_ir_summary: Condensed string representation of the RepoIR.
        spec: The synthetic spec dict.
        file_manifest: List of real file paths in the repo (for grounding).

    Returns:
        Dict with plan fields.
    """
    # Include a subset of the file manifest (truncated for token limits)
    manifest_str = "\n".join(file_manifest[:500])

    user_prompt = f"""## Codebase Analysis
{repo_ir_summary}

## File Manifest (real files in the repo)
{manifest_str}

## Feature Specification
{json.dumps(spec, indent=2)}

Generate an implementation plan for this feature. Reference REAL file paths from the manifest above."""

    raw = _call_deepseek(PLAN_SYSTEM_PROMPT, user_prompt, model="deepseek-reasoner", max_tokens=8192)
    return _extract_json(raw)


# ---------------------------------------------------------------------------
# RepoIR → summary string (condensed for prompts)
# ---------------------------------------------------------------------------

def summarize_repo_ir_for_prompt(repo_ir: dict) -> str:
    """Create a condensed text representation of a RepoIR for LLM prompts.

    Keeps it under ~2000 tokens while retaining key structural info.
    """
    sections = []

    meta = repo_ir.get("repo_metadata", {})
    sections.append(f"# {meta.get('name', 'Unknown')} ({meta.get('primary_language', '?')})")
    sections.append(f"Stars: {meta.get('star_count', 0)} | Contributors: {meta.get('num_contributors', 0)} | LOC: {meta.get('total_loc', 0)}")
    if meta.get("scale_tier"):
        sections.append(f"Scale: {meta['scale_tier']}")

    summary = repo_ir.get("architectural_summary", "")
    if summary and not summary.startswith("("):
        sections.append(f"\n## Architectural Summary\n{summary}")

    tree = repo_ir.get("directory_tree", "")
    if tree:
        sections.append(f"\n## Directory Structure\n```\n{tree}\n```")

    deps = repo_ir.get("dependencies", [])
    if deps:
        runtime_deps = [d["name"] for d in deps if d.get("dep_type") == "runtime"][:20]
        sections.append(f"\n## Dependencies\n{', '.join(runtime_deps)}")

    routes = repo_ir.get("api_routes", [])
    if routes:
        route_lines = [f"  {r['method']} {r['path']} ({r.get('framework', '?')})" for r in routes[:15]]
        sections.append(f"\n## API Routes ({len(routes)} total)\n" + "\n".join(route_lines))

    models = repo_ir.get("data_models", [])
    if models:
        model_lines = [f"  {m['name']} ({m.get('orm', '?')})" for m in models[:10]]
        sections.append(f"\n## Data Models ({len(models)} total)\n" + "\n".join(model_lines))

    infra = repo_ir.get("infrastructure", {})
    if infra:
        infra_items = []
        if infra.get("containerization"):
            infra_items.append(f"Container: {infra['containerization']}")
        if infra.get("databases"):
            infra_items.append(f"DB: {', '.join(infra['databases'])}")
        if infra.get("ci_cd"):
            infra_items.append(f"CI: {infra['ci_cd']}")
        if infra.get("cloud_provider"):
            infra_items.append(f"Cloud: {infra['cloud_provider']}")
        if infra_items:
            sections.append(f"\n## Infrastructure\n" + " | ".join(infra_items))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Full pipeline: RepoIR → training example
# ---------------------------------------------------------------------------

def generate_training_example(repo_ir_path: str, output_dir: str) -> dict | None:
    """Generate a complete training example (spec + teacher plan) from a RepoIR.

    Args:
        repo_ir_path: Path to repo_ir.json file.
        output_dir: Directory to save spec.json and teacher_plan.json.

    Returns:
        Dict with keys: repo_ir_summary, spec, teacher_plan, file_manifest.
        Returns None on failure.
    """
    try:
        with open(repo_ir_path) as f:
            repo_ir = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {repo_ir_path}: {e}")
        return None

    repo_name = repo_ir.get("repo_metadata", {}).get("name", "unknown")
    file_manifest = repo_ir.get("file_manifest", [])
    summary = summarize_repo_ir_for_prompt(repo_ir)

    # Generate synthetic spec
    logger.info(f"  Generating spec for {repo_name}...")
    try:
        spec = generate_synthetic_spec(summary, repo_name)
    except Exception as e:
        logger.error(f"  Spec generation failed: {e}")
        return None

    # Generate teacher plan
    logger.info(f"  Generating teacher plan for {repo_name}...")
    try:
        teacher_plan = generate_teacher_plan(summary, spec, file_manifest)
    except Exception as e:
        logger.error(f"  Teacher plan generation failed: {e}")
        return None

    # Save outputs
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(output_dir, "teacher_plan.json"), "w") as f:
        json.dump(teacher_plan, f, indent=2)

    logger.info(f"  Saved spec + teacher plan to {output_dir}")

    return {
        "repo_name": repo_name,
        "repo_ir_summary": summary,
        "spec": spec,
        "teacher_plan": teacher_plan,
        "file_manifest": file_manifest,
        "diagram_paths": repo_ir.get("diagram_paths", []),
    }
