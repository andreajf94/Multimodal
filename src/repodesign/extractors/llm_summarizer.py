"""LLM-based architectural summarizer: generates natural language summary of repo architecture."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """You are a senior software architect. Given structured information about a code repository (its directory tree, dependencies, API routes, data models, and infrastructure), produce a concise architectural summary.

Your summary should cover:
1. **Overall Architecture Pattern**: Is this a monolith, microservices, serverless, etc.?
2. **Tech Stack**: Key frameworks, databases, and infrastructure choices.
3. **API Surface**: How many endpoints, what patterns (REST, GraphQL, RPC)?
4. **Data Layer**: What databases/ORMs are used, key models and their relationships.
5. **Infrastructure Maturity**: CI/CD, containerization, deployment approach.
6. **Strengths**: What is well-structured about this codebase?
7. **Weaknesses / Gaps**: What's missing or could be improved?

Keep the summary to 200-400 words. Be specific—reference actual file paths and technology names from the provided data."""


def _build_summary_prompt(extracted_data: dict) -> str:
    """Build the user prompt from extracted data."""
    sections = []

    sections.append(f"## Directory Tree\n```\n{extracted_data.get('directory_tree', 'N/A')}\n```")

    deps = extracted_data.get("dependencies", [])
    if deps:
        runtime = [d["name"] for d in deps if d.get("dep_type") == "runtime"][:30]
        sections.append(f"## External Dependencies (runtime)\n{', '.join(runtime)}")

    routes = extracted_data.get("api_routes", [])
    if routes:
        route_lines = [f"  {r['method']} {r['path']} -> {r['handler_file']}" for r in routes[:20]]
        sections.append(f"## API Routes ({len(routes)} total)\n" + "\n".join(route_lines))

    models = extracted_data.get("data_models", [])
    if models:
        model_lines = [f"  {m['name']} ({m['orm']}) in {m['file_path']}" for m in models[:15]]
        sections.append(f"## Data Models ({len(models)} total)\n" + "\n".join(model_lines))

    infra = extracted_data.get("infrastructure", {})
    if infra:
        sections.append(f"## Infrastructure\n{json.dumps(infra, indent=2)}")

    key_dirs = extracted_data.get("key_directories", {})
    if key_dirs:
        sections.append(f"## Key Directories\n{json.dumps(key_dirs, indent=2)}")

    return "\n\n".join(sections)


def generate_architectural_summary(extracted_data: dict, provider: str = "anthropic") -> str:
    """Generate an LLM-based architectural summary.

    Args:
        extracted_data: Combined output from all deterministic extractors.
        provider: "anthropic" or "openai".

    Returns:
        Natural language architectural summary string.
    """
    user_prompt = _build_summary_prompt(extracted_data)

    if provider == "anthropic":
        return _call_anthropic(user_prompt)
    elif provider == "openai":
        return _call_openai(user_prompt)
    else:
        logger.warning(f"Unknown provider {provider}, returning empty summary")
        return ""


def _call_anthropic(user_prompt: str) -> str:
    """Call Anthropic API for summary."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, skipping LLM summary")
        return "(LLM summary unavailable - ANTHROPIC_API_KEY not set)"

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SUMMARY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Anthropic API call failed: {e}")
        return f"(LLM summary failed: {e})"


def _call_openai(user_prompt: str) -> str:
    """Call OpenAI API for summary."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping LLM summary")
        return "(LLM summary unavailable - OPENAI_API_KEY not set)"

    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return f"(LLM summary failed: {e})"
