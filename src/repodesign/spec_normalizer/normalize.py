"""Spec Normalizer: convert raw PRD text into canonical Spec JSON."""

from __future__ import annotations

import json
import logging
import os

from ..schemas.spec import Constraints, ScaleConstraints, ScaleTier, Spec

logger = logging.getLogger(__name__)

NORMALIZER_SYSTEM_PROMPT = """You are a product specification normalizer. Given a raw product requirements document (PRD) or feature description, extract structured information into a JSON object.

Return a JSON object with exactly these fields:
{
  "project_name": "string - name of the project/feature",
  "description": "string - 1-2 sentence summary",
  "functional_requirements": ["list of specific functional requirements"],
  "scale": {
    "expected_users": number or null,
    "team_size": number or null,
    "budget": "string or null",
    "timeline": "string or null",
    "performance_targets": {} or specific targets,
    "availability_target": "string or null (e.g., '99.9%')"
  },
  "scale_tier": "hobby" | "startup" | "growth" | "enterprise",
  "constraints": {
    "must_use": ["technologies that must be used"],
    "must_not_use": ["technologies that must not be used"],
    "no_new_infrastructure": false,
    "other": ["other constraints"]
  }
}

Scale tier definitions:
- hobby: <1k users, 1 developer, minimal infrastructure
- startup: 1k-50k users, 2-5 developers, basic infrastructure
- growth: 50k-1M users, 10-30 developers, production infrastructure
- enterprise: 1M+ users, 30+ developers, high-availability infrastructure

If information is not provided in the PRD, use null for optional fields. Always infer scale_tier from available signals. Return ONLY the JSON object, no markdown formatting."""


def normalize_spec(
    raw_prd: str,
    spec_id: str = "spec-001",
    provider: str = "anthropic",
) -> Spec:
    """Convert raw PRD text to a normalized Spec.

    Args:
        raw_prd: Raw product requirements text.
        spec_id: Unique identifier for this spec.
        provider: LLM provider ("anthropic" or "openai").

    Returns:
        Validated Spec instance.
    """
    if provider == "anthropic":
        raw_json = _call_anthropic(raw_prd)
    elif provider == "openai":
        raw_json = _call_openai(raw_prd)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Parse and validate
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        # Try to extract JSON from markdown code block
        import re

        match = re.search(r"```(?:json)?\s*(.*?)```", raw_json, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e

    return Spec(
        id=spec_id,
        project_name=data.get("project_name", "Unknown"),
        description=data.get("description", ""),
        functional_requirements=data.get("functional_requirements", []),
        scale=ScaleConstraints(**data.get("scale", {})),
        scale_tier=ScaleTier(data.get("scale_tier", "startup")),
        constraints=Constraints(**data.get("constraints", {})),
        raw_prd=raw_prd,
    )


def normalize_spec_manual(
    project_name: str,
    description: str,
    functional_requirements: list[str],
    expected_users: int | None = None,
    team_size: int | None = None,
    scale_tier: str = "startup",
    must_use: list[str] | None = None,
    spec_id: str = "spec-001",
) -> Spec:
    """Create a Spec manually without LLM (for testing or when API is unavailable)."""
    return Spec(
        id=spec_id,
        project_name=project_name,
        description=description,
        functional_requirements=functional_requirements,
        scale=ScaleConstraints(expected_users=expected_users, team_size=team_size),
        scale_tier=ScaleTier(scale_tier),
        constraints=Constraints(must_use=must_use or []),
    )


def _call_anthropic(raw_prd: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=NORMALIZER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Normalize this PRD:\n\n{raw_prd}"}],
    )
    return response.content[0].text


def _call_openai(raw_prd: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    import openai

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2048,
        messages=[
            {"role": "system", "content": NORMALIZER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Normalize this PRD:\n\n{raw_prd}"},
        ],
    )
    return response.choices[0].message.content or ""
