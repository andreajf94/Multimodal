"""Repo Grounding Score (RGS): primary evaluation metric.

Measures what percentage of file paths referenced in a generated
implementation plan actually exist in the target repository.
This is fully deterministic — no LLM-as-judge subjectivity.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ..schemas.plan import ImplementationPlan

logger = logging.getLogger(__name__)


@dataclass
class RGSResult:
    """Result of a Repo Grounding Score computation."""

    score: float  # 0.0 to 1.0
    total_paths: int
    valid_paths: int
    invalid_paths: list[str]
    valid_path_list: list[str]


def compute_rgs(plan: ImplementationPlan, repo_path: str) -> RGSResult:
    """Compute the Repo Grounding Score for a plan against a repository.

    RGS = (number of referenced paths that exist) / (total referenced paths)

    Args:
        plan: The implementation plan to evaluate.
        repo_path: Path to the target repository.

    Returns:
        RGSResult with score and detailed breakdown.
    """
    referenced_paths = plan.get_all_referenced_paths()

    if not referenced_paths:
        return RGSResult(
            score=1.0,  # No paths to ground = vacuously true
            total_paths=0,
            valid_paths=0,
            invalid_paths=[],
            valid_path_list=[],
        )

    repo = Path(repo_path)
    valid: list[str] = []
    invalid: list[str] = []

    for ref_path in referenced_paths:
        # Normalize the path
        normalized = ref_path.strip().lstrip("/")

        # Check if the path exists in the repo
        full_path = repo / normalized
        if full_path.exists():
            valid.append(ref_path)
        else:
            # Also check with common variations
            found = False
            # Try without leading src/
            if normalized.startswith("src/"):
                if (repo / normalized[4:]).exists():
                    valid.append(ref_path)
                    found = True
            # Try with src/ prefix
            if not found and not normalized.startswith("src/"):
                if (repo / "src" / normalized).exists():
                    valid.append(ref_path)
                    found = True
            if not found:
                invalid.append(ref_path)

    total = len(referenced_paths)
    score = len(valid) / total if total > 0 else 1.0

    return RGSResult(
        score=score,
        total_paths=total,
        valid_paths=len(valid),
        invalid_paths=invalid,
        valid_path_list=valid,
    )


def compute_rgs_batch(
    plans: list[tuple[ImplementationPlan, str]],
) -> dict:
    """Compute RGS across multiple plan-repo pairs.

    Args:
        plans: List of (plan, repo_path) tuples.

    Returns:
        Dict with aggregate statistics.
    """
    results: list[RGSResult] = []
    for plan, repo_path in plans:
        result = compute_rgs(plan, repo_path)
        results.append(result)

    scores = [r.score for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "mean_rgs": avg_score,
        "min_rgs": min(scores) if scores else 0.0,
        "max_rgs": max(scores) if scores else 0.0,
        "num_plans": len(results),
        "total_paths_checked": sum(r.total_paths for r in results),
        "total_valid_paths": sum(r.valid_paths for r in results),
        "results": results,
    }
