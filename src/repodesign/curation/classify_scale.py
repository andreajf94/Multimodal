"""Scale-tier classification for curated repositories."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..schemas.spec import ScaleTier

if TYPE_CHECKING:
    from .scrape_repos import RepoEntry

logger = logging.getLogger(__name__)


def classify_scale_tier(entry: "RepoEntry") -> ScaleTier:
    """Classify a repository into a scale tier based on heuristics.

    Scale tiers (from the Kleppmann-based framework):
    - HOBBY: <1k users, 1 dev — small personal projects
    - STARTUP: 1k-50k users, 2-5 devs — early-stage products
    - GROWTH: 50k-1M users, 10-30 devs — scaling products
    - ENTERPRISE: 1M+ users, 30+ devs — mature platforms

    We approximate these from GitHub signals:
    - Stars as a proxy for user base/popularity
    - Contributors as a proxy for team size
    - Infrastructure signals (Docker, CI, K8s) as maturity indicators
    """
    score = 0

    # Star-based scoring
    if entry.star_count >= 5000:
        score += 3
    elif entry.star_count >= 500:
        score += 2
    elif entry.star_count >= 50:
        score += 1

    # Contributor-based scoring
    if entry.num_contributors >= 30:
        score += 3
    elif entry.num_contributors >= 10:
        score += 2
    elif entry.num_contributors >= 3:
        score += 1

    # Infrastructure maturity
    if entry.has_docker:
        score += 1
    if entry.has_ci:
        score += 1

    # Size-based (larger repos tend to be more mature)
    if entry.size_kb >= 10000:
        score += 1

    # Classify based on total score
    if score >= 7:
        return ScaleTier.ENTERPRISE
    elif score >= 4:
        return ScaleTier.GROWTH
    elif score >= 2:
        return ScaleTier.STARTUP
    else:
        return ScaleTier.HOBBY


def classify_all(entries: list["RepoEntry"]) -> dict[ScaleTier, list["RepoEntry"]]:
    """Classify all entries and return grouped by tier."""
    grouped: dict[ScaleTier, list] = {tier: [] for tier in ScaleTier}
    for entry in entries:
        tier = classify_scale_tier(entry)
        entry.scale_tier = tier.value
        grouped[tier].append(entry)

    # Log distribution
    for tier in ScaleTier:
        count = len(grouped[tier])
        logger.info(f"  {tier.value}: {count} repos")

    return grouped
