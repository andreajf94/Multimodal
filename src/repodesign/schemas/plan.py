"""Implementation Plan schema: codebase-grounded design plan."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .spec import ScaleTier

# The 20 design dimensions from the Kleppmann-based framework
DESIGN_DIMENSIONS = [
    "problem_scope",
    "functional_requirements",
    "performance",
    "availability",
    "consistency",
    "durability",
    "api_design",
    "data_modeling",
    "caching",
    "replication",
    "partitioning",
    "networking",
    "message_queues",
    "compute_architecture",
    "batch_stream_processing",
    "security",
    "privacy",
    "failure_resilience",
    "monitoring",
    "deployment",
]


class ArchitectureDecision(BaseModel):
    """A single architecture decision along one design dimension."""

    dimension: str  # One of DESIGN_DIMENSIONS
    recommendation: str
    rationale: str
    alternatives_considered: list[str] = Field(default_factory=list)
    files_affected: list[str] = Field(default_factory=list)


class Ticket(BaseModel):
    """An actionable implementation ticket with real file paths."""

    id: str
    title: str
    description: str
    files_to_modify: list[str] = Field(default_factory=list)
    files_to_create: list[str] = Field(default_factory=list)
    estimated_effort: Literal["small", "medium", "large"] = "medium"
    dependencies: list[str] = Field(default_factory=list)


class TechnologyChoice(BaseModel):
    """A specific technology recommendation."""

    category: str  # e.g., "database", "caching", "deployment"
    choice: str
    rationale: str


class ImplementationPlan(BaseModel):
    """A codebase-grounded implementation plan.

    This is the training target for the RepoDesign model. Plans are
    evaluated by the Repo Grounding Score (RGS) which checks whether
    file paths referenced in the plan actually exist in the repo.
    """

    spec_id: str
    repo_id: str
    scale_tier: ScaleTier
    architecture_decisions: list[ArchitectureDecision] = Field(default_factory=list)
    tickets: list[Ticket] = Field(default_factory=list)
    technology_choices: list[TechnologyChoice] = Field(default_factory=list)

    def get_all_referenced_paths(self) -> list[str]:
        """Extract all file paths referenced in this plan.

        Used by the Repo Grounding Score (RGS) metric to check
        whether paths actually exist in the target repo.
        """
        paths: set[str] = set()
        for decision in self.architecture_decisions:
            paths.update(decision.files_affected)
        for ticket in self.tickets:
            paths.update(ticket.files_to_modify)
            paths.update(ticket.files_to_create)
        return sorted(paths)
