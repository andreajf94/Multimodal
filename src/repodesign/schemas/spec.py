"""Spec schema: normalized product requirement document."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ScaleTier(str, Enum):
    """Scale tier classification for a project."""

    HOBBY = "hobby"  # <1k users, 1 dev
    STARTUP = "startup"  # 1k-50k users, 2-5 devs
    GROWTH = "growth"  # 50k-1M users, 10-30 devs
    ENTERPRISE = "enterprise"  # 1M+ users, 30+ devs


class ScaleConstraints(BaseModel):
    """Non-functional scale constraints for the project."""

    expected_users: Optional[int] = None
    team_size: Optional[int] = None
    budget: Optional[str] = None
    timeline: Optional[str] = None
    performance_targets: dict = Field(default_factory=dict)
    availability_target: Optional[str] = None


class Constraints(BaseModel):
    """Hard constraints from the specification."""

    must_use: list[str] = Field(default_factory=list)
    must_not_use: list[str] = Field(default_factory=list)
    no_new_infrastructure: bool = False
    other: list[str] = Field(default_factory=list)


class Spec(BaseModel):
    """Normalized product specification."""

    id: str
    project_name: str
    description: str
    functional_requirements: list[str]
    scale: ScaleConstraints
    scale_tier: ScaleTier
    constraints: Constraints = Field(default_factory=Constraints)
    raw_prd: Optional[str] = None
