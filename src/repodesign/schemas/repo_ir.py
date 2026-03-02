"""Repo IR schema: structured intermediate representation of repository architecture."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .spec import ScaleTier


class RepoMetadata(BaseModel):
    """High-level repository metadata."""

    name: str
    url: str
    primary_language: str
    language_breakdown: dict[str, float] = Field(default_factory=dict)
    total_loc: int = 0
    num_contributors: int = 0
    star_count: int = 0
    scale_tier: Optional[ScaleTier] = None


class Dependency(BaseModel):
    """An external dependency (package)."""

    name: str
    version: Optional[str] = None
    dep_type: Literal["runtime", "dev", "optional"] = "runtime"


class InternalImport(BaseModel):
    """An import relationship between files within the repo."""

    from_file: str
    to_file: str
    imported_names: list[str] = Field(default_factory=list)


class APIRoute(BaseModel):
    """A detected API endpoint."""

    path: str
    method: str
    handler_file: str
    handler_function: Optional[str] = None
    auth_required: Optional[bool] = None
    framework: str  # flask, fastapi, django, express


class DataModelField(BaseModel):
    """A field within a data model."""

    name: str
    field_type: str
    constraints: list[str] = Field(default_factory=list)


class DataModel(BaseModel):
    """A detected ORM/database model."""

    name: str
    file_path: str
    fields: list[DataModelField] = Field(default_factory=list)
    orm: str  # django, sqlalchemy, prisma, mongoose, etc.
    relationships: list[str] = Field(default_factory=list)


class InfraConfig(BaseModel):
    """Detected infrastructure configuration."""

    containerization: Optional[str] = None
    databases: list[str] = Field(default_factory=list)
    caching: list[str] = Field(default_factory=list)
    message_queues: list[str] = Field(default_factory=list)
    ci_cd: Optional[str] = None
    cloud_provider: Optional[str] = None
    deployment_files: list[str] = Field(default_factory=list)


class RepoIR(BaseModel):
    """Structured intermediate representation of a repository's architecture.

    This is the core data structure of the RepoDesign system. It captures
    a repository's architecture through deterministic code analysis augmented
    by LLM summarization.
    """

    repo_metadata: RepoMetadata
    dependencies: list[Dependency] = Field(default_factory=list)
    internal_imports: list[InternalImport] = Field(default_factory=list)
    api_routes: list[APIRoute] = Field(default_factory=list)
    data_models: list[DataModel] = Field(default_factory=list)
    infrastructure: InfraConfig = Field(default_factory=InfraConfig)
    directory_tree: str = ""
    key_directories: dict[str, str] = Field(default_factory=dict)
    architectural_summary: str = ""
    extraction_timestamp: str = ""
    extraction_warnings: list[str] = Field(default_factory=list)
