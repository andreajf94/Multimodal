"""Extraction pipeline orchestrator: runs all extractors and produces a RepoIR."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..schemas.repo_ir import (
    APIRoute,
    DataModel,
    DataModelField,
    Dependency,
    InfraConfig,
    InternalImport,
    RepoIR,
    RepoMetadata,
)
from ..schemas.spec import ScaleTier
from .api_routes import extract_api_routes
from .dependency_graph import extract_dependency_info
from .directory_analysis import extract_directory_info
from .infra_config import extract_infra_config
from .llm_summarizer import generate_architectural_summary
from .orm_models import extract_data_models

logger = logging.getLogger(__name__)


def extract_repo_ir(
    repo_path: str,
    repo_url: str = "",
    star_count: int = 0,
    num_contributors: int = 0,
    skip_llm: bool = False,
    llm_provider: str = "anthropic",
) -> RepoIR:
    """Extract a complete Repo IR from a local repository.

    Args:
        repo_path: Path to the cloned repository.
        repo_url: GitHub URL of the repository.
        star_count: Number of GitHub stars.
        num_contributors: Number of contributors.
        skip_llm: If True, skip LLM-based summarization.
        llm_provider: "anthropic" or "openai".

    Returns:
        A fully populated RepoIR instance.
    """
    warnings: list[str] = []
    repo = Path(repo_path)
    repo_name = repo.name

    logger.info(f"Extracting Repo IR for {repo_name} at {repo_path}")

    # 1. Directory analysis
    logger.info("  [1/6] Directory analysis...")
    try:
        dir_info = extract_directory_info(repo_path)
    except Exception as e:
        warnings.append(f"Directory analysis failed: {e}")
        dir_info = {"directory_tree": "", "total_loc": 0, "language_breakdown": {}, "primary_language": "unknown", "key_directories": {}}

    # 2. Dependencies
    logger.info("  [2/6] Dependency extraction...")
    try:
        dep_info = extract_dependency_info(repo_path)
    except Exception as e:
        warnings.append(f"Dependency extraction failed: {e}")
        dep_info = {"dependencies": [], "internal_imports": []}

    # 3. API routes
    logger.info("  [3/6] API route extraction...")
    try:
        routes = extract_api_routes(repo_path)
    except Exception as e:
        warnings.append(f"API route extraction failed: {e}")
        routes = []

    # 4. ORM models
    logger.info("  [4/6] ORM model extraction...")
    try:
        models = extract_data_models(repo_path)
    except Exception as e:
        warnings.append(f"ORM model extraction failed: {e}")
        models = []

    # 5. Infrastructure
    logger.info("  [5/6] Infrastructure config extraction...")
    try:
        infra = extract_infra_config(repo_path)
    except Exception as e:
        warnings.append(f"Infrastructure extraction failed: {e}")
        infra = {}

    # 6. LLM summary
    if skip_llm:
        summary = "(LLM summary skipped)"
    else:
        logger.info("  [6/6] LLM architectural summary...")
        try:
            combined = {
                "directory_tree": dir_info.get("directory_tree", ""),
                "dependencies": dep_info.get("dependencies", []),
                "api_routes": routes,
                "data_models": models,
                "infrastructure": infra,
                "key_directories": dir_info.get("key_directories", {}),
            }
            summary = generate_architectural_summary(combined, provider=llm_provider)
        except Exception as e:
            warnings.append(f"LLM summary failed: {e}")
            summary = f"(LLM summary failed: {e})"

    # Build RepoIR
    metadata = RepoMetadata(
        name=repo_name,
        url=repo_url,
        primary_language=dir_info.get("primary_language", "unknown"),
        language_breakdown=dir_info.get("language_breakdown", {}),
        total_loc=dir_info.get("total_loc", 0),
        num_contributors=num_contributors,
        star_count=star_count,
        scale_tier=_classify_scale(star_count, num_contributors, infra),
    )

    dependencies = [Dependency(**d) for d in dep_info.get("dependencies", [])]
    internal_imports = [InternalImport(**i) for i in dep_info.get("internal_imports", [])]
    api_routes = [APIRoute(**r) for r in routes]
    data_models_typed = [
        DataModel(
            name=m["name"],
            file_path=m["file_path"],
            fields=[DataModelField(**f) for f in m.get("fields", [])],
            orm=m["orm"],
            relationships=m.get("relationships", []),
        )
        for m in models
    ]
    infra_config = InfraConfig(**infra) if infra else InfraConfig()

    return RepoIR(
        repo_metadata=metadata,
        dependencies=dependencies,
        internal_imports=internal_imports,
        api_routes=api_routes,
        data_models=data_models_typed,
        infrastructure=infra_config,
        directory_tree=dir_info.get("directory_tree", ""),
        key_directories=dir_info.get("key_directories", {}),
        architectural_summary=summary,
        extraction_timestamp=datetime.now(timezone.utc).isoformat(),
        extraction_warnings=warnings,
    )


def _classify_scale(star_count: int, num_contributors: int, infra: dict) -> ScaleTier | None:
    """Heuristic scale classification based on repo metadata."""
    has_k8s = infra.get("containerization") == "kubernetes"
    has_ci = infra.get("ci_cd") is not None
    has_docker = infra.get("containerization") in ("docker", "docker-compose", "kubernetes")

    if star_count >= 5000 or num_contributors >= 30 or has_k8s:
        return ScaleTier.ENTERPRISE
    if star_count >= 500 or num_contributors >= 5 or (has_docker and has_ci):
        return ScaleTier.GROWTH
    if star_count >= 10 or num_contributors >= 2 or has_ci:
        return ScaleTier.STARTUP
    return ScaleTier.HOBBY


def save_repo_ir(repo_ir: RepoIR, output_path: str) -> None:
    """Save RepoIR to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(repo_ir.model_dump_json(indent=2))
    logger.info(f"Saved Repo IR to {output_path}")


def load_repo_ir(input_path: str) -> RepoIR:
    """Load RepoIR from JSON file."""
    with open(input_path) as f:
        data = json.load(f)
    return RepoIR(**data)
