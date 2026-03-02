"""Tests for extraction pipeline (runs against our own repo as a smoke test)."""

import os
from pathlib import Path

import pytest

# The repo root is 2 levels up from tests/
REPO_ROOT = str(Path(__file__).resolve().parent.parent)


class TestDirectoryAnalysis:
    def test_generate_tree(self):
        from repodesign.extractors.directory_analysis import generate_directory_tree

        tree = generate_directory_tree(REPO_ROOT, max_depth=2)
        assert "src" in tree
        assert "scripts" in tree

    def test_count_loc(self):
        from repodesign.extractors.directory_analysis import count_loc_by_language

        loc, total = count_loc_by_language(REPO_ROOT)
        assert "python" in loc
        assert total > 0

    def test_key_directories(self):
        from repodesign.extractors.directory_analysis import identify_key_directories

        dirs = identify_key_directories(REPO_ROOT)
        assert "source" in dirs  # Should find src/
        assert "tests" in dirs  # Should find tests/

    def test_full_extract(self):
        from repodesign.extractors.directory_analysis import extract_directory_info

        info = extract_directory_info(REPO_ROOT)
        assert info["primary_language"] == "python"
        assert info["total_loc"] > 0
        assert "directory_tree" in info


class TestDependencyGraph:
    def test_extract_from_pyproject(self):
        from repodesign.extractors.dependency_graph import extract_external_dependencies

        deps = extract_external_dependencies(REPO_ROOT)
        dep_names = {d["name"] for d in deps}
        assert "pydantic" in dep_names

    def test_internal_imports(self):
        from repodesign.extractors.dependency_graph import extract_internal_imports

        imports = extract_internal_imports(REPO_ROOT)
        # Our own code should have internal imports
        assert isinstance(imports, list)


class TestAPIRoutes:
    def test_no_routes_in_this_repo(self):
        from repodesign.extractors.api_routes import extract_api_routes

        routes = extract_api_routes(REPO_ROOT)
        # Our repo doesn't have Flask/Django routes, should return empty
        assert isinstance(routes, list)


class TestInfraConfig:
    def test_extract_infra(self):
        from repodesign.extractors.infra_config import extract_infra_config

        infra = extract_infra_config(REPO_ROOT)
        assert isinstance(infra, dict)
        assert "databases" in infra
        assert "ci_cd" in infra


class TestORMModels:
    def test_extract_models(self):
        from repodesign.extractors.orm_models import extract_data_models

        models = extract_data_models(REPO_ROOT)
        # Our repo doesn't have Django/SQLAlchemy models
        assert isinstance(models, list)


class TestPipeline:
    def test_full_extraction(self):
        from repodesign.extractors.pipeline import extract_repo_ir

        ir = extract_repo_ir(REPO_ROOT, skip_llm=True)
        assert ir.repo_metadata.name == "Multimodal"
        assert ir.repo_metadata.primary_language == "python"
        assert ir.repo_metadata.total_loc > 0
        assert len(ir.dependencies) > 0
        assert ir.architectural_summary == "(LLM summary skipped)"


class TestRGS:
    def test_compute_rgs(self):
        from repodesign.evaluation.repo_grounding_score import compute_rgs
        from repodesign.schemas import ImplementationPlan, ScaleTier, Ticket

        # Plan with real paths from our repo
        plan = ImplementationPlan(
            spec_id="test",
            repo_id="test",
            scale_tier=ScaleTier.HOBBY,
            tickets=[
                Ticket(
                    id="t1",
                    title="Test",
                    description="Test",
                    files_to_modify=["pyproject.toml", "src/repodesign/__init__.py"],
                    files_to_create=["nonexistent/fake/file.py"],
                ),
            ],
        )
        result = compute_rgs(plan, REPO_ROOT)
        assert result.total_paths == 3
        assert result.valid_paths == 2  # pyproject.toml and __init__.py exist
        assert len(result.invalid_paths) == 1
        assert 0.6 < result.score < 0.7  # 2/3 ≈ 0.667
