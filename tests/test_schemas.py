"""Tests for Pydantic schemas."""

import json

import pytest

from repodesign.schemas import (
    ArchitectureDecision,
    Constraints,
    DataModel,
    DataModelField,
    Dependency,
    ImplementationPlan,
    InfraConfig,
    RepoIR,
    RepoMetadata,
    ScaleConstraints,
    ScaleTier,
    Spec,
    Ticket,
    TechnologyChoice,
)


class TestSpec:
    def test_basic_creation(self):
        spec = Spec(
            id="test-001",
            project_name="TestApp",
            description="A test application",
            functional_requirements=["User login", "Dashboard"],
            scale=ScaleConstraints(expected_users=5000, team_size=3),
            scale_tier=ScaleTier.STARTUP,
        )
        assert spec.id == "test-001"
        assert spec.scale_tier == ScaleTier.STARTUP
        assert len(spec.functional_requirements) == 2

    def test_serialization_roundtrip(self):
        spec = Spec(
            id="test-002",
            project_name="RoundTrip",
            description="Roundtrip test",
            functional_requirements=["req1"],
            scale=ScaleConstraints(expected_users=100),
            scale_tier=ScaleTier.HOBBY,
            constraints=Constraints(must_use=["PostgreSQL"], no_new_infrastructure=True),
        )
        json_str = spec.model_dump_json()
        restored = Spec.model_validate_json(json_str)
        assert restored.id == spec.id
        assert restored.constraints.must_use == ["PostgreSQL"]
        assert restored.constraints.no_new_infrastructure is True

    def test_scale_tiers(self):
        for tier in ScaleTier:
            assert tier.value in ("hobby", "startup", "growth", "enterprise")


class TestRepoIR:
    def test_minimal_creation(self):
        ir = RepoIR(
            repo_metadata=RepoMetadata(
                name="test-repo",
                url="https://github.com/test/repo",
                primary_language="python",
            ),
        )
        assert ir.repo_metadata.name == "test-repo"
        assert ir.dependencies == []
        assert ir.api_routes == []

    def test_full_creation(self):
        ir = RepoIR(
            repo_metadata=RepoMetadata(
                name="full-repo",
                url="https://github.com/test/full",
                primary_language="python",
                language_breakdown={"python": 0.8, "javascript": 0.2},
                total_loc=5000,
                num_contributors=5,
                star_count=100,
                scale_tier=ScaleTier.STARTUP,
            ),
            dependencies=[
                Dependency(name="django", version="4.2", dep_type="runtime"),
                Dependency(name="pytest", dep_type="dev"),
            ],
            api_routes=[],
            data_models=[
                DataModel(
                    name="User",
                    file_path="app/models.py",
                    fields=[DataModelField(name="email", field_type="EmailField")],
                    orm="django",
                    relationships=["ForeignKey -> Profile"],
                ),
            ],
            infrastructure=InfraConfig(
                containerization="docker-compose",
                databases=["postgresql"],
                caching=["redis"],
                ci_cd="github-actions",
            ),
        )
        assert len(ir.dependencies) == 2
        assert ir.infrastructure.databases == ["postgresql"]

    def test_serialization_roundtrip(self):
        ir = RepoIR(
            repo_metadata=RepoMetadata(
                name="roundtrip",
                url="",
                primary_language="go",
            ),
            dependencies=[Dependency(name="gin", version="v1.9")],
        )
        json_str = ir.model_dump_json()
        restored = RepoIR.model_validate_json(json_str)
        assert restored.dependencies[0].name == "gin"


class TestImplementationPlan:
    def test_get_all_referenced_paths(self):
        plan = ImplementationPlan(
            spec_id="s1",
            repo_id="r1",
            scale_tier=ScaleTier.STARTUP,
            architecture_decisions=[
                ArchitectureDecision(
                    dimension="caching",
                    recommendation="Add Redis",
                    rationale="Performance",
                    files_affected=["src/cache/client.py", "src/api/views.py"],
                ),
            ],
            tickets=[
                Ticket(
                    id="t1",
                    title="Add cache",
                    description="Implement caching",
                    files_to_modify=["src/api/views.py"],
                    files_to_create=["src/cache/client.py", "src/cache/__init__.py"],
                ),
            ],
        )
        paths = plan.get_all_referenced_paths()
        # Should be deduplicated and sorted
        assert "src/api/views.py" in paths
        assert "src/cache/client.py" in paths
        assert "src/cache/__init__.py" in paths
        assert len(paths) == 3  # Deduplicated

    def test_empty_plan_paths(self):
        plan = ImplementationPlan(
            spec_id="s1",
            repo_id="r1",
            scale_tier=ScaleTier.HOBBY,
        )
        assert plan.get_all_referenced_paths() == []
