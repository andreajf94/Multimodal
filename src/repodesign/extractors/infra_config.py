"""Infrastructure config extractor: Docker, K8s, Terraform, CI/CD."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Known database images/services
DB_PATTERNS = {
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mysql": "mysql",
    "mariadb": "mariadb",
    "mongo": "mongodb",
    "mongodb": "mongodb",
    "redis": "redis",
    "sqlite": "sqlite",
    "cassandra": "cassandra",
    "dynamodb": "dynamodb",
    "elasticsearch": "elasticsearch",
    "opensearch": "opensearch",
    "cockroach": "cockroachdb",
    "neo4j": "neo4j",
    "influxdb": "influxdb",
    "timescaledb": "timescaledb",
}

CACHE_PATTERNS = {
    "redis": "redis",
    "memcached": "memcached",
    "varnish": "varnish",
}

QUEUE_PATTERNS = {
    "rabbitmq": "rabbitmq",
    "kafka": "kafka",
    "celery": "celery",
    "bull": "bull",
    "sqs": "sqs",
    "nats": "nats",
    "pulsar": "pulsar",
}

CLOUD_PATTERNS = {
    "aws": "aws",
    "amazon": "aws",
    "gcp": "gcp",
    "google-cloud": "gcp",
    "azure": "azure",
    "digitalocean": "digitalocean",
    "heroku": "heroku",
    "vercel": "vercel",
    "netlify": "netlify",
    "fly.io": "fly",
}


def _parse_docker_compose(path: str) -> dict:
    """Parse docker-compose.yml to detect services."""
    result = {"databases": set(), "caching": set(), "queues": set(), "services": []}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return result
        services = data.get("services", {})
        if not isinstance(services, dict):
            return result

        for name, svc in services.items():
            if not isinstance(svc, dict):
                continue
            image = str(svc.get("image", "")).lower()
            name_lower = name.lower()
            combined = f"{image} {name_lower}"

            for pattern, db_name in DB_PATTERNS.items():
                if pattern in combined:
                    result["databases"].add(db_name)
            for pattern, cache_name in CACHE_PATTERNS.items():
                if pattern in combined:
                    result["caching"].add(cache_name)
            for pattern, queue_name in QUEUE_PATTERNS.items():
                if pattern in combined:
                    result["queues"].add(queue_name)
            result["services"].append(name)
    except Exception as e:
        logger.warning(f"Failed to parse {path}: {e}")
    return result


def _parse_dockerfile(path: str) -> dict:
    """Parse Dockerfile for base image and exposed ports."""
    result = {"base_images": [], "ports": []}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.upper().startswith("FROM "):
                    result["base_images"].append(line.split()[1])
                elif line.upper().startswith("EXPOSE "):
                    result["ports"].extend(line.split()[1:])
    except Exception as e:
        logger.warning(f"Failed to parse {path}: {e}")
    return result


def _detect_ci_cd(repo_path: str) -> str | None:
    """Detect CI/CD system."""
    repo = Path(repo_path)
    if (repo / ".github" / "workflows").exists():
        return "github-actions"
    if (repo / ".gitlab-ci.yml").exists():
        return "gitlab-ci"
    if (repo / "Jenkinsfile").exists():
        return "jenkins"
    if (repo / ".circleci").exists():
        return "circleci"
    if (repo / ".travis.yml").exists():
        return "travis"
    if (repo / "bitbucket-pipelines.yml").exists():
        return "bitbucket-pipelines"
    if (repo / "azure-pipelines.yml").exists():
        return "azure-devops"
    return None


def _detect_cloud_provider(repo_path: str) -> str | None:
    """Detect cloud provider from config files."""
    repo = Path(repo_path)

    # Check for AWS
    if (repo / "serverless.yml").exists() or (repo / "template.yaml").exists():
        return "aws"
    if (repo / "cdk.json").exists() or (repo / "samconfig.toml").exists():
        return "aws"

    # Check for GCP
    if (repo / "app.yaml").exists() or (repo / "cloudbuild.yaml").exists():
        return "gcp"

    # Check Terraform files for provider
    tf_dir = repo / "terraform"
    if not tf_dir.exists():
        tf_dir = repo / "infra"
    if tf_dir.exists():
        for tf_file in tf_dir.glob("*.tf"):
            try:
                content = tf_file.read_text(errors="ignore").lower()
                for pattern, provider in CLOUD_PATTERNS.items():
                    if pattern in content:
                        return provider
            except OSError:
                pass

    return None


def _find_deployment_files(repo_path: str) -> list[str]:
    """Find all infrastructure/deployment-related files."""
    repo = Path(repo_path)
    patterns = [
        "Dockerfile",
        "docker-compose*.yml",
        "docker-compose*.yaml",
        ".dockerignore",
        "*.tf",
        "*.tfvars",
        "k8s/*.yml",
        "k8s/*.yaml",
        "kubernetes/*.yml",
        "kubernetes/*.yaml",
        "helm/**/*.yaml",
        ".github/workflows/*.yml",
        ".gitlab-ci.yml",
        "Jenkinsfile",
        "Procfile",
        "serverless.yml",
        "app.yaml",
        "fly.toml",
        "render.yaml",
        "railway.json",
        "vercel.json",
        "netlify.toml",
        "nginx.conf",
        "Caddyfile",
    ]
    found = []
    for pattern in patterns:
        for match in repo.glob(pattern):
            found.append(str(match.relative_to(repo)))
    return sorted(set(found))


def _detect_containerization(repo_path: str) -> str | None:
    """Detect containerization approach."""
    repo = Path(repo_path)
    # Check for k8s first (implies Docker too)
    if (repo / "k8s").exists() or (repo / "kubernetes").exists() or list(repo.glob("**/k8s*.yml")):
        return "kubernetes"
    if list(repo.glob("docker-compose*.yml")) or list(repo.glob("docker-compose*.yaml")):
        return "docker-compose"
    if (repo / "Dockerfile").exists():
        return "docker"
    return None


def _scan_for_services_in_code(repo_path: str) -> dict:
    """Scan code files for database/cache/queue connection strings and imports."""
    databases: set[str] = set()
    caching: set[str] = set()
    queues: set[str] = set()

    repo = Path(repo_path)
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv"}

    # Check Python requirements
    for dep_file in ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]:
        p = repo / dep_file
        if p.exists():
            try:
                content = p.read_text(errors="ignore").lower()
                if "psycopg" in content or "django" in content:
                    databases.add("postgresql")
                if "pymysql" in content or "mysqlclient" in content:
                    databases.add("mysql")
                if "pymongo" in content or "motor" in content:
                    databases.add("mongodb")
                if "redis" in content:
                    caching.add("redis")
                if "celery" in content:
                    queues.add("celery")
                if "pika" in content or "amqp" in content:
                    queues.add("rabbitmq")
                if "kafka" in content:
                    queues.add("kafka")
                if "sqlalchemy" in content and "sqlite" not in content:
                    pass  # SQLAlchemy could be any DB
                if "sqlite" in content:
                    databases.add("sqlite")
            except OSError:
                pass

    # Check package.json
    pkg = repo / "package.json"
    if pkg.exists():
        try:
            content = pkg.read_text(errors="ignore").lower()
            if "pg" in content or "postgres" in content:
                databases.add("postgresql")
            if "mysql" in content:
                databases.add("mysql")
            if "mongoose" in content or "mongodb" in content:
                databases.add("mongodb")
            if "redis" in content or "ioredis" in content:
                caching.add("redis")
            if "bull" in content or "bullmq" in content:
                queues.add("bull")
            if "amqplib" in content:
                queues.add("rabbitmq")
            if "kafkajs" in content:
                queues.add("kafka")
        except OSError:
            pass

    return {
        "databases": databases,
        "caching": caching,
        "queues": queues,
    }


def extract_infra_config(repo_path: str) -> dict:
    """Extract infrastructure configuration from a repository."""
    repo = Path(repo_path)
    databases: set[str] = set()
    caching: set[str] = set()
    queues: set[str] = set()

    # Parse docker-compose
    for compose_file in list(repo.glob("docker-compose*.yml")) + list(repo.glob("docker-compose*.yaml")):
        dc = _parse_docker_compose(str(compose_file))
        databases.update(dc["databases"])
        caching.update(dc["caching"])
        queues.update(dc["queues"])

    # Scan code for service patterns
    code_services = _scan_for_services_in_code(repo_path)
    databases.update(code_services["databases"])
    caching.update(code_services["caching"])
    queues.update(code_services["queues"])

    return {
        "containerization": _detect_containerization(repo_path),
        "databases": sorted(databases),
        "caching": sorted(caching),
        "message_queues": sorted(queues),
        "ci_cd": _detect_ci_cd(repo_path),
        "cloud_provider": _detect_cloud_provider(repo_path),
        "deployment_files": _find_deployment_files(repo_path),
    }
