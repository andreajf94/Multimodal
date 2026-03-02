"""Dependency graph extractor: external packages and internal imports."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build", ".tox"}


def _parse_requirements_txt(path: str) -> list[dict]:
    """Parse requirements.txt file."""
    deps = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                # Handle: package==1.0, package>=1.0, package~=1.0, package
                match = re.match(r"^([a-zA-Z0-9_.-]+)\s*([><=!~]+\s*[\d.]+)?", line)
                if match:
                    name = match.group(1)
                    version = match.group(2).strip() if match.group(2) else None
                    deps.append({"name": name, "version": version, "dep_type": "runtime"})
    except (OSError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to parse {path}: {e}")
    return deps


def _parse_pyproject_toml(path: str) -> list[dict]:
    """Parse pyproject.toml for dependencies."""
    deps = []
    try:
        import toml

        data = toml.load(path)
        # PEP 621 style
        project_deps = data.get("project", {}).get("dependencies", [])
        for dep_str in project_deps:
            match = re.match(r"^([a-zA-Z0-9_.-]+)", dep_str)
            if match:
                deps.append({"name": match.group(1), "version": None, "dep_type": "runtime"})

        # Optional/dev deps
        optional = data.get("project", {}).get("optional-dependencies", {})
        for group, group_deps in optional.items():
            dtype: Literal["dev", "optional"] = "dev" if group in ("dev", "test", "testing") else "optional"
            for dep_str in group_deps:
                match = re.match(r"^([a-zA-Z0-9_.-]+)", dep_str)
                if match:
                    deps.append({"name": match.group(1), "version": None, "dep_type": dtype})

        # Poetry style
        poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        for name, spec in poetry_deps.items():
            if name == "python":
                continue
            version = spec if isinstance(spec, str) else None
            deps.append({"name": name, "version": version, "dep_type": "runtime"})

        poetry_dev = data.get("tool", {}).get("poetry", {}).get("dev-dependencies", {})
        for name, spec in poetry_dev.items():
            version = spec if isinstance(spec, str) else None
            deps.append({"name": name, "version": version, "dep_type": "dev"})
    except Exception as e:
        logger.warning(f"Failed to parse {path}: {e}")
    return deps


def _parse_setup_py(path: str) -> list[dict]:
    """Parse setup.py for install_requires."""
    deps = []
    try:
        with open(path) as f:
            content = f.read()
        # Simple regex for install_requires
        match = re.search(r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if match:
            for dep_str in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
                name_match = re.match(r"^([a-zA-Z0-9_.-]+)", dep_str)
                if name_match:
                    deps.append({"name": name_match.group(1), "version": None, "dep_type": "runtime"})
    except Exception as e:
        logger.warning(f"Failed to parse {path}: {e}")
    return deps


def _parse_package_json(path: str) -> list[dict]:
    """Parse package.json for npm dependencies."""
    deps = []
    try:
        with open(path) as f:
            data = json.load(f)
        for name, version in data.get("dependencies", {}).items():
            deps.append({"name": name, "version": version, "dep_type": "runtime"})
        for name, version in data.get("devDependencies", {}).items():
            deps.append({"name": name, "version": version, "dep_type": "dev"})
    except Exception as e:
        logger.warning(f"Failed to parse {path}: {e}")
    return deps


def _parse_go_mod(path: str) -> list[dict]:
    """Parse go.mod for Go dependencies."""
    deps = []
    try:
        with open(path) as f:
            in_require = False
            for line in f:
                line = line.strip()
                if line.startswith("require ("):
                    in_require = True
                    continue
                if in_require and line == ")":
                    in_require = False
                    continue
                if in_require:
                    parts = line.split()
                    if len(parts) >= 2:
                        deps.append({"name": parts[0], "version": parts[1], "dep_type": "runtime"})
                elif line.startswith("require "):
                    parts = line.split()
                    if len(parts) >= 3:
                        deps.append({"name": parts[1], "version": parts[2], "dep_type": "runtime"})
    except Exception as e:
        logger.warning(f"Failed to parse {path}: {e}")
    return deps


def extract_external_dependencies(repo_path: str) -> list[dict]:
    """Extract all external dependencies from a repository."""
    repo = Path(repo_path)
    all_deps: list[dict] = []

    # Python
    for req_file in ["requirements.txt", "requirements/base.txt", "requirements/prod.txt"]:
        p = repo / req_file
        if p.exists():
            all_deps.extend(_parse_requirements_txt(str(p)))

    if (repo / "pyproject.toml").exists():
        all_deps.extend(_parse_pyproject_toml(str(repo / "pyproject.toml")))

    if (repo / "setup.py").exists():
        all_deps.extend(_parse_setup_py(str(repo / "setup.py")))

    # JavaScript/TypeScript
    if (repo / "package.json").exists():
        all_deps.extend(_parse_package_json(str(repo / "package.json")))

    # Go
    if (repo / "go.mod").exists():
        all_deps.extend(_parse_go_mod(str(repo / "go.mod")))

    # Deduplicate by name
    seen: set[str] = set()
    unique: list[dict] = []
    for dep in all_deps:
        if dep["name"] not in seen:
            seen.add(dep["name"])
            unique.append(dep)

    return unique


def extract_internal_imports(repo_path: str) -> list[dict]:
    """Extract internal import relationships between Python files."""
    repo = Path(repo_path)
    imports: list[dict] = []

    # Find all Python packages (directories with __init__.py)
    packages: set[str] = set()
    py_files: list[Path] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if fname.endswith(".py"):
                py_files.append(Path(root) / fname)
                if fname == "__init__.py":
                    packages.add(Path(root).relative_to(repo).parts[0] if Path(root) != repo else "")

    # Top-level package names for matching
    top_packages = {p for p in packages if p}

    for py_file in py_files:
        try:
            with open(py_file, "r", errors="ignore") as f:
                tree = ast.parse(f.read(), filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        rel_from = str(py_file.relative_to(repo))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in top_packages:
                        imports.append({
                            "from_file": rel_from,
                            "to_file": alias.name.replace(".", "/"),
                            "imported_names": [alias.asname or alias.name.split(".")[-1]],
                        })
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    if top in top_packages:
                        names = [a.name for a in node.names]
                        imports.append({
                            "from_file": rel_from,
                            "to_file": node.module.replace(".", "/"),
                            "imported_names": names,
                        })

    return imports


def extract_dependency_info(repo_path: str) -> dict:
    """Run full dependency analysis."""
    return {
        "dependencies": extract_external_dependencies(repo_path),
        "internal_imports": extract_internal_imports(repo_path),
    }
