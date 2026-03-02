"""API route extractor: detect endpoints from Flask, FastAPI, Django, Express."""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build"}


def _extract_flask_routes(file_path: str, rel_path: str) -> list[dict]:
    """Extract Flask/Blueprint routes via AST."""
    routes = []
    try:
        with open(file_path, "r", errors="ignore") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError):
        return routes

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            # @app.route("/path") or @blueprint.route("/path")
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                attr_name = decorator.func.attr
                if attr_name == "route" and decorator.args:
                    path = _get_str_value(decorator.args[0])
                    if path is None:
                        continue
                    methods = ["GET"]
                    for kw in decorator.keywords:
                        if kw.arg == "methods" and isinstance(kw.value, ast.List):
                            methods = [_get_str_value(m) for m in kw.value.elts if _get_str_value(m)]
                    for method in methods:
                        routes.append({
                            "path": path,
                            "method": method,
                            "handler_file": rel_path,
                            "handler_function": node.name,
                            "auth_required": None,
                            "framework": "flask",
                        })
                # @app.get("/path"), @app.post("/path") etc.
                elif attr_name in ("get", "post", "put", "delete", "patch") and decorator.args:
                    path = _get_str_value(decorator.args[0])
                    if path:
                        routes.append({
                            "path": path,
                            "method": attr_name.upper(),
                            "handler_file": rel_path,
                            "handler_function": node.name,
                            "auth_required": None,
                            "framework": "flask",
                        })
    return routes


def _extract_fastapi_routes(file_path: str, rel_path: str) -> list[dict]:
    """Extract FastAPI routes via AST."""
    routes = []
    try:
        with open(file_path, "r", errors="ignore") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError):
        return routes

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                attr_name = decorator.func.attr
                if attr_name in ("get", "post", "put", "delete", "patch", "options", "head"):
                    path = _get_str_value(decorator.args[0]) if decorator.args else None
                    if path:
                        routes.append({
                            "path": path,
                            "method": attr_name.upper(),
                            "handler_file": rel_path,
                            "handler_function": node.name,
                            "auth_required": None,
                            "framework": "fastapi",
                        })
    return routes


def _extract_django_routes(file_path: str, rel_path: str) -> list[dict]:
    """Extract Django URL patterns via regex parsing."""
    routes = []
    try:
        with open(file_path, "r", errors="ignore") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return routes

    # Match: path("api/users/", views.user_list, name="user-list")
    pattern = r"""(?:path|re_path)\(\s*['\"]([^'\"]+)['\"]"""
    for match in re.finditer(pattern, content):
        url_path = match.group(1)
        routes.append({
            "path": "/" + url_path.lstrip("/"),
            "method": "ANY",
            "handler_file": rel_path,
            "handler_function": None,
            "auth_required": None,
            "framework": "django",
        })

    return routes


def _extract_express_routes(file_path: str, rel_path: str) -> list[dict]:
    """Extract Express.js routes via regex."""
    routes = []
    try:
        with open(file_path, "r", errors="ignore") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return routes

    # Match: router.get("/path", ...) or app.post("/path", ...)
    pattern = r"""(?:app|router)\.(get|post|put|delete|patch|all)\(\s*['\"]([^'\"]+)['\"]"""
    for match in re.finditer(pattern, content):
        method = match.group(1).upper()
        path = match.group(2)
        routes.append({
            "path": path,
            "method": method,
            "handler_file": rel_path,
            "handler_function": None,
            "auth_required": None,
            "framework": "express",
        })

    return routes


def _get_str_value(node: ast.expr) -> str | None:
    """Extract string value from AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _detect_framework(repo_path: str) -> set[str]:
    """Detect which web frameworks are used in the repo."""
    frameworks: set[str] = set()
    repo = Path(repo_path)

    # Check Python deps
    for dep_file in ["requirements.txt", "pyproject.toml", "setup.py"]:
        p = repo / dep_file
        if p.exists():
            try:
                content = p.read_text(errors="ignore").lower()
                if "flask" in content:
                    frameworks.add("flask")
                if "fastapi" in content:
                    frameworks.add("fastapi")
                if "django" in content:
                    frameworks.add("django")
            except OSError:
                pass

    # Check JS deps
    pkg = repo / "package.json"
    if pkg.exists():
        try:
            content = pkg.read_text(errors="ignore").lower()
            if "express" in content:
                frameworks.add("express")
        except OSError:
            pass

    # Check manage.py for Django
    if (repo / "manage.py").exists():
        frameworks.add("django")

    return frameworks


def extract_api_routes(repo_path: str) -> list[dict]:
    """Extract all API routes from a repository."""
    repo = Path(repo_path)
    frameworks = _detect_framework(repo_path)
    all_routes: list[dict] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            fpath = os.path.join(root, fname)
            rel_path = str(Path(fpath).relative_to(repo))

            if fname.endswith(".py"):
                if "flask" in frameworks or "fastapi" in frameworks:
                    all_routes.extend(_extract_flask_routes(fpath, rel_path))
                    all_routes.extend(_extract_fastapi_routes(fpath, rel_path))
                if "django" in frameworks and ("urls.py" in fname or "routes" in fname):
                    all_routes.extend(_extract_django_routes(fpath, rel_path))
            elif fname.endswith((".js", ".ts")) and "express" in frameworks:
                all_routes.extend(_extract_express_routes(fpath, rel_path))

    # Deduplicate
    seen: set[tuple] = set()
    unique: list[dict] = []
    for route in all_routes:
        key = (route["path"], route["method"], route["handler_file"])
        if key not in seen:
            seen.add(key)
            unique.append(route)

    return unique
