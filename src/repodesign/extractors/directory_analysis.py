"""Directory analysis extractor: tree structure, LOC counts, key directories."""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# File extensions → language mapping
LANG_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".ex": "elixir",
    ".exs": "elixir",
}

# Directories to skip during analysis
SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "vendor",
    "target",
    ".idea",
    ".vscode",
}

# Patterns for identifying key directories
KEY_DIR_PATTERNS: dict[str, list[str]] = {
    "source": ["src", "lib", "app", "pkg", "internal", "cmd"],
    "tests": ["tests", "test", "spec", "__tests__", "testing"],
    "docs": ["docs", "doc", "documentation"],
    "config": ["config", "conf", "settings", "cfg"],
    "api": ["api", "routes", "endpoints", "controllers", "views"],
    "models": ["models", "schemas", "entities"],
    "migrations": ["migrations", "migrate", "alembic"],
    "static": ["static", "public", "assets", "media"],
    "templates": ["templates", "views", "pages"],
    "infrastructure": ["infra", "deploy", "k8s", "terraform", "helm", "docker"],
    "scripts": ["scripts", "bin", "tools"],
}


def generate_directory_tree(repo_path: str, max_depth: int = 4, max_entries: int = 200) -> str:
    """Generate a truncated directory tree string."""
    repo = Path(repo_path)
    lines: list[str] = []
    entry_count = 0

    def _walk(directory: Path, prefix: str, depth: int) -> None:
        nonlocal entry_count
        if depth > max_depth or entry_count >= max_entries:
            return

        try:
            entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        dirs = [e for e in entries if e.is_dir() and e.name not in SKIP_DIRS and not e.name.startswith(".")]
        files = [e for e in entries if e.is_file() and not e.name.startswith(".")]

        items = dirs + files
        for i, entry in enumerate(items):
            if entry_count >= max_entries:
                lines.append(f"{prefix}... (truncated)")
                return
            connector = "└── " if i == len(items) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}/") if entry.is_dir() else lines.append(
                f"{prefix}{connector}{entry.name}"
            )
            entry_count += 1
            if entry.is_dir():
                extension = "    " if i == len(items) - 1 else "│   "
                _walk(entry, prefix + extension, depth + 1)

    lines.append(repo.name + "/")
    _walk(repo, "", 1)
    return "\n".join(lines)


def count_loc_by_language(repo_path: str) -> tuple[dict[str, int], int]:
    """Count lines of code per language. Returns (lang_loc_dict, total_loc)."""
    loc_counter: Counter[str] = Counter()
    total = 0

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            lang = LANG_EXTENSIONS.get(ext)
            if lang is None:
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", errors="ignore") as f:
                    line_count = sum(1 for _ in f)
                loc_counter[lang] += line_count
                total += line_count
            except (OSError, UnicodeDecodeError):
                continue

    return dict(loc_counter), total


def compute_language_breakdown(loc_by_lang: dict[str, int], total_loc: int) -> dict[str, float]:
    """Compute fractional language breakdown."""
    if total_loc == 0:
        return {}
    return {lang: round(count / total_loc, 4) for lang, count in sorted(loc_by_lang.items(), key=lambda x: -x[1])}


def identify_key_directories(repo_path: str) -> dict[str, str]:
    """Identify key directories (source, tests, docs, etc.) by matching patterns."""
    repo = Path(repo_path)
    found: dict[str, str] = {}

    # Check top-level and one level down
    for child in repo.iterdir():
        if not child.is_dir() or child.name in SKIP_DIRS or child.name.startswith("."):
            continue
        name_lower = child.name.lower()
        rel = str(child.relative_to(repo))
        for role, patterns in KEY_DIR_PATTERNS.items():
            if role not in found and name_lower in patterns:
                found[role] = rel

    return found


def extract_directory_info(repo_path: str) -> dict:
    """Run full directory analysis. Returns dict with tree, loc, breakdown, key_dirs."""
    loc_by_lang, total_loc = count_loc_by_language(repo_path)
    breakdown = compute_language_breakdown(loc_by_lang, total_loc)
    primary_language = max(loc_by_lang, key=loc_by_lang.get) if loc_by_lang else "unknown"

    return {
        "directory_tree": generate_directory_tree(repo_path),
        "total_loc": total_loc,
        "language_breakdown": breakdown,
        "primary_language": primary_language,
        "key_directories": identify_key_directories(repo_path),
    }
