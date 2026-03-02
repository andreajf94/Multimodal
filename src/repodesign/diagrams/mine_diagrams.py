"""Diagram mining: find architecture diagrams in repositories."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build", "vendor"}

# Diagram file extensions
DIAGRAM_EXTENSIONS = {
    ".puml": "plantuml",
    ".plantuml": "plantuml",
    ".pu": "plantuml",
    ".mmd": "mermaid",
    ".mermaid": "mermaid",
}

# Image extensions that might be architecture diagrams
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}

# Keywords indicating architecture-related content
ARCHITECTURE_KEYWORDS = {
    "architecture",
    "system-design",
    "system_design",
    "diagram",
    "infrastructure",
    "topology",
    "deployment",
    "overview",
    "high-level",
    "design",
    "flow",
    "pipeline",
    "service",
    "microservice",
}


@dataclass
class DiagramEntry:
    """A discovered diagram file."""

    repo_name: str
    repo_path: str
    file_path: str  # Relative to repo root
    diagram_type: str  # plantuml, mermaid, image, markdown_embedded
    file_extension: str
    size_bytes: int = 0
    contains_mermaid: bool = False  # For markdown files with embedded mermaid
    contains_plantuml: bool = False


def mine_diagrams_from_repo(repo_path: str) -> list[DiagramEntry]:
    """Find all architecture diagram files in a repository.

    Searches for:
    - PlantUML files (.puml, .plantuml)
    - Mermaid files (.mmd, .mermaid)
    - Markdown files with embedded diagrams (architecture.md, etc.)
    - Architecture-related images in docs/
    """
    repo = Path(repo_path)
    repo_name = repo.name
    entries: list[DiagramEntry] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            fpath = Path(root) / fname
            rel_path = str(fpath.relative_to(repo))
            ext = fpath.suffix.lower()
            fname_lower = fname.lower()

            # Direct diagram files
            if ext in DIAGRAM_EXTENSIONS:
                entries.append(DiagramEntry(
                    repo_name=repo_name,
                    repo_path=str(repo),
                    file_path=rel_path,
                    diagram_type=DIAGRAM_EXTENSIONS[ext],
                    file_extension=ext,
                    size_bytes=fpath.stat().st_size,
                ))
                continue

            # Markdown files with architecture-related names
            if ext == ".md":
                name_no_ext = fpath.stem.lower()
                if any(kw in name_no_ext for kw in ARCHITECTURE_KEYWORDS):
                    has_mermaid, has_plantuml = _check_embedded_diagrams(str(fpath))
                    if has_mermaid or has_plantuml:
                        entries.append(DiagramEntry(
                            repo_name=repo_name,
                            repo_path=str(repo),
                            file_path=rel_path,
                            diagram_type="markdown_embedded",
                            file_extension=ext,
                            size_bytes=fpath.stat().st_size,
                            contains_mermaid=has_mermaid,
                            contains_plantuml=has_plantuml,
                        ))
                    continue

            # Architecture-related images (only in docs-like directories)
            if ext in IMAGE_EXTENSIONS:
                rel_parts = rel_path.lower()
                if any(kw in rel_parts for kw in ARCHITECTURE_KEYWORDS):
                    entries.append(DiagramEntry(
                        repo_name=repo_name,
                        repo_path=str(repo),
                        file_path=rel_path,
                        diagram_type="image",
                        file_extension=ext,
                        size_bytes=fpath.stat().st_size,
                    ))

    return entries


def _check_embedded_diagrams(md_path: str) -> tuple[bool, bool]:
    """Check if a markdown file contains embedded Mermaid or PlantUML diagrams."""
    has_mermaid = False
    has_plantuml = False
    try:
        with open(md_path, "r", errors="ignore") as f:
            content = f.read()
        if "```mermaid" in content:
            has_mermaid = True
        if "```plantuml" in content or "@startuml" in content:
            has_plantuml = True
    except (OSError, UnicodeDecodeError):
        pass
    return has_mermaid, has_plantuml


def mine_diagrams_batch(repo_paths: list[str]) -> list[DiagramEntry]:
    """Mine diagrams from multiple repositories."""
    all_entries: list[DiagramEntry] = []
    for repo_path in repo_paths:
        logger.info(f"Mining diagrams from {repo_path}...")
        entries = mine_diagrams_from_repo(repo_path)
        all_entries.extend(entries)
        if entries:
            logger.info(f"  Found {len(entries)} diagrams")
    logger.info(f"Total diagrams found: {len(all_entries)}")
    return all_entries


def save_diagram_manifest(entries: list[DiagramEntry], output_path: str) -> None:
    """Save diagram manifest to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)
    logger.info(f"Saved diagram manifest with {len(entries)} entries to {output_path}")


def load_diagram_manifest(input_path: str) -> list[DiagramEntry]:
    """Load diagram manifest from JSON."""
    with open(input_path) as f:
        data = json.load(f)
    return [DiagramEntry(**d) for d in data]
