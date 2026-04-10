"""Generate Mermaid-based IR diagrams for all commit pairs.

For each pair in commit_pairs_production/:
  - api_routes  → diagrams/api_routes_N.png  (filtered by diff, chunked)
  - data_models → diagrams/data_models_N.png (filtered by diff mention, chunked)

Updates diagram_paths in repo_ir.json to point at the generated PNGs.

Usage:
    python scripts/generate_ir_diagrams.py
    python scripts/generate_ir_diagrams.py --force
    python scripts/generate_ir_diagrams.py --dry-run
    python scripts/generate_ir_diagrams.py --data-dir path/to/pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from repodesign.diagrams.ir_to_mermaid import (
    api_routes_to_mermaid,
    data_models_to_er,
    models_mentioned_in_diff,
)
from repodesign.training.reward import parse_diff_files

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "commit_pairs_production"


# ---------------------------------------------------------------------------
# mmdc rendering
# ---------------------------------------------------------------------------

def _find_mmdc() -> str:
    """Return the mmdc executable path, or raise if not found."""
    cmd = shutil.which("mmdc") or shutil.which("mmdc.cmd")
    if cmd is None:
        raise RuntimeError(
            "mmdc not found on PATH.\n"
            "Install it with:  npm install -g @mermaid-js/mermaid-cli\n"
            "Then re-run this script."
        )
    return cmd


def render_mermaid(mermaid_str: str, output_path: Path, mmdc: str) -> bool:
    """Render a Mermaid string to PNG via the mmdc CLI."""
    with tempfile.NamedTemporaryFile(
        suffix=".mmd", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(mermaid_str)
        tmp_path = Path(f.name)

    try:
        result = subprocess.run(
            [mmdc, "-i", str(tmp_path), "-o", str(output_path),
             "-b", "white", "-w", "1600", "--quiet"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning(f"    mmdc error: {result.stderr.strip()[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning("    mmdc timed out after 60s")
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Diff-based filtering
# ---------------------------------------------------------------------------

def _read_diff(pair_dir: Path) -> str | None:
    """Read the ground truth diff for a commit pair. Returns None if missing."""
    diff_path = pair_dir / "ground_truth_diff.txt"
    if not diff_path.exists():
        return None
    try:
        return diff_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def _touched_files(diff_text: str) -> set[str]:
    """Extract normalised file paths touched by a diff."""
    parsed = parse_diff_files(diff_text)
    all_files = parsed.get("modified", []) + parsed.get("created", [])
    return {p.replace("\\", "/").lower() for p in all_files}


def _filter_routes_by_diff(routes: list[dict], touched: set[str]) -> list[dict]:
    """Filter routes to those whose handler_file is in the diff."""
    return [
        r for r in routes
        if r.get("handler_file", "").replace("\\", "/").lower() in touched
    ]


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

MAX_DIAGRAMS = 5  # load_diagram_images default cap


def process_pair(
    pair_dir: Path,
    mmdc: str,
    force: bool,
    dry_run: bool,
) -> dict[str, int]:
    """Process one commit pair directory."""
    counts = {"api": 0, "er": 0, "skipped": 0, "errors": 0}

    ir_path = pair_dir / "repo_ir.json"
    if not ir_path.exists():
        return counts

    try:
        with open(ir_path, encoding="utf-8") as f:
            repo_ir = json.load(f)
    except Exception as e:
        logger.warning(f"  [{pair_dir.name}] Failed to read repo_ir.json: {e}")
        counts["errors"] += 1
        return counts

    diff_text = _read_diff(pair_dir)
    touched = _touched_files(diff_text) if diff_text else set()

    diagrams_dir = pair_dir / "diagrams"
    new_diagram_paths: list[str] = []
    total_diagrams = 0

    # --- API routes: filter by handler file in diff ---
    all_routes = repo_ir.get("api_routes", [])
    if touched:
        routes = _filter_routes_by_diff(all_routes, touched)
    else:
        routes = all_routes
    mmd_charts = api_routes_to_mermaid(routes)

    for i, mmd in enumerate(mmd_charts):
        if total_diagrams >= MAX_DIAGRAMS:
            break
        fname = f"api_routes_{i}.png"
        out_png = diagrams_dir / fname
        rel_path = f"diagrams/{fname}"

        if out_png.exists() and not force:
            new_diagram_paths.append(rel_path)
            counts["skipped"] += 1
            total_diagrams += 1
        elif dry_run:
            logger.info(f"  [{pair_dir.name}] [dry-run] would render {fname}")
            new_diagram_paths.append(rel_path)
            counts["api"] += 1
            total_diagrams += 1
        else:
            diagrams_dir.mkdir(exist_ok=True)
            ok = render_mermaid(mmd, out_png, mmdc)
            if ok:
                new_diagram_paths.append(rel_path)
                counts["api"] += 1
                total_diagrams += 1
            else:
                logger.warning(f"  [{pair_dir.name}] {fname} render failed")
                counts["errors"] += 1

    # --- Data models: filter by name mention in diff ---
    all_models = repo_ir.get("data_models", [])
    if diff_text:
        models = models_mentioned_in_diff(all_models, diff_text)
    else:
        models = all_models
    # Fallback: if diff filtering gives nothing but models exist, use top models
    if not models and all_models:
        by_name: dict[str, dict] = {}
        for m in all_models:
            name = m.get("name", "").strip()
            if not name:
                continue
            existing = by_name.get(name)
            if existing is None or len(m.get("fields", [])) > len(existing.get("fields", [])):
                by_name[name] = m
        models = [m for m in by_name.values() if m.get("fields")]
        models.sort(key=lambda m: len(m.get("fields", [])), reverse=True)
        models = models[:_MAX_FALLBACK_MODELS]

    mmd_charts = data_models_to_er(models)

    for i, mmd in enumerate(mmd_charts):
        if total_diagrams >= MAX_DIAGRAMS:
            break
        fname = f"data_models_{i}.png"
        out_png = diagrams_dir / fname
        rel_path = f"diagrams/{fname}"

        if out_png.exists() and not force:
            new_diagram_paths.append(rel_path)
            counts["skipped"] += 1
            total_diagrams += 1
        elif dry_run:
            logger.info(f"  [{pair_dir.name}] [dry-run] would render {fname}")
            new_diagram_paths.append(rel_path)
            counts["er"] += 1
            total_diagrams += 1
        else:
            diagrams_dir.mkdir(exist_ok=True)
            ok = render_mermaid(mmd, out_png, mmdc)
            if ok:
                new_diagram_paths.append(rel_path)
                counts["er"] += 1
                total_diagrams += 1
            else:
                logger.warning(f"  [{pair_dir.name}] {fname} render failed")
                counts["errors"] += 1

    # Patch diagram_paths and write back
    if not dry_run and new_diagram_paths != repo_ir.get("diagram_paths"):
        repo_ir["diagram_paths"] = new_diagram_paths
        try:
            with open(ir_path, "w", encoding="utf-8") as f:
                json.dump(repo_ir, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"  [{pair_dir.name}] Failed to write repo_ir.json: {e}")
            counts["errors"] += 1

    return counts


# Fallback: if no models match the diff, show top N from the full schema
_MAX_FALLBACK_MODELS = 16  # 2 charts × 8 models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to commit_pairs directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-render diagrams that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without writing anything")
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    mmdc = ""
    if not args.dry_run:
        try:
            mmdc = _find_mmdc()
            logger.info(f"Using mmdc: {mmdc}")
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)

    pair_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    logger.info(f"Processing {len(pair_dirs)} commit pairs in {args.data_dir}")
    if args.dry_run:
        logger.info("(dry-run mode)")

    totals = {"api": 0, "er": 0, "skipped": 0, "errors": 0}

    for pair_dir in pair_dirs:
        counts = process_pair(pair_dir, mmdc, args.force, args.dry_run)
        for k, v in counts.items():
            totals[k] += v

    print()
    print("=" * 50)
    print(f"  Pairs processed : {len(pair_dirs)}")
    print(f"  API diagrams    : {totals['api']}")
    print(f"  ER diagrams     : {totals['er']}")
    print(f"  Already existed : {totals['skipped']}")
    print(f"  Errors          : {totals['errors']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
