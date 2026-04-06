"""Generate Mermaid-based IR diagrams for all commit pairs.

For each pair in commit_pairs_production/:
  - api_routes  → diagrams/api_routes.png   (if non-empty after dedup)
  - data_models → diagrams/data_models.png  (if non-empty after dedup)

Updates diagram_paths in repo_ir.json to point at the generated PNGs,
replacing any previously scraped diagram paths.

Usage:
    python scripts/generate_ir_diagrams.py
    python scripts/generate_ir_diagrams.py --force          # re-render existing
    python scripts/generate_ir_diagrams.py --dry-run        # preview, no writes
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

from repodesign.diagrams.ir_to_mermaid import api_routes_to_mermaid, data_models_to_er
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
    """Render a Mermaid string to PNG via the mmdc CLI.

    Returns True on success, False on failure (logs a warning).
    """
    with tempfile.NamedTemporaryFile(
        suffix=".mmd", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(mermaid_str)
        tmp_path = Path(f.name)

    try:
        result = subprocess.run(
            [mmdc, "-i", str(tmp_path), "-o", str(output_path),
             "-b", "white", "--quiet"],
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

def _touched_files(pair_dir: Path) -> set[str] | None:
    """Return the set of normalised file paths touched by this PR's diff.

    Paths are normalised to forward slashes and lowercased so they can be
    compared against handler_file / file_path values from the RepoIR regardless
    of OS path separator.  Returns None if no diff file is found (fallback:
    caller should use all routes/models).
    """
    diff_path = pair_dir / "ground_truth_diff.txt"
    if not diff_path.exists():
        return None
    try:
        diff_text = diff_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    parsed = parse_diff_files(diff_text)
    all_files = parsed.get("modified", []) + parsed.get("created", [])
    return {p.replace("\\", "/").lower() for p in all_files}


def _filter_by_diff(items: list[dict], file_key: str, touched: set[str] | None) -> list[dict]:
    """Filter a list of route/model dicts to only those in touched files.

    If touched is None (no diff available), returns all items unchanged.
    """
    if touched is None:
        return items
    return [
        item for item in items
        if item.get(file_key, "").replace("\\", "/").lower() in touched
    ]


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

def process_pair(
    pair_dir: Path,
    mmdc: str,
    force: bool,
    dry_run: bool,
) -> dict[str, int]:
    """Process one commit pair directory.

    Returns a counter dict with keys: api, er, skipped, errors.
    """
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

    touched = _touched_files(pair_dir)

    diagrams_dir = pair_dir / "diagrams"
    new_diagram_paths: list[str] = []

    # --- api_routes → api_routes.png ---
    routes = _filter_by_diff(repo_ir.get("api_routes", []), "handler_file", touched)
    mmd_api = api_routes_to_mermaid(routes)
    if mmd_api:
        out_png = diagrams_dir / "api_routes.png"
        rel_path = "diagrams/api_routes.png"
        if out_png.exists() and not force:
            logger.debug(f"  [{pair_dir.name}] api_routes.png exists, skipping")
            new_diagram_paths.append(rel_path)
            counts["skipped"] += 1
        elif dry_run:
            logger.info(f"  [{pair_dir.name}] [dry-run] would render api_routes.png")
            new_diagram_paths.append(rel_path)
            counts["api"] += 1
        else:
            diagrams_dir.mkdir(exist_ok=True)
            ok = render_mermaid(mmd_api, out_png, mmdc)
            if ok:
                new_diagram_paths.append(rel_path)
                counts["api"] += 1
            else:
                logger.warning(f"  [{pair_dir.name}] api_routes render failed")
                counts["errors"] += 1

    # --- data_models → data_models.png ---
    models = _filter_by_diff(repo_ir.get("data_models", []), "file_path", touched)
    mmd_er = data_models_to_er(models)
    if mmd_er:
        out_png = diagrams_dir / "data_models.png"
        rel_path = "diagrams/data_models.png"
        if out_png.exists() and not force:
            logger.debug(f"  [{pair_dir.name}] data_models.png exists, skipping")
            new_diagram_paths.append(rel_path)
            counts["skipped"] += 1
        elif dry_run:
            logger.info(f"  [{pair_dir.name}] [dry-run] would render data_models.png")
            new_diagram_paths.append(rel_path)
            counts["er"] += 1
        else:
            diagrams_dir.mkdir(exist_ok=True)
            ok = render_mermaid(mmd_er, out_png, mmdc)
            if ok:
                new_diagram_paths.append(rel_path)
                counts["er"] += 1
            else:
                logger.warning(f"  [{pair_dir.name}] data_models render failed")
                counts["errors"] += 1

    # Patch diagram_paths and write back (skip in dry-run)
    if not dry_run and new_diagram_paths != repo_ir.get("diagram_paths"):
        repo_ir["diagram_paths"] = new_diagram_paths
        try:
            with open(ir_path, "w", encoding="utf-8") as f:
                json.dump(repo_ir, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"  [{pair_dir.name}] Failed to write repo_ir.json: {e}")
            counts["errors"] += 1

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to commit_pairs_production/ directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-render diagrams that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without writing anything")
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Verify mmdc is available (skip check in dry-run so you can preview without Node)
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
        logger.info("(dry-run mode — no files will be written)")

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
