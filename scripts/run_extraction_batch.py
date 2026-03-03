#!/usr/bin/env python3
"""CLI: Run Repo IR extraction on a batch of repositories.

Clones repos temporarily, extracts RepoIR + diagrams + file manifest,
then deletes the clone to save disk space.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from repodesign.curation.scrape_repos import load_repo_list
from repodesign.extractors.pipeline import extract_repo_ir, save_repo_ir


def clone_repo(clone_url: str, dest: str, timeout: int = 180) -> bool:
    """Shallow-clone a repository. Returns True on success."""
    if Path(dest).exists():
        logging.info(f"  Already cloned: {dest}")
        return True
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", clone_url, dest],
            capture_output=True,
            timeout=timeout,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logging.warning(f"  Failed to clone {clone_url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch Repo IR extraction")
    parser.add_argument("repo_list", help="Path to repo_list.json")
    parser.add_argument("--output-dir", default="data/repo_irs", help="Directory for Repo IR output (per-repo subdirs)")
    parser.add_argument("--limit", type=int, default=0, help="Max repos to process (0=all)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM summaries")
    parser.add_argument("--keep-clones", action="store_true", help="Keep cloned repos (default: delete after extraction)")
    parser.add_argument("--clone-dir", default=None, help="Directory for clones (default: temp dir, deleted after)")
    parser.add_argument("--llm-provider", default="deepseek", choices=["deepseek", "anthropic", "openai"], help="LLM provider for summaries")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    entries = load_repo_list(args.repo_list)
    if args.limit > 0:
        entries = entries[: args.limit]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Use temp dir for clones unless user specified one
    use_temp = args.clone_dir is None and not args.keep_clones
    clone_base = args.clone_dir or tempfile.mkdtemp(prefix="repodesign_clones_")
    Path(clone_base).mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    try:
        for i, entry in enumerate(entries):
            print(f"\n[{i + 1}/{len(entries)}] {entry.full_name}")

            # Per-repo output directory
            repo_output_dir = os.path.join(args.output_dir, entry.name)
            repo_ir_path = os.path.join(repo_output_dir, "repo_ir.json")

            if Path(repo_ir_path).exists():
                print(f"  Already extracted, skipping")
                success += 1
                continue

            Path(repo_output_dir).mkdir(parents=True, exist_ok=True)

            # Clone
            repo_dir = os.path.join(clone_base, entry.name)
            if not clone_repo(entry.clone_url, repo_dir):
                failed += 1
                continue

            # Extract (with output_dir for diagram copying)
            try:
                repo_ir = extract_repo_ir(
                    repo_path=repo_dir,
                    repo_url=entry.url,
                    star_count=entry.star_count,
                    num_contributors=entry.num_contributors,
                    skip_llm=args.skip_llm,
                    llm_provider=args.llm_provider,
                    output_dir=repo_output_dir,
                )
                save_repo_ir(repo_ir, repo_ir_path)
                print(f"  ✓ Extracted: {len(repo_ir.file_manifest)} files, {len(repo_ir.diagram_paths)} diagrams")
                success += 1
            except Exception as e:
                logging.error(f"  Extraction failed: {e}")
                failed += 1

            # Delete clone to save space (unless keeping)
            if not args.keep_clones and Path(repo_dir).exists():
                shutil.rmtree(repo_dir, ignore_errors=True)

    finally:
        # Clean up temp clone dir
        if use_temp and Path(clone_base).exists():
            shutil.rmtree(clone_base, ignore_errors=True)

    print(f"\n{'=' * 40}")
    print(f"Batch extraction complete:")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Total:   {len(entries)}")
    print(f"  Output:  {args.output_dir}")


if __name__ == "__main__":
    main()
