#!/usr/bin/env python3
"""CLI: Run Repo IR extraction on a batch of repositories."""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repodesign.curation.scrape_repos import load_repo_list
from repodesign.extractors.pipeline import extract_repo_ir, save_repo_ir


def clone_repo(clone_url: str, dest: str) -> bool:
    """Clone a repository. Returns True on success."""
    if Path(dest).exists():
        logging.info(f"  Already cloned: {dest}")
        return True
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, dest],
            capture_output=True,
            timeout=120,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logging.warning(f"  Failed to clone {clone_url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch Repo IR extraction")
    parser.add_argument("repo_list", help="Path to repo_list.json")
    parser.add_argument("--repos-dir", default="data/repos", help="Directory to clone repos into")
    parser.add_argument("--output-dir", default="data/repo_irs", help="Directory for Repo IR JSONs")
    parser.add_argument("--limit", type=int, default=0, help="Max repos to process (0=all)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM summaries")
    parser.add_argument("--skip-clone", action="store_true", help="Skip cloning (use existing)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    entries = load_repo_list(args.repo_list)
    if args.limit > 0:
        entries = entries[: args.limit]

    Path(args.repos_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for i, entry in enumerate(entries):
        print(f"\n[{i + 1}/{len(entries)}] {entry.full_name}")

        # Check if already extracted
        output_path = os.path.join(args.output_dir, f"{entry.name}.json")
        if Path(output_path).exists():
            print(f"  Already extracted, skipping")
            success += 1
            continue

        # Clone
        repo_dir = os.path.join(args.repos_dir, entry.name)
        if not args.skip_clone:
            if not clone_repo(entry.clone_url, repo_dir):
                failed += 1
                continue

        if not Path(repo_dir).exists():
            print(f"  Repo not found at {repo_dir}, skipping")
            failed += 1
            continue

        # Extract
        try:
            repo_ir = extract_repo_ir(
                repo_path=repo_dir,
                repo_url=entry.url,
                star_count=entry.star_count,
                num_contributors=entry.num_contributors,
                skip_llm=args.skip_llm,
            )
            save_repo_ir(repo_ir, output_path)
            success += 1
        except Exception as e:
            logging.error(f"  Extraction failed: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Batch extraction complete:")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(entries)}")


if __name__ == "__main__":
    main()
