#!/usr/bin/env python3
"""CLI: Extract commit pairs from repos and generate training data from real PRs.

Usage:
    # Find squash-merged PRs, extract RepoIR at before-commit, generate teacher explanations
    python scripts/extract_commit_pairs.py pocketbase/pocketbase formbricks/formbricks loco-rs/loco \
        --pairs-per-repo 1 --output-dir data/commit_pairs -v

    # Use existing commit_pairs.json (skip PR discovery)
    python scripts/extract_commit_pairs.py --from-pairs data/commit_pairs/commit_pairs.json \
        --output-dir data/commit_pairs -v
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from repodesign.curation.commit_pairs import (
    CommitPairFinder,
    save_commit_pairs,
    load_commit_pairs,
)
from repodesign.extractors.pipeline import extract_repo_ir, save_repo_ir
from repodesign.training.data_gen_commit_pair import generate_commit_pair_example


def clone_at_commit(clone_url: str, sha: str, dest: str, timeout: int = 180) -> bool:
    """Clone a repo and checkout a specific commit."""
    try:
        # Fetch just enough history to reach the target commit
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", clone_url, dest],
            capture_output=True, timeout=timeout, check=True,
        )
        subprocess.run(
            ["git", "checkout", sha],
            capture_output=True, timeout=60, check=True, cwd=dest,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logging.warning(f"  Failed to clone/checkout {sha}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract commit pairs and generate training data")
    parser.add_argument("repos", nargs="*", help="Repo full names (e.g. pocketbase/pocketbase)")
    parser.add_argument("--from-pairs", default=None, help="Load existing commit_pairs.json instead of discovering")
    parser.add_argument("--pairs-per-repo", type=int, default=1, help="Commit pairs per repo")
    parser.add_argument("--output-dir", "-o", default="data/commit_pairs", help="Output directory")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip RepoIR extraction (use existing)")
    parser.add_argument("--skip-teacher", action="store_true", help="Skip teacher explanation generation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Discover commit pairs ───
    if args.from_pairs:
        print(f"Loading commit pairs from {args.from_pairs}...")
        all_pairs = load_commit_pairs(args.from_pairs)
    else:
        if not args.repos:
            parser.error("Provide repo names or --from-pairs")

        finder = CommitPairFinder()
        all_pairs = []
        for repo in args.repos:
            pairs = finder.extract_commit_pairs(repo, max_pairs=args.pairs_per_repo)
            all_pairs.extend(pairs)

        # Save discovered pairs
        pairs_path = os.path.join(args.output_dir, "commit_pairs.json")
        save_commit_pairs(all_pairs, pairs_path)

    print(f"\nFound {len(all_pairs)} commit pairs:")
    for p in all_pairs:
        print(f"  PR #{p.pr_number} ({p.repo_full_name}): {p.pr_title}")
        print(f"    {p.diff_stats.get('files_changed', '?')} files, "
              f"+{p.diff_stats.get('additions', '?')}/-{p.diff_stats.get('deletions', '?')}")

    if not all_pairs:
        print("No commit pairs found. Try different repos or relax filters.")
        return

    # ─── Step 2: Clone at before-commit and extract RepoIR ───
    if not args.skip_extraction:
        print(f"\n{'='*50}")
        print("Extracting RepoIR at before-commit for each pair...")

        for i, pair in enumerate(all_pairs):
            pair_data = asdict(pair) if hasattr(pair, '__dataclass_fields__') else pair
            repo_name = pair_data["repo_full_name"].split("/")[-1]
            pr_num = pair_data["pr_number"]
            before_sha = pair_data["before_sha"]
            pair_dir = os.path.join(args.output_dir, f"{repo_name}_pr{pr_num}")

            ir_path = os.path.join(pair_dir, "repo_ir.json")
            if Path(ir_path).exists():
                print(f"\n[{i+1}/{len(all_pairs)}] {repo_name} PR #{pr_num}: already extracted, skipping")
                continue

            print(f"\n[{i+1}/{len(all_pairs)}] {repo_name} PR #{pr_num}: cloning at {before_sha[:8]}...")

            clone_url = f"https://github.com/{pair_data['repo_full_name']}.git"
            with tempfile.TemporaryDirectory(prefix="cp_clone_") as tmp_dir:
                repo_dir = os.path.join(tmp_dir, repo_name)
                if not clone_at_commit(clone_url, before_sha, repo_dir):
                    print(f"  ✗ Clone failed")
                    continue

                # Extract RepoIR (with LLM summary)
                try:
                    Path(pair_dir).mkdir(parents=True, exist_ok=True)
                    repo_ir = extract_repo_ir(
                        repo_path=repo_dir,
                        repo_url=f"https://github.com/{pair_data['repo_full_name']}",
                        star_count=0,
                        num_contributors=0,
                        skip_llm=False,
                        output_dir=pair_dir,
                    )
                    save_repo_ir(repo_ir, ir_path)
                    print(f"  ✓ Extracted: {len(repo_ir.file_manifest)} files, "
                          f"{len(repo_ir.diagram_paths)} diagrams")
                except Exception as e:
                    logging.error(f"  ✗ Extraction failed: {e}")

    # ─── Step 3: Generate teacher explanations ───
    if not args.skip_teacher:
        print(f"\n{'='*50}")
        print("Generating teacher explanations from real diffs...")

        for i, pair in enumerate(all_pairs):
            pair_data = asdict(pair) if hasattr(pair, '__dataclass_fields__') else pair
            repo_name = pair_data["repo_full_name"].split("/")[-1]
            pr_num = pair_data["pr_number"]
            pair_dir = os.path.join(args.output_dir, f"{repo_name}_pr{pr_num}")

            spec_path = os.path.join(pair_dir, "spec.json")
            if Path(spec_path).exists():
                print(f"\n[{i+1}/{len(all_pairs)}] {repo_name} PR #{pr_num}: teacher data exists, skipping")
                continue

            ir_path = os.path.join(pair_dir, "repo_ir.json")
            if not Path(ir_path).exists():
                print(f"\n[{i+1}/{len(all_pairs)}] {repo_name} PR #{pr_num}: no RepoIR, skipping")
                continue

            print(f"\n[{i+1}/{len(all_pairs)}] {repo_name} PR #{pr_num}: generating teacher explanation...")

            with open(ir_path, encoding="utf-8") as f:
                repo_ir = json.load(f)

            result = generate_commit_pair_example(pair_data, repo_ir, pair_dir)
            if result:
                print(f"  ✓ Generated spec + teacher plan")
            else:
                print(f"  ✗ Generation failed")

    # ─── Summary ───
    print(f"\n{'='*50}")
    print("Commit-pair extraction complete!")
    print(f"Output: {args.output_dir}")

    # Check what was produced
    for d in sorted(Path(args.output_dir).iterdir()):
        if d.is_dir():
            files = list(d.glob("*"))
            has_ir = (d / "repo_ir.json").exists()
            has_spec = (d / "spec.json").exists()
            has_plan = (d / "teacher_plan.json").exists()
            has_diff = (d / "ground_truth_diff.txt").exists()
            status = lambda ok: "✓" if ok else "✗"
            print(f"  {d.name}: IR={status(has_ir)} Spec={status(has_spec)} "
                  f"Plan={status(has_plan)} Diff={status(has_diff)}")


if __name__ == "__main__":
    main()
