#!/usr/bin/env python3
"""CLI: Generate training data (specs + teacher plans) from extracted RepoIRs.

Usage:
    python scripts/generate_training_data.py data/repo_irs --limit 5 -v
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from repodesign.training.data_gen import generate_training_example


def main():
    parser = argparse.ArgumentParser(description="Generate training data from RepoIRs")
    parser.add_argument("repo_irs_dir", help="Directory containing per-repo subdirs with repo_ir.json")
    parser.add_argument("--limit", type=int, default=0, help="Max repos to process (0=all)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    repo_irs_dir = Path(args.repo_irs_dir)
    if not repo_irs_dir.exists():
        print(f"Error: {repo_irs_dir} does not exist")
        sys.exit(1)

    # Find all repo_ir.json files
    repo_dirs = sorted([
        d for d in repo_irs_dir.iterdir()
        if d.is_dir() and (d / "repo_ir.json").exists()
    ])

    if args.limit > 0:
        repo_dirs = repo_dirs[:args.limit]

    print(f"Found {len(repo_dirs)} repos with extracted RepoIRs")

    success = 0
    failed = 0
    skipped = 0

    for i, repo_dir in enumerate(repo_dirs):
        repo_name = repo_dir.name
        print(f"\n[{i + 1}/{len(repo_dirs)}] {repo_name}")

        # Skip if already generated
        if (repo_dir / "spec.json").exists() and (repo_dir / "teacher_plan.json").exists():
            print(f"  Already generated, skipping")
            skipped += 1
            success += 1
            continue

        result = generate_training_example(
            repo_ir_path=str(repo_dir / "repo_ir.json"),
            output_dir=str(repo_dir),
        )

        if result:
            n_decisions = len(result["teacher_plan"].get("architecture_decisions", []))
            n_tickets = len(result["teacher_plan"].get("tickets", []))
            n_diagrams = len(result.get("diagram_paths", []))
            print(f"  ✓ Spec: {result['spec'].get('feature_name', '?')}")
            print(f"    Plan: {n_decisions} decisions, {n_tickets} tickets, {n_diagrams} diagrams")
            success += 1
        else:
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Training data generation complete:")
    print(f"  Success:  {success} ({skipped} skipped)")
    print(f"  Failed:   {failed}")
    print(f"  Total:    {len(repo_dirs)}")


if __name__ == "__main__":
    main()
