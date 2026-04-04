#!/usr/bin/env python3
"""Augment teacher plans to mention created files in implementation_summary.

For examples where ground truth has created files but the teacher's
implementation_summary doesn't mention them, append a sentence listing them.
This gives the semantic similarity reward a direct training signal for
file creation, without changing the substance of the plan.

Usage:
    python scripts/augment_teacher_plans.py data/commit_pairs_production [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repodesign.training.reward import parse_diff_files


def needs_augmentation(summary: str, created_files: list[str]) -> list[str]:
    """Return created files not mentioned in the summary (by filename or full path)."""
    unmentioned = []
    for f in created_files:
        filename = Path(f).name
        if filename not in summary and f not in summary:
            unmentioned.append(f)
    return unmentioned


def augment_summary(summary: str, unmentioned: list[str]) -> str:
    """Append a sentence listing unmentioned created files."""
    file_list = ", ".join(f"`{f}`" for f in unmentioned)
    suffix = f" New files introduced by this change: {file_list}."
    return summary.rstrip() + suffix


def main():
    parser = argparse.ArgumentParser(description="Augment teacher plans with created file mentions")
    parser.add_argument("data_dir", help="Directory with per-repo training data")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    base = Path(args.data_dir)
    total = skipped = augmented = already_mentioned = no_created = 0

    for repo_dir in sorted(base.iterdir()):
        if not repo_dir.is_dir():
            continue

        plan_path = repo_dir / "teacher_plan.json"
        diff_path = repo_dir / "ground_truth_diff.txt"

        if not plan_path.exists() or not diff_path.exists():
            skipped += 1
            continue

        total += 1
        diff_files = parse_diff_files(diff_path.read_text(encoding="utf-8"))
        created = diff_files.get("created", [])

        if not created:
            no_created += 1
            continue

        with open(plan_path, encoding="utf-8") as f:
            plan = json.load(f)

        summary = plan.get("implementation_summary", "")
        if not isinstance(summary, str) or not summary:
            skipped += 1
            continue

        unmentioned = needs_augmentation(summary, created)

        if not unmentioned:
            already_mentioned += 1
            continue

        new_summary = augment_summary(summary, unmentioned)
        augmented += 1

        print(f"[{'DRY' if args.dry_run else 'AUG'}] {repo_dir.name}")
        print(f"  Created files: {created}")
        print(f"  Unmentioned:   {unmentioned}")
        print(f"  Appended: '...{new_summary[-120:]}'")
        print()

        if not args.dry_run:
            plan["implementation_summary"] = new_summary
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Total repos:          {total}")
    print(f"No created files:     {no_created}")
    print(f"Already mentioned:    {already_mentioned}")
    print(f"Augmented:            {augmented}")
    print(f"Skipped (missing):    {skipped}")
    if args.dry_run:
        print("\nDRY RUN — no files written.")


if __name__ == "__main__":
    main()
