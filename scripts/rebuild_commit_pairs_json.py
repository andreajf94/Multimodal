#!/usr/bin/env python3
"""Rebuild commit_pairs.json from the directories on disk.

Reads each directory's teacher_plan.json for PR metadata (source_pr, before_sha,
after_sha, diff_stats) and repo_ir.json for repo info.

Usage:
    python scripts/rebuild_commit_pairs_json.py [--data-dir data/commit_pairs_production]
"""

import json
import re
from pathlib import Path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/commit_pairs_production")
    args = parser.parse_args()

    base = Path(args.data_dir)
    dirs = sorted([d for d in base.iterdir() if d.is_dir()])

    entries = []
    for d in dirs:
        plan_path = d / "teacher_plan.json"
        ir_path = d / "repo_ir.json"
        spec_path = d / "spec.json"
        diff_path = d / "ground_truth_diff.txt"

        if not all(p.exists() for p in [plan_path, ir_path, spec_path]):
            continue
        if spec_path.stat().st_size == 0:
            continue

        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            ir = json.loads(ir_path.read_text(encoding="utf-8"))
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            continue

        meta = ir.get("repo_metadata", {})
        pr_url = plan.get("source_pr", spec.get("source_pr", ""))

        # Extract repo_full_name and pr_number from directory name or PR URL
        repo_key = d.name.split("_pr")[0]
        pr_match = re.search(r"_pr(\d+)$", d.name)
        pr_number = int(pr_match.group(1)) if pr_match else 0

        # Try to get full name from URL
        url_match = re.search(r"github\.com/([^/]+/[^/]+)/pull/", pr_url)
        repo_full_name = url_match.group(1) if url_match else meta.get("url", "").replace("https://github.com/", "")

        # Collect diff file list from plan tickets
        diff_files = set()
        for ticket in plan.get("tickets", []):
            diff_files.update(ticket.get("files_to_modify", []))
            diff_files.update(ticket.get("files_to_create", []))

        entry = {
            "repo_full_name": repo_full_name,
            "pr_number": pr_number,
            "pr_title": spec.get("feature_name", ""),
            "pr_url": pr_url,
            "before_sha": plan.get("before_sha", ""),
            "after_sha": plan.get("after_sha", ""),
            "diff_stats": plan.get("diff_stats", {}),
            "diff_files": sorted(diff_files),
            "primary_language": meta.get("primary_language", "unknown"),
            "star_count": meta.get("star_count", 0),
            "scale_tier": meta.get("scale_tier", ""),
        }
        entries.append(entry)

    # Write
    out_path = base / "commit_pairs.json"
    out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {out_path}")
    print(f"  (skipped {len(dirs) - len(entries)} incomplete directories)")


if __name__ == "__main__":
    main()
