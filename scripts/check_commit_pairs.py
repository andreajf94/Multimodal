#!/usr/bin/env python3
"""Check quality of commit-pair extraction results."""
import json, os
from pathlib import Path

base = Path("data/commit_pairs")

for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    ir_path = d / "repo_ir.json"
    spec_path = d / "spec.json"
    plan_path = d / "teacher_plan.json"
    diff_path = d / "ground_truth_diff.txt"

    if not all(p.exists() for p in [ir_path, spec_path, plan_path]):
        print(f"\n{d.name}: INCOMPLETE")
        continue

    ir = json.load(open(ir_path, encoding="utf-8"))
    spec = json.load(open(spec_path, encoding="utf-8"))
    plan = json.load(open(plan_path, encoding="utf-8"))
    diff_text = open(diff_path, encoding="utf-8").read() if diff_path.exists() else ""

    manifest = set(ir.get("file_manifest", []))
    decisions = plan.get("architecture_decisions", [])
    tickets = plan.get("tickets", [])

    # Collect all file paths from teacher plan
    plan_paths = set()
    for dec in decisions:
        plan_paths.update(dec.get("files_affected", []))
    for t in tickets:
        plan_paths.update(t.get("files_to_modify", []))
        plan_paths.update(t.get("files_to_create", []))

    # RGS: fraction of plan paths that exist in repo manifest
    existing = [p for p in plan_paths if p in manifest]
    rgs = len(existing) / len(plan_paths) if plan_paths else 0

    # Diff stats
    diff_lines = len(diff_text.splitlines())
    diff_kb = len(diff_text) / 1024

    print(f"\n{'='*60}")
    print(f"{d.name}")
    print(f"  Spec: {spec.get('feature_name', '?')}")
    print(f"  Source PR: {spec.get('source_pr', '?')}")
    print(f"  Architecture decisions: {len(decisions)}")
    print(f"  Tickets: {len(tickets)}")
    print(f"  File paths in plan: {len(plan_paths)}")
    print(f"  Paths in manifest: {len(existing)}/{len(plan_paths)}")
    print(f"  RGS: {rgs:.2f}")
    print(f"  Diff: {diff_lines} lines, {diff_kb:.1f} KB")
    print(f"  RepoIR: {len(manifest)} files, {ir['repo_metadata'].get('primary_language','?')}")

    # Show which paths are grounded vs not
    missing = plan_paths - set(existing)
    if missing:
        print(f"  Ungrounded paths ({len(missing)}):")
        for p in sorted(missing)[:5]:
            print(f"    - {p}")
