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

    # Collect file paths from teacher plan (split modify vs create)
    must_exist = set()
    new_files = set()
    for t in tickets:
        must_exist.update(t.get("files_to_modify", []))
        new_files.update(t.get("files_to_create", []))

    # RGS: only files_to_modify must exist in manifest
    existing = [p for p in must_exist if p in manifest]
    rgs = len(existing) / len(must_exist) if must_exist else 1.0

    # Diff stats
    diff_lines = len(diff_text.splitlines())
    diff_kb = len(diff_text) / 1024

    print(f"\n{'='*60}")
    print(f"{d.name}")
    print(f"  Spec: {spec.get('feature_name', '?')}")
    print(f"  Source PR: {spec.get('source_pr', '?')}")
    print(f"  Architecture decisions: {len(decisions)}")
    print(f"  Tickets: {len(tickets)}")
    print(f"  files_to_modify: {len(must_exist)}, files_to_create: {len(new_files)}")
    print(f"  Grounded (in manifest): {len(existing)}/{len(must_exist)}")
    print(f"  RGS: {rgs:.2f}")
    print(f"  Diff: {diff_lines} lines, {diff_kb:.1f} KB")
    print(f"  RepoIR: {len(manifest)} files, {ir['repo_metadata'].get('primary_language','?')}")

    # Show which files_to_modify are NOT in manifest
    missing = must_exist - set(existing)
    if missing:
        print(f"  Ungrounded files_to_modify ({len(missing)}):")
        for p in sorted(missing)[:5]:
            print(f"    - {p}")
