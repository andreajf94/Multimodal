#!/usr/bin/env python3
"""Quick check of extraction quality across test repos."""
import json, os

base = "data/repo_irs_test"
for r in sorted(os.listdir(base)):
    path = os.path.join(base, r, "repo_ir.json")
    if not os.path.exists(path):
        continue
    ir = json.load(open(path, encoding="utf-8"))
    meta = ir.get("repo_metadata", {})
    summary = ir.get("architectural_summary", "")
    has_summary = not summary.startswith("(LLM summary")
    print(f"\n{'='*60}")
    print(f"{r} ({meta.get('primary_language','?')})")
    print(f"  LOC: {meta.get('total_loc',0):,}")
    print(f"  Files: {len(ir.get('file_manifest',[]))}")
    print(f"  Dependencies: {len(ir.get('dependencies',[]))}")
    print(f"  API routes: {len(ir.get('api_routes',[]))}")
    print(f"  Data models: {len(ir.get('data_models',[]))}")
    print(f"  Diagrams: {len(ir.get('diagram_paths',[]))}")
    print(f"  LLM Summary: {'YES' if has_summary else 'NO'}")
    if has_summary:
        print(f"  Summary preview: {summary[:200]}...")
    warnings = ir.get("extraction_warnings", [])
    if warnings:
        print(f"  Warnings: {warnings}")
