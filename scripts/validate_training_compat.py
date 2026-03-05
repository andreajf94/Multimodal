#!/usr/bin/env python3
"""Validate that commit-pair data is compatible with the training pipeline."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repodesign.training.data_gen import summarize_repo_ir_for_prompt
from repodesign.training.reward import format_compliance, format_partial, rgs_score

base = Path("data/commit_pairs")
errors = []

for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    ir_path = d / "repo_ir.json"
    spec_path = d / "spec.json"
    plan_path = d / "teacher_plan.json"
    if not all(p.exists() for p in [ir_path, spec_path, plan_path]):
        continue

    with open(ir_path, encoding="utf-8") as f:
        repo_ir = json.load(f)
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    with open(plan_path, encoding="utf-8") as f:
        teacher_plan = json.load(f)

    print(f"\n{'='*60}")
    print(f"{d.name}")

    # 1. Check summarizer works
    summary = summarize_repo_ir_for_prompt(repo_ir)
    print(f"  RepoIR summary: {len(summary)} chars ({'OK' if len(summary) > 100 else 'TOO SHORT'})")

    # 2. Check spec has required fields for build_prompt
    for key in ["feature_name", "description", "functional_requirements"]:
        if key not in spec:
            errors.append(f"{d.name}: spec missing '{key}'")
            print(f"  ✗ spec missing '{key}'")
        else:
            print(f"  ✓ spec.{key}")

    # 3. Check teacher_plan as if it were model output (format_compliance)
    plan_json = json.dumps(teacher_plan)
    fc = format_compliance([plan_json])
    fp = format_partial([plan_json])
    manifest = repo_ir.get("file_manifest", [])
    rgs = rgs_score([plan_json], [manifest])
    print(f"  format_compliance: {fc[0]}")
    print(f"  format_partial:    {fp[0]:.3f}")
    print(f"  rgs_score:         {rgs[0]:.3f}")

    if fc[0] == 0.0:
        errors.append(f"{d.name}: teacher_plan fails format_compliance")
        # Debug: check what's missing
        plan_keys = set(teacher_plan.keys())
        core = {"architecture_decisions", "tickets"}
        opt_a = {"technology_choices"}
        opt_b = {"implementation_summary"}
        print(f"    plan keys: {plan_keys}")
        print(f"    has core: {core.issubset(plan_keys)}")
        print(f"    has opt_a: {bool(opt_a & plan_keys)}")
        print(f"    has opt_b: {bool(opt_b & plan_keys)}")

    # 4. Check file_manifest exists
    print(f"  file_manifest: {len(manifest)} files")
    if len(manifest) == 0:
        errors.append(f"{d.name}: empty file_manifest")

print(f"\n{'='*60}")
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — commit-pair data is training-compatible")
