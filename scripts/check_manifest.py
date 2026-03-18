"""Check if teacher files are in the repo file_manifest."""
import json
from pathlib import Path

repo_dir = Path("data/commit_pairs_production/astro_pr15779")

with open(repo_dir / "repo_ir.json", encoding="utf-8") as f:
    repo_ir = json.load(f)

with open(repo_dir / "teacher_plan.json", encoding="utf-8") as f:
    teacher_plan = json.load(f)

manifest = set(repo_ir.get("file_manifest", []))
print(f"File manifest size: {len(manifest)}")

# Get teacher files
teacher_files = set()
for ticket in teacher_plan.get("tickets", []):
    teacher_files.update(ticket.get("files_to_modify", []))
    teacher_files.update(ticket.get("files_to_create", []))
for decision in teacher_plan.get("architecture_decisions", []):
    teacher_files.update(decision.get("files_affected", []))

print(f"\nTeacher files: {len(teacher_files)}")
for f in sorted(teacher_files):
    in_manifest = f in manifest
    print(f"  {'✓' if in_manifest else '✗'} {f}")

print(f"\n{'='*80}")
print(f"Files in manifest: {sum(1 for f in teacher_files if f in manifest)}/{len(teacher_files)}")
print(f"Files NOT in manifest: {sum(1 for f in teacher_files if f not in manifest)}/{len(teacher_files)}")
