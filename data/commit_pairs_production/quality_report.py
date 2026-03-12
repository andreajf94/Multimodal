#!/usr/bin/env python3
"""Generate quality report for commit-pair extraction."""
import json
from pathlib import Path
from collections import defaultdict

data_dir = Path(__file__).parent
complete = [d for d in data_dir.iterdir() if d.is_dir() and (d / "teacher_plan.json").exists()]

print("="*60)
print("COMMIT-PAIR EXTRACTION QUALITY REPORT")
print("="*60)
print()

# By repo
repos = defaultdict(list)
for d in complete:
    repo_name = d.name.split('_pr')[0]
    repos[repo_name].append(d)

print(f"Total complete examples: {len(complete)}")
print(f"\nBy repo:")
for repo, dirs in sorted(repos.items()):
    print(f"  {repo}: {len(dirs)} PRs")

# By language (read from repo_ir metadata, not hardcoded)
langs = defaultdict(int)
for repo, dirs in repos.items():
    # Read language from the first repo_ir in this group
    sample_ir_path = dirs[0] / "repo_ir.json"
    if sample_ir_path.exists():
        sample_ir = json.loads(sample_ir_path.read_text(encoding='utf-8'))
        lang = sample_ir.get("repo_metadata", {}).get("primary_language", "unknown")
        langs[lang] += len(dirs)
    else:
        langs["unknown"] += len(dirs)

print(f"\nBy language:")
for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
    print(f"  {lang}: {count}")

# RGS computation
print(f"\n{'='*60}")
print("REPO GROUNDING SCORE (RGS) ANALYSIS")
print("="*60)
print()

rgs_scores = []
for d in sorted(complete):
    plan = json.loads((d / "teacher_plan.json").read_text(encoding='utf-8'))
    repo_ir = json.loads((d / "repo_ir.json").read_text(encoding='utf-8'))
    manifest = set(repo_ir['file_manifest'])
    
    # RGS: only files_to_modify must exist; files_to_create are new
    must_exist = set()
    for ticket in plan.get('tickets', []):
        must_exist.update(ticket.get('files_to_modify', []))
    
    if must_exist:
        grounded = sum(1 for p in must_exist if p in manifest)
        rgs = grounded / len(must_exist)
    else:
        rgs = 1.0  # no files_to_modify = vacuously grounded
    
    rgs_scores.append(rgs)

print(f"Average RGS: {sum(rgs_scores)/len(rgs_scores):.3f}")
print(f"Min RGS: {min(rgs_scores):.3f}")
print(f"Max RGS: {max(rgs_scores):.3f}")
print(f"Median RGS: {sorted(rgs_scores)[len(rgs_scores)//2]:.3f}")
print()
print(f"RGS >= 0.90: {sum(1 for r in rgs_scores if r >= 0.90)}/{len(rgs_scores)} ({100*sum(1 for r in rgs_scores if r >= 0.90)/len(rgs_scores):.1f}%)")
print(f"RGS >= 0.95: {sum(1 for r in rgs_scores if r >= 0.95)}/{len(rgs_scores)} ({100*sum(1 for r in rgs_scores if r >= 0.95)/len(rgs_scores):.1f}%)")
print(f"RGS = 1.00: {sum(1 for r in rgs_scores if r == 1.0)}/{len(rgs_scores)} ({100*sum(1 for r in rgs_scores if r == 1.0)/len(rgs_scores):.1f}%)")

print(f"\n{'='*60}")
print("Sample examples (showing RGS, arch decisions, tickets, diff size):")
print("="*60)
print()

for d in sorted(complete)[:5]:
    plan = json.loads((d / "teacher_plan.json").read_text(encoding='utf-8'))
    repo_ir = json.loads((d / "repo_ir.json").read_text(encoding='utf-8'))
    manifest = set(repo_ir['file_manifest'])
    
    # RGS: only files_to_modify must exist
    must_exist = set()
    new_files = set()
    for ticket in plan.get('tickets', []):
        must_exist.update(ticket.get('files_to_modify', []))
        new_files.update(ticket.get('files_to_create', []))
    
    if must_exist:
        grounded = sum(1 for p in must_exist if p in manifest)
        rgs = grounded / len(must_exist)
    else:
        rgs = 1.0
    
    diff_kb = (d / "ground_truth_diff.txt").stat().st_size / 1024
    
    print(f"{d.name}")
    print(f"  RGS: {rgs:.2f}")
    print(f"  Architecture decisions: {len(plan.get('architecture_decisions', []))}")
    print(f"  Tickets: {len(plan.get('tickets', []))}")
    print(f"  files_to_modify: {len(must_exist)}, files_to_create: {len(new_files)}")
    print(f"  Diff: {diff_kb:.1f} KB")
    print()
