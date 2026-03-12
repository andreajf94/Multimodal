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

# By language
ts_repos = ['dub', 'polar', 'documenso', 'wxt', 'slidev', 'formbricks']
rust_repos = ['axum', 'shuttle', 'loco', 'meilisearch', 'qdrant', 'tauri']
go_repos = ['pocketbase', 'sdk']

langs = defaultdict(int)
for repo, dirs in repos.items():
    if repo in ts_repos:
        langs['TypeScript'] += len(dirs)
    elif repo in rust_repos:
        langs['Rust'] += len(dirs)
    elif repo in go_repos:
        langs['Go'] += len(dirs)

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
    
    plan_paths = []
    for dec in plan.get('architecture_decisions', []):
        plan_paths.extend(dec.get('files_affected', []))
    for ticket in plan.get('tickets', []):
        plan_paths.extend(ticket.get('files_to_modify', []))
        plan_paths.extend(ticket.get('files_to_create', []))
    
    if plan_paths:
        grounded = sum(1 for p in plan_paths if p in manifest)
        rgs = grounded / len(plan_paths)
    else:
        rgs = 0.0
    
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
    
    plan_paths = []
    for dec in plan.get('architecture_decisions', []):
        plan_paths.extend(dec.get('files_affected', []))
    for ticket in plan.get('tickets', []):
        plan_paths.extend(ticket.get('files_to_modify', []))
        plan_paths.extend(ticket.get('files_to_create', []))
    
    if plan_paths:
        grounded = sum(1 for p in plan_paths if p in manifest)
        rgs = grounded / len(plan_paths)
    else:
        rgs = 0.0
    
    diff_kb = (d / "ground_truth_diff.txt").stat().st_size / 1024
    
    print(f"{d.name}")
    print(f"  RGS: {rgs:.2f}")
    print(f"  Architecture decisions: {len(plan.get('architecture_decisions', []))}")
    print(f"  Tickets: {len(plan.get('tickets', []))}")
    print(f"  Diff: {diff_kb:.1f} KB")
    print()
