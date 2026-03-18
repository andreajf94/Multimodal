"""Verify that ground_truth_diff files align with teacher plan files."""
import json
import re
from pathlib import Path

def extract_files_from_diff(diff_text: str) -> dict[str, set[str]]:
    """Extract modified and created files from git diff."""
    modified = set()
    created = set()
    
    # Parse git diff headers: "diff --git a/path b/path"
    for line in diff_text.split('\n'):
        if line.startswith('diff --git'):
            # Extract path: "diff --git a/path b/path"
            match = re.search(r'a/(.*?) b/', line)
            if match:
                filepath = match.group(1)
                
                # Check if it's a new file
                lines_after = diff_text[diff_text.find(line):].split('\n')[:5]
                if 'new file mode' in '\n'.join(lines_after):
                    created.add(filepath)
                else:
                    modified.add(filepath)
    
    return {"modified": modified, "created": created}

# Check all commit pairs
data_dir = Path("data/commit_pairs_production")
results = []

for repo_dir in sorted(data_dir.iterdir()):
    if not repo_dir.is_dir():
        continue
    
    diff_path = repo_dir / "ground_truth_diff.txt"
    teacher_path = repo_dir / "teacher_plan.json"
    
    if not (diff_path.exists() and teacher_path.exists()):
        continue
    
    with open(diff_path, encoding="utf-8") as f:
        diff_text = f.read()
    
    with open(teacher_path, encoding="utf-8") as f:
        teacher = json.load(f)
    
    # Extract from diff
    diff_files = extract_files_from_diff(diff_text)
    all_diff_files = diff_files["modified"] | diff_files["created"]
    
    # Extract from teacher
    teacher_files = set()
    for ticket in teacher.get("tickets", []):
        teacher_files.update(ticket.get("files_to_modify", []))
        teacher_files.update(ticket.get("files_to_create", []))
    for decision in teacher.get("architecture_decisions", []):
        teacher_files.update(decision.get("files_affected", []))
    
    # Compare
    match = all_diff_files == teacher_files
    results.append({
        "repo": repo_dir.name,
        "diff_files": sorted(all_diff_files),
        "teacher_files": sorted(teacher_files),
        "match": match,
        "diff_only": sorted(all_diff_files - teacher_files),
        "teacher_only": sorted(teacher_files - all_diff_files),
    })

# Print results
print("="*80)
print("DIFF vs TEACHER FILE ALIGNMENT")
print("="*80)

for r in results:
    status = "✓" if r["match"] else "✗"
    print(f"\n{status} {r['repo']}")
    print(f"  Diff files: {len(r['diff_files'])}")
    print(f"  Teacher files: {len(r['teacher_files'])}")
    
    if not r["match"]:
        if r["diff_only"]:
            print(f"  Only in diff: {r['diff_only']}")
        if r["teacher_only"]:
            print(f"  Only in teacher: {r['teacher_only']}")

total = len(results)
matched = sum(1 for r in results if r["match"])
print(f"\n{'='*80}")
print(f"SUMMARY: {matched}/{total} repos have perfect alignment")
print(f"{'='*80}")
