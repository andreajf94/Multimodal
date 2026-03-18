import json
import random

# Load step 2 data
with open('output/grpo_235b_constrained/completions.jsonl') as f:
    lines = list(f)
    step2 = json.loads(lines[2])

gt_modified = step2['diff_files']['modified']
gt_created = step2['diff_files']['created']

print(f"Step 2: {step2['repo']}")
print(f"\nGround truth files that should have been in candidate list:")
for f in gt_modified:
    print(f"  - {f}")

# Load the repo data to check what would have been in the candidate list
repo_name = step2['repo']
repo_path = f"data/commit_pairs_production/{repo_name}"

import os
if os.path.exists(f"{repo_path}/repo_ir.json"):
    with open(f"{repo_path}/repo_ir.json") as f:
        repo_ir = json.load(f)
    
    file_manifest = repo_ir.get('file_manifest', [])
    
    # Simulate candidate list generation
    actual_files = set(gt_modified) | set(gt_created)
    available_distractors = [f for f in file_manifest if f not in actual_files]
    num_distractors = min(50, len(available_distractors))
    
    # Note: random.sample will give different results each time
    # Just check if the GT files are in the manifest at all
    print(f"\nFile manifest has {len(file_manifest)} total files")
    
    print("\nChecking if GT files exist in manifest:")
    for gt_file in gt_modified:
        if gt_file in file_manifest:
            print(f"  ✓ {gt_file} - IN manifest")
        else:
            print(f"  ✗ {gt_file} - NOT in manifest")
    
    print(f"\nCandidate list would have:")
    print(f"  - {len(actual_files)} actual files (GT modified + created)")
    print(f"  - {num_distractors} distractors")
    print(f"  - Total: {len(actual_files) + num_distractors} files")
    
else:
    print(f"Repo data not found at {repo_path}")
