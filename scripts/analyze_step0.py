"""Analyze step 0 file accuracy scores."""
import json
import re

# Load step 0 completion (JSONL format - one JSON per line)
with open("output/grpo_235b_split_rewards/completions.jsonl", encoding="utf-8") as f:
    step0 = json.loads(f.readline())

# Load repo IR for manifest
with open("data/commit_pairs_production/astro_pr15779/repo_ir.json", encoding="utf-8") as f:
    repo_ir = json.load(f)

manifest = set(repo_ir.get("file_manifest", []))

# Parse model completion - remove endoftext token
completion_raw = step0["completion"]
# Find the JSON object (everything before endoftext marker)
match = re.search(r'^(\{.*\})', completion_raw, re.DOTALL)
if match:
    completion_json = match.group(1)
else:
    completion_json = completion_raw

try:
    plan = json.loads(completion_json)
except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")
    print(f"First 500 chars: {completion_json[:500]}")
    exit(1)

# Extract model files
model_modified = set()
model_created = set()
for ticket in plan.get("tickets", []):
    model_modified.update(ticket.get("files_to_modify", []))
    model_created.update(ticket.get("files_to_create", []))

# Ground truth from diff
gt_modified = set(step0["diff_files"]["modified"])
gt_created = set(step0["diff_files"]["created"])

print("=" * 80)
print("GROUND TRUTH (from git diff)")
print("=" * 80)
print(f"Modified: {sorted(gt_modified)}")
print(f"Created:  {sorted(gt_created)}")

print("\n" + "=" * 80)
print(f"MODEL GENERATED ({len(model_modified)} modified, {len(model_created)} created)")
print("=" * 80)
print("\nModified files:")
for f in sorted(model_modified):
    in_manifest = f in manifest
    in_gt = f in gt_modified
    status = "✓ MATCH" if in_gt else ("✗ WRONG" if in_manifest else "✗ HALLUCINATED")
    print(f"  {status:15s} {f}")

print("\nCreated files:")
for f in sorted(model_created):
    in_gt = f in gt_created
    status = "✓ MATCH" if in_gt else "✗ WRONG"
    print(f"  {status:15s} {f}")

print("\n" + "=" * 80)
print("SCORES")
print("=" * 80)
overlap_modified = len(model_modified & gt_modified)
overlap_created = len(model_created & gt_created)
union_modified = len(model_modified | gt_modified)
union_created = len(model_created | gt_created)

jaccard_modified = overlap_modified / union_modified if union_modified > 0 else 0
jaccard_created = overlap_created / union_created if union_created > 0 else 0

print(f"Modified files:")
print(f"  Overlap: {overlap_modified}/{len(gt_modified)}")
print(f"  Jaccard: {jaccard_modified:.3f}")
print(f"  Score:   {jaccard_modified * 1.5:.3f} / 1.5")

print(f"\nCreated files:")
print(f"  Overlap: {overlap_created}/{len(gt_created)}")
print(f"  Jaccard: {jaccard_created:.3f}")
print(f"  Score:   {jaccard_created * 1.5:.3f} / 1.5")

print(f"\nActual rewards: {step0['rewards']}")
