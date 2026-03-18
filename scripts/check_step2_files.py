import json

with open('output/grpo_235b_constrained/completions.jsonl') as f:
    lines = list(f)
    step2 = json.loads(lines[2])

gt_modified = step2['diff_files']['modified']
gt_created = step2['diff_files']['created']

print("=" * 70)
print(f"Step {step2['step']}: {step2['repo']}")
print("=" * 70)
print(f"\nGround Truth:")
print(f"  Modified: {gt_modified}")
print(f"  Created: {gt_created}")

# Parse completion
completion_text = step2['completion']

# Find the end of the JSON object
brace_count = 0
json_end = 0
for i, char in enumerate(completion_text):
    if char == '{':
        brace_count += 1
    elif char == '}':
        brace_count -= 1
        if brace_count == 0:
            json_end = i + 1
            break

json_only = completion_text[:json_end]
plan = json.loads(json_only)

# Extract predicted files
predicted_modified = []
predicted_created = []

for ticket in plan.get('tickets', []):
    predicted_modified.extend(ticket.get('files_to_modify', []))
    predicted_created.extend(ticket.get('files_to_create', []))

print(f"\nModel Predicted:")
print(f"  Modified ({len(predicted_modified)}): {predicted_modified[:5]}{'...' if len(predicted_modified) > 5 else ''}")
print(f"  Created ({len(predicted_created)}): {predicted_created}")

# Calculate overlaps
gt_mod_set = set(gt_modified)
pred_mod_set = set(predicted_modified)
mod_intersection = gt_mod_set & pred_mod_set

print(f"\nModified Files Analysis:")
print(f"  Intersection (correct): {mod_intersection}")
print(f"  Model missed: {gt_mod_set - pred_mod_set}")
print(f"  Model hallucinated: {pred_mod_set - gt_mod_set if pred_mod_set - gt_mod_set else 'None'}")

gt_create_set = set(gt_created)
pred_create_set = set(predicted_created)

print(f"\nCreated Files Analysis:")
print(f"  Both empty: {len(gt_create_set) == 0 and len(pred_create_set) == 0}")
print(f"  → Score: 1.5 (full credit for correctly identifying no new files)")

print(f"\nRewards:")
print(f"  existing_file_accuracy: {step2['rewards']['existing_file_accuracy']}")
print(f"  created_file_accuracy: {step2['rewards']['created_file_accuracy']}")
