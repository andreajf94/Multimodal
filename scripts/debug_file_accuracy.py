"""Debug script to test file_accuracy reward function."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.repodesign.training.reward import file_accuracy, _parse_plan_json, _normalize_path

# Load a sample teacher plan
teacher_path = Path("data/commit_pairs_production/dub_pr3514/teacher_plan.json")
with open(teacher_path) as f:
    teacher_plan = json.load(f)

# Extract teacher files manually
print("=" * 80)
print("TEACHER PLAN FILES:")
print("=" * 80)
teacher_files = set()
for ticket in teacher_plan.get("tickets", []):
    for f in ticket.get("files_to_modify", []):
        teacher_files.add(f)
        print(f"  [modify] {f}")
    for f in ticket.get("files_to_create", []):
        teacher_files.add(f)
        print(f"  [create] {f}")
for decision in teacher_plan.get("architecture_decisions", []):
    for f in decision.get("files_affected", []):
        teacher_files.add(f)
        print(f"  [affected] {f}")

print(f"\nTotal unique files: {len(teacher_files)}")
print("\nNormalized:")
teacher_files_norm = {_normalize_path(f) for f in teacher_files if f}
for f in sorted(teacher_files_norm):
    print(f"  {f}")

# Test with a mock completion that should match
print("\n" + "=" * 80)
print("TEST 1: Perfect match (copy teacher plan)")
print("=" * 80)
mock_completion = json.dumps(teacher_plan, indent=2)
score = file_accuracy([mock_completion], [teacher_plan])
print(f"Score: {score[0]:.3f} (expected: 3.000)")

# Test with partial match
print("\n" + "=" * 80)
print("TEST 2: Partial match (only first 2 files)")
print("=" * 80)
partial_plan = {
    "architecture_decisions": [],
    "tickets": [{
        "id": "T-001",
        "title": "Test",
        "description": "Test",
        "files_to_modify": list(teacher_files_norm)[:2],
        "files_to_create": []
    }],
    "implementation_summary": "Test"
}
mock_completion2 = json.dumps(partial_plan, indent=2)
score2 = file_accuracy([mock_completion2], [teacher_plan])
print(f"Files in completion: {list(teacher_files_norm)[:2]}")
print(f"Score: {score2[0]:.3f}")
print(f"Expected Jaccard: {2 / len(teacher_files_norm):.3f} * 3.0 = {(2 / len(teacher_files_norm)) * 3.0:.3f}")

# Test with completely wrong files
print("\n" + "=" * 80)
print("TEST 3: No match (wrong files)")
print("=" * 80)
wrong_plan = {
    "architecture_decisions": [],
    "tickets": [{
        "id": "T-001",
        "title": "Test",
        "description": "Test",
        "files_to_modify": ["src/main.py", "config/settings.yml"],
        "files_to_create": []
    }],
    "implementation_summary": "Test"
}
mock_completion3 = json.dumps(wrong_plan, indent=2)
score3 = file_accuracy([mock_completion3], [teacher_plan])
print(f"Files in completion: {wrong_plan['tickets'][0]['files_to_modify']}")
print(f"Score: {score3[0]:.3f} (expected: 0.000)")

# Test with invalid JSON
print("\n" + "=" * 80)
print("TEST 4: Invalid JSON")
print("=" * 80)
invalid_completion = "This is not valid JSON"
score4 = file_accuracy([invalid_completion], [teacher_plan])
print(f"Score: {score4[0]:.3f} (expected: 0.000)")
