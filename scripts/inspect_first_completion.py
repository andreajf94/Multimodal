"""Quick script to manually test reward on a sample to ensure it's working."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.repodesign.training.reward import file_accuracy, _parse_plan_json

# Simulate what the model might generate (typical untrained output)
fake_model_output = """
{
  "architecture_decisions": [
    {
      "dimension": "api_design",
      "recommendation": "Add new endpoint",
      "rationale": "To support the feature",
      "alternatives_considered": [],
      "files_affected": ["src/api/routes.py", "src/handlers/new_handler.py"]
    }
  ],
  "tickets": [
    {
      "id": "T-001",
      "title": "Implement feature",
      "description": "Add the new feature",
      "files_to_modify": ["src/main.py", "config.yaml"],
      "files_to_create": ["src/new_module.py"],
      "estimated_effort": "medium"
    }
  ],
  "implementation_summary": "This adds the new feature by modifying core files."
}
"""

# Load real teacher plan
teacher_path = Path("data/commit_pairs_production/dub_pr3514/teacher_plan.json")
with open(teacher_path) as f:
    teacher = json.load(f)

print("Testing file_accuracy with fake model output vs real teacher plan...")
print("\n" + "="*80)
print("FAKE MODEL FILES:")
fake_plan = _parse_plan_json(fake_model_output)
if fake_plan:
    for ticket in fake_plan.get("tickets", []):
        print(f"  files_to_modify: {ticket.get('files_to_modify', [])}")
        print(f"  files_to_create: {ticket.get('files_to_create', [])}")
    for decision in fake_plan.get("architecture_decisions", []):
        print(f"  files_affected: {decision.get('files_affected', [])}")

print("\n" + "="*80)
print("TEACHER FILES:")
for ticket in teacher.get("tickets", []):
    print(f"  files_to_modify: {ticket.get('files_to_modify', [])}")
    if ticket.get('files_to_create'):
        print(f"  files_to_create: {ticket.get('files_to_create', [])}")

score = file_accuracy([fake_model_output], [teacher])
print("\n" + "="*80)
print(f"FILE ACCURACY SCORE: {score[0]:.3f}")
print("="*80)
print("\nThis simulates what an untrained model might generate.")
print("Expected: 0.000 (no overlap with teacher files)")
