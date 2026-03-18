"""Test the new split file accuracy rewards."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.repodesign.training.reward import compute_rewards, parse_diff_files

# Load astro example
repo_dir = Path("data/commit_pairs_production/astro_pr15779")

with open(repo_dir / "teacher_plan.json", encoding="utf-8") as f:
    teacher_plan = json.load(f)

with open(repo_dir / "ground_truth_diff.txt", encoding="utf-8") as f:
    diff_text = f.read()

diff_files = parse_diff_files(diff_text)

print("="*80)
print("GROUND TRUTH DIFF FILES")
print("="*80)
print(f"Modified: {diff_files['modified']}")
print(f"Created:  {diff_files['created']}")

# Test with teacher plan as completion (should get high score)
teacher_completion = json.dumps(teacher_plan, indent=2)

results = compute_rewards(
    completions=[teacher_completion],
    file_manifests=[[]],
    specs=[{}],
    teacher_plans=[teacher_plan],
    diff_files=[diff_files],
)

print("\n" + "="*80)
print("TEST 1: Teacher plan as completion")
print("="*80)
for key, value in results[0].items():
    print(f"  {key}: {value:.3f}")

# Test with fake completion
fake_plan = {
    "architecture_decisions": [],
    "tickets": [
        {
            "id": "T-001",
            "title": "Test",
            "description": "Test",
            "files_to_modify": ["packages/astro/test/units/remote-pattern.test.js"],
            "files_to_create": [".changeset/perky-dots-prove.md"]
        }
    ],
    "implementation_summary": "This is a test implementation that modifies some files."
}

fake_completion = json.dumps(fake_plan, indent=2)

results2 = compute_rewards(
    completions=[fake_completion],
    file_manifests=[[]],
    specs=[{}],
    teacher_plans=[teacher_plan],
    diff_files=[diff_files],
)

print("\n" + "="*80)
print("TEST 2: Partial match (1 modified, 1 created)")
print("="*80)
for key, value in results2[0].items():
    print(f"  {key}: {value:.3f}")

print(f"\nExpected existing_file_accuracy: 1/3 = 0.333 * 1.5 = {(1/3) * 1.5:.3f}")
print(f"Expected created_file_accuracy: 1/1 = 1.000 * 1.5 = {1.0 * 1.5:.3f}")
