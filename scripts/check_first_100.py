"""Check distribution in first 100 repos (what --max-samples 100 sees)."""
from pathlib import Path

data_dir = Path("data/commit_pairs_production")
repos = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:100]

with_created = 0
without_created = 0

for repo_dir in repos:
    diff_path = repo_dir / "ground_truth_diff.txt"
    if diff_path.exists():
        diff_text = diff_path.read_text(encoding='utf-8', errors='ignore')
        created = len([1 for line in diff_text.split('\n') if 'new file mode' in line])
        if created > 0:
            with_created += 1
        else:
            without_created += 1

total = with_created + without_created
print(f"First 100 repos:")
print(f"  With created files: {with_created} ({with_created/total*100:.1f}%)")
print(f"  Without created files: {without_created} ({without_created/total*100:.1f}%)")
print(f"\nThis is what gets stratified (31 with, 69 without)")
