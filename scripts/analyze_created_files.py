import json
from pathlib import Path
import re

data_dir = Path("data/commit_pairs_production")
repos = [d for d in data_dir.iterdir() if d.is_dir()]

created_counts = {}
total = 0
zero_created = 0

for repo_dir in repos:
    diff_path = repo_dir / "ground_truth_diff.txt"
    if diff_path.exists():
        total += 1
        diff = diff_path.read_text(encoding='utf-8', errors='ignore')
        # Count "new file mode" lines
        created = len([1 for line in diff.split('\n') if 'new file mode' in line])
        if created == 0:
            zero_created += 1
        created_counts[repo_dir.name] = created

print(f"Total repos: {total}")
print(f"Repos with 0 created files: {zero_created} ({zero_created/total*100:.1f}%)")
print(f"Repos with created files: {total - zero_created} ({(total-zero_created)/total*100:.1f}%)")
print(f"\nDistribution:")
print(f"  0 created: {zero_created}")
print(f"  1-5 created: {len([c for c in created_counts.values() if 1 <= c <= 5])}")
print(f"  6-10 created: {len([c for c in created_counts.values() if 6 <= c <= 10])}")
print(f"  >10 created: {len([c for c in created_counts.values() if c > 10])}")
