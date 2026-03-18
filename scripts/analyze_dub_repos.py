from pathlib import Path

data_dir = Path("data/commit_pairs_production")
dub_repos = [d for d in data_dir.iterdir() if d.name.startswith('dub_')]
non_dub_repos = [d for d in data_dir.iterdir() if not d.name.startswith('dub_') and d.is_dir()]

print(f"Total dub repos: {len(dub_repos)}")

# Check created files in dub repos
dub_created = 0
dub_total = 0
for d in dub_repos:
    diff_path = d / 'ground_truth_diff.txt'
    if diff_path.exists():
        dub_total += 1
        diff = diff_path.read_text(encoding='utf-8', errors='ignore')
        created = len([1 for line in diff.split('\n') if 'new file mode' in line])
        dub_created += created

# Check created files in non-dub repos
non_dub_created = 0
non_dub_total = 0
for d in non_dub_repos[:len(dub_repos)]:  # Sample same number
    diff_path = d / 'ground_truth_diff.txt'
    if diff_path.exists():
        non_dub_total += 1
        diff = diff_path.read_text(encoding='utf-8', errors='ignore')
        created = len([1 for line in diff.split('\n') if 'new file mode' in line])
        non_dub_created += created

print(f"\nDub repos: {dub_created} total created files across {dub_total} repos = {dub_created/dub_total if dub_total > 0 else 0:.2f} avg")
print(f"Non-dub repos (sample): {non_dub_created} total created files across {non_dub_total} repos = {non_dub_created/non_dub_total if non_dub_total > 0 else 0:.2f} avg")

# Check how many dub repos have 0 created
dub_zero = sum(1 for d in dub_repos if (d / 'ground_truth_diff.txt').exists() and 
               len([1 for line in (d / 'ground_truth_diff.txt').read_text(encoding='utf-8', errors='ignore').split('\n') if 'new file mode' in line]) == 0)
print(f"\nDub repos with 0 created: {dub_zero}/{dub_total} ({dub_zero/dub_total*100 if dub_total > 0 else 0:.1f}%)")
