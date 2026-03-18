"""Test stratified sampling logic before running full training."""
import json
from pathlib import Path

# Load examples
data_dir = Path("data/commit_pairs_production")
examples = []

for repo_dir in sorted(data_dir.iterdir()):
    if not repo_dir.is_dir():
        continue
    
    diff_path = repo_dir / "ground_truth_diff.txt"
    if not diff_path.exists():
        continue
    
    diff_text = diff_path.read_text(encoding='utf-8', errors='ignore')
    created = len([1 for line in diff_text.split('\n') if 'new file mode' in line])
    
    examples.append({
        "repo_name": repo_dir.name,
        "has_created": created > 0,
        "num_created": created
    })

# Split by created files
with_created = [ex for ex in examples if ex["has_created"]]
without_created = [ex for ex in examples if not ex["has_created"]]

print(f"Total examples: {len(examples)}")
print(f"With created files: {len(with_created)} ({len(with_created)/len(examples)*100:.1f}%)")
print(f"Without created files: {len(without_created)} ({len(without_created)/len(examples)*100:.1f}%)")

# Simulate stratified batching
batch_size = 1
half_batch = batch_size // 2
n_batches = 100

print(f"\nSimulating {n_batches} batches with batch_size={batch_size}:")
print(f"  Half batch size: {half_batch}")

if half_batch == 0:
    print("\n⚠️  WARNING: batch_size=1 means half_batch=0, stratified sampling won't work!")
    print("  Need batch_size >= 2 for 50/50 split")
else:
    # Simulate first 10 batches
    import random
    random.seed(42)
    random.shuffle(with_created)
    random.shuffle(without_created)
    
    created_count = 0
    for batch_idx in range(10):
        batch_with = with_created[(batch_idx * half_batch) % len(with_created):(batch_idx * half_batch + half_batch) % len(with_created) + half_batch]
        batch_without = without_created[(batch_idx * half_batch) % len(without_created):(batch_idx * half_batch + half_batch) % len(without_created) + half_batch]
        
        if len(batch_with) < half_batch:
            batch_with = with_created[(batch_idx * half_batch) % len(with_created):] + with_created[:half_batch - len(batch_with)]
        if len(batch_without) < half_batch:
            batch_without = without_created[(batch_idx * half_batch) % len(without_created):] + without_created[:half_batch - len(batch_without)]
        
        batch = batch_with + batch_without
        created_in_batch = sum(1 for ex in batch if ex["has_created"])
        created_count += created_in_batch
        
        print(f"  Batch {batch_idx}: {created_in_batch}/{len(batch)} with created files")
    
    print(f"\n  Overall: {created_count}/10 batches have created files ({created_count/10*100:.0f}%)")
