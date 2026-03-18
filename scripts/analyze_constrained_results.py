import json

completions_path = "output/grpo_235b_constrained/completions.jsonl"

with open(completions_path) as f:
    steps = [json.loads(line) for line in f if line.strip()]

print(f"Total steps completed: {len(steps)}\n")
print("Step | Repo | Modified (GT) | Created (GT) | Exist Acc | Created Acc | TF-IDF | Total")
print("-" * 95)

for step_data in steps:
    step = step_data['step']
    repo = step_data['repo']
    rewards = step_data['rewards']
    diff = step_data['diff_files']
    
    n_modified = len(diff.get('modified', []))
    n_created = len(diff.get('created', []))
    
    exist_acc = rewards['existing_file_accuracy']
    created_acc = rewards['created_file_accuracy']
    tfidf = rewards['tfidf_similarity']
    total = rewards['total']
    
    print(f"{step:4d} | {repo:20s} | {n_modified:13d} | {n_created:12d} | {exist_acc:9.3f} | {created_acc:11.3f} | {tfidf:6.3f} | {total:5.2f}")

print("\nSummary:")
avg_exist = sum(s['rewards']['existing_file_accuracy'] for s in steps) / len(steps)
avg_created = sum(s['rewards']['created_file_accuracy'] for s in steps) / len(steps)
avg_tfidf = sum(s['rewards']['tfidf_similarity'] for s in steps) / len(steps)
avg_total = sum(s['rewards']['total'] for s in steps) / len(steps)

print(f"Average existing_file_accuracy: {avg_exist:.3f}")
print(f"Average created_file_accuracy: {avg_created:.3f}")
print(f"Average tfidf_similarity: {avg_tfidf:.3f}")
print(f"Average total reward: {avg_total:.2f}")
