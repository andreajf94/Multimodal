import json
from pathlib import Path

completions_file = Path("output/grpo_235b_100steps/completions.jsonl")

if completions_file.exists():
    with open(completions_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print("Steps 20-30 analysis:")
    steps_20_30 = [d for d in data if 20 <= d['step'] <= 30]
    
    for d in steps_20_30:
        r = d['rewards']
        print(f"Step {d['step']:2d}: {d['repo'][:20]:20s} | total={r['total']:.2f} exist={r['existing_file_accuracy']:.2f} created={r['created_file_accuracy']:.2f} tfidf={r['tfidf_similarity']:.2f}")
    
    if steps_20_30:
        avg_total = sum(d['rewards']['total'] for d in steps_20_30) / len(steps_20_30)
        print(f"\nAverage reward for steps 20-30: {avg_total:.2f}")
        
        # Compare to steps 0-10
        steps_0_10 = [d for d in data if 0 <= d['step'] <= 10]
        if steps_0_10:
            avg_0_10 = sum(d['rewards']['total'] for d in steps_0_10) / len(steps_0_10)
            print(f"Average reward for steps 0-10: {avg_0_10:.2f}")
else:
    print("completions.jsonl not found")
