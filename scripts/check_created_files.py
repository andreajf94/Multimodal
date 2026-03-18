import json

with open('output/grpo_235b_constrained/completions.jsonl') as f:
    step0 = json.loads(f.readline())

gt_created = step0['diff_files']['created']
print(f"Ground truth created files: {gt_created}")
print(f"Number of GT created files: {len(gt_created)}")

# Parse the completion to get predicted files
completion_text = step0['completion']
try:
    # Find the end of the JSON object (matching braces)
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
    
    predicted_created = []
    for ticket in plan.get('tickets', []):
        predicted_created.extend(ticket.get('files_to_create', []))
    
    print(f"\nModel predicted created files: {predicted_created}")
    print(f"Number of predicted created files: {len(predicted_created)}")
    
    # Calculate Jaccard
    gt_set = set(gt_created)
    pred_set = set(predicted_created)
    
    intersection = gt_set & pred_set
    union = gt_set | pred_set
    
    print(f"\nIntersection (correct): {intersection}")
    print(f"Union (all mentioned): {union}")
    print(f"Jaccard similarity: {len(intersection)}/{len(union)} = {len(intersection)/len(union) if union else 0:.3f}")
    print(f"Score (Jaccard * 1.5): {(len(intersection)/len(union) if union else 0) * 1.5:.3f}")
    print(f"Actual score from rewards: {step0['rewards']['created_file_accuracy']:.3f}")
    
except Exception as e:
    print(f"Error parsing: {e}")
