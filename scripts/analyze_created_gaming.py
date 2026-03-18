import json

with open('output/grpo_235b_constrained/completions.jsonl') as f:
    steps = [json.loads(line) for line in f if line.strip()]

print(f"Analyzing {len(steps)} steps for created file reward gaming\n")

# Categorize by created file scenario
zero_gt_zero_pred = []  # Both 0 - gets 1.5
zero_gt_nonzero_pred = []  # GT 0, predicted >0 - gets 0.0
nonzero_gt_zero_pred = []  # GT >0, predicted 0 - gets 0.0
nonzero_gt_nonzero_pred = []  # Both >0 - gets Jaccard

for step in steps:
    gt_created = len(step['diff_files'].get('created', []))
    created_score = step['rewards']['created_file_accuracy']
    
    # Infer predicted created count from score
    if gt_created == 0:
        if created_score == 1.5:
            zero_gt_zero_pred.append(step)
        elif created_score == 0.0:
            zero_gt_nonzero_pred.append(step)
    else:
        if created_score == 0.0:
            nonzero_gt_zero_pred.append(step)
        else:
            nonzero_gt_nonzero_pred.append(step)

print("Created File Reward Distribution:")
print(f"  GT=0, Pred=0 (score 1.5): {len(zero_gt_zero_pred)} steps ({len(zero_gt_zero_pred)/len(steps)*100:.1f}%)")
print(f"  GT=0, Pred>0 (score 0.0): {len(zero_gt_nonzero_pred)} steps ({len(zero_gt_nonzero_pred)/len(steps)*100:.1f}%)")
print(f"  GT>0, Pred=0 (score 0.0): {len(nonzero_gt_zero_pred)} steps ({len(nonzero_gt_zero_pred)/len(steps)*100:.1f}%)")
print(f"  GT>0, Pred>0 (Jaccard):   {len(nonzero_gt_nonzero_pred)} steps ({len(nonzero_gt_nonzero_pred)/len(steps)*100:.1f}%)")

print(f"\nModel is predicting 0 created files in {len(zero_gt_zero_pred) + len(nonzero_gt_zero_pred)}/{len(steps)} steps ({(len(zero_gt_zero_pred) + len(nonzero_gt_zero_pred))/len(steps)*100:.1f}%)")

# Calculate what the data split should be (50/50 stratified)
gt_with_created = sum(1 for s in steps if len(s['diff_files'].get('created', [])) > 0)
gt_without_created = len(steps) - gt_with_created

print(f"\nGround Truth Distribution:")
print(f"  With created files: {gt_with_created} ({gt_with_created/len(steps)*100:.1f}%)")
print(f"  Without created files: {gt_without_created} ({gt_without_created/len(steps)*100:.1f}%)")

print(f"\n🚨 REWARD GAMING CHECK:")
correct_zeros = len(zero_gt_zero_pred)
incorrect_zeros = len(nonzero_gt_zero_pred)
print(f"  Model correctly predicts 0: {correct_zeros} steps")
print(f"  Model incorrectly predicts 0 (should have files): {incorrect_zeros} steps")
print(f"  Zero prediction rate: {(correct_zeros + incorrect_zeros)/len(steps)*100:.1f}%")

if incorrect_zeros > 0:
    print(f"\n  ⚠️  Model is predicting 0 when GT>0 in {incorrect_zeros} cases!")
    print(f"  This suggests potential reward gaming.")
    
avg_created_score = sum(s['rewards']['created_file_accuracy'] for s in steps) / len(steps)
print(f"\nAverage created_file_accuracy: {avg_created_score:.3f}")
print(f"Max possible: 1.500")
print(f"Utilization: {avg_created_score/1.5*100:.1f}%")
