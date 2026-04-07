"""Print corrected paper Tables 1 & 2 from output/eval_*.json.

Reports both reward values (component max 1.5) and TRUE F1 = reward / 1.5,
to fix the F1 = 1.03 bug in the original Table 2.
"""
import json
from pathlib import Path
from statistics import mean

def load(p):
    return [e for e in json.load(open(p)) if e.get("best_rewards")]

unseen = load("output/eval_results_trained.json")
held   = load("output/eval_held_out_trained.json")

KEYS = ["format_compliance", "format_partial", "existing_file_accuracy",
        "created_file_accuracy", "semantic_similarity", "structural_quality", "total"]

def avg(data, k):
    return mean(e["best_rewards"][k] for e in data)

print("=" * 72)
print("Table 1 (CORRECTED): per-component reward, 12 unseen-repo examples")
print("=" * 72)
print(f"{'Component':<28}{'Score':>8}{'Max':>8}{'% max':>10}")
maxes = {"format_compliance":0.50,"format_partial":0.25,"existing_file_accuracy":1.50,
         "created_file_accuracy":1.50,"semantic_similarity":3.00,"structural_quality":0.50}
for k in ["format_compliance","format_partial","existing_file_accuracy",
          "created_file_accuracy","semantic_similarity","structural_quality"]:
    s = avg(unseen, k); m = maxes[k]
    print(f"  {k:<26}{s:>8.3f}{m:>8.2f}{s/m*100:>9.0f}%")
total = avg(unseen, "total")
total_max = sum(maxes.values())
print(f"  {'TOTAL':<26}{total:>8.3f}{total_max:>8.2f}{total/total_max*100:>9.0f}%")
print(f"  (paper claimed max=7.75; actual schema max={total_max:.2f})")
print()

print("=" * 72)
print("Table 2 (CORRECTED): comparative results — reward AND true F1")
print("=" * 72)
print(f"{'Split':<28}{'n':>4}{'Reward':>9}{'FmtPass':>10}"
      f"{'Mod F1':>10}{'Cre F1':>10}{'SemSim':>10}")
for name, data in [("RepoDesign GRPO unseen", unseen),
                   ("RepoDesign GRPO held-out", held)]:
    n = len(data)
    r = avg(data, "total")
    fp = mean(1.0 if e["best_rewards"]["format_compliance"] >= 0.5 else 0.0 for e in data)
    mod_f1 = avg(data, "existing_file_accuracy") / 1.5
    cre_f1 = avg(data, "created_file_accuracy") / 1.5
    sem    = avg(data, "semantic_similarity")
    print(f"  {name:<26}{n:>4}{r:>9.2f}{fp*100:>9.0f}%"
          f"{mod_f1:>10.2f}{cre_f1:>10.2f}{sem:>10.2f}")
print()
print("Note: original Table 2 'Modify F1 = 1.03' was the unscaled REWARD value")
print("      (reward max 1.5, not F1). True F1 ≤ 1 by definition.")
