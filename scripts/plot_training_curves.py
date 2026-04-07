"""Plot GRPO training curves (paper Fig 5) and reward component bars (paper Fig 6).

Outputs use paper-aligned filenames: fig5_training_curves.* and fig6_reward_components.*.
Reward component values are read from output/eval_results_trained.json — no estimates.
"""
import json
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt

METRICS = Path("output/grpo_v5_100steps/metrics.jsonl")
EVAL_UNSEEN = Path("output/eval_results_trained.json")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ---------- Fig 5: training curves ----------
train = []
with METRICS.open() as f:
    for line in f:
        d = json.loads(line)
        if "progress/global_step" in d:
            train.append(d)

steps = [d["progress/global_step"] for d in train]
reward = [d["reward/total"] for d in train]
fmt = [d["reward/format_compliance_rate"] for d in train]
exist = [d["reward/existing_file_accuracy"] for d in train]
sem = [d["reward/semantic_similarity"] for d in train]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(steps, reward, label="Total reward", color="C0", lw=2)
ax.plot(steps, sem, label="Semantic sim.", color="C2", lw=1.5, alpha=0.8)
ax.plot(steps, exist, label="Existing-file reward", color="C3", lw=1.5, alpha=0.8)
ax.set_xlabel("GRPO step")
ax.set_ylabel("Reward")
ax.set_title("(a) Reward components vs. step")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(steps, [f * 100 for f in fmt], color="C1", lw=2)
ax.axhline(100, ls="--", color="gray", alpha=0.5)
ax.axvline(43, ls=":", color="C3", alpha=0.7, label="step 43: 100%")
ax.set_xlabel("GRPO step")
ax.set_ylabel("Format compliance (%)")
ax.set_title("(b) Format compliance rate")
ax.set_ylim(0, 105)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "fig5_training_curves.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig5_training_curves.png", dpi=200, bbox_inches="tight")
print(f"Wrote {OUT/'fig5_training_curves.pdf'}")

# ---------- Fig 6: reward components from REAL eval data ----------
evals = json.load(EVAL_UNSEEN.open())
KEYS = [
    ("format_compliance",      "Format\ncompliance",      0.50),
    ("format_partial",         "Format\npartial",         0.25),
    ("existing_file_accuracy", "Existing file\naccuracy", 1.50),
    ("created_file_accuracy",  "Created file\naccuracy",  1.50),
    ("semantic_similarity",    "Semantic\nsimilarity",    3.00),
    ("structural_quality",     "Structural\nquality",     0.50),
]
scores = [mean(e["best_rewards"][k] for e in evals) for k, _, _ in KEYS]
labels = [lbl for _, lbl, _ in KEYS]
maxes  = [m for _, _, m in KEYS]
total  = sum(scores)
total_max = sum(maxes)
print(f"Unseen-12 mean total reward = {total:.3f} / {total_max:.2f}")

fig, ax = plt.subplots(figsize=(8.5, 4))
x = range(len(labels))
ax.bar(x, maxes, color="lightgray", label="Max", width=0.65)
ax.bar(x, scores, color="C0", label="RepoDesign GRPO", width=0.65)
for i, (s, m) in enumerate(zip(scores, maxes)):
    ax.text(i, s + 0.05, f"{s/m*100:.0f}%", ha="center", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Reward")
ax.set_title(f"Per-component reward on 12 unseen-repo examples (total {total:.2f}/{total_max:.2f})")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig6_reward_components.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig6_reward_components.png", dpi=200, bbox_inches="tight")
print(f"Wrote {OUT/'fig6_reward_components.pdf'}")
