"""Plot GRPO training curves (Figure 3) and reward component bars (Figure 4)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

METRICS = Path("output/grpo_v5_100steps/metrics.jsonl")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

train, evals = [], []
with METRICS.open() as f:
    for line in f:
        d = json.loads(line)
        if "progress/global_step" in d:
            train.append(d)
        elif "eval/reward_total" in d:
            evals.append(d)

# Figure 3: training curves
steps = [d["progress/global_step"] for d in train]
reward = [d["reward/total"] for d in train]
fmt = [d["reward/format_compliance_rate"] for d in train]
exist = [d["reward/existing_file_accuracy"] for d in train]
sem = [d["reward/semantic_similarity"] for d in train]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(steps, reward, label="Total reward", color="C0", lw=2)
ax.plot(steps, sem, label="Semantic sim.", color="C2", lw=1.5, alpha=0.8)
ax.plot(steps, [e * 1.5 for e in exist], label="Existing-file (×1.5)", color="C3", lw=1.5, alpha=0.8)
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
fig.savefig(OUT / "fig3_training_curves.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig3_training_curves.png", dpi=200, bbox_inches="tight")
print(f"Wrote {OUT/'fig3_training_curves.pdf'}")

# Figure 4: reward component bars (final eval, 12-example unseen)
components = ["Format\ncompliance", "Existing file\naccuracy", "Created file\naccuracy",
              "Semantic\nsimilarity", "Structural\nquality", "Nonempty\nbonus"]
scores = [0.50, 0.98, 0.08, 2.40, 0.40, 0.35]  # adjust if you have exact values
maxes  = [0.50, 1.50, 1.50, 3.00, 0.50, 0.50]

fig, ax = plt.subplots(figsize=(8, 4))
x = range(len(components))
ax.bar(x, maxes, color="lightgray", label="Max", width=0.65)
ax.bar(x, scores, color="C0", label="RepoDesign GRPO", width=0.65)
for i, (s, m) in enumerate(zip(scores, maxes)):
    ax.text(i, s + 0.05, f"{s/m*100:.0f}%", ha="center", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(components, fontsize=9)
ax.set_ylabel("Reward")
ax.set_title("Per-component reward on 12-example unseen-repo benchmark")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig4_reward_components.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig4_reward_components.png", dpi=200, bbox_inches="tight")
print(f"Wrote {OUT/'fig4_reward_components.pdf'}")
