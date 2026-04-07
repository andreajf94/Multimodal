"""Generate per-repo dataset stats, scale-tier distribution, and Fig 4 plot.

Outputs:
  figures/dataset_stats.csv  — per-repo table
  figures/fig4_dataset.pdf   — two-panel: languages (left), scale tiers (right)
"""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

ROOT = Path("data/commit_pairs_production")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

by_repo = defaultdict(lambda: {"prs": 0, "loc": 0, "stars": 0, "contrib": 0,
                                "tier": "?", "lang": "?", "manifest": []})

for d in sorted(ROOT.iterdir()):
    ir = d / "repo_ir.json"
    if not ir.exists():
        continue
    data = json.loads(ir.read_text())
    m = data.get("repo_metadata") or {}
    name = m.get("name", d.name.split("_pr")[0])
    r = by_repo[name]
    r["prs"] += 1
    r["loc"] = m.get("total_loc", r["loc"])
    r["stars"] = m.get("star_count", r["stars"])
    r["contrib"] = m.get("num_contributors", r["contrib"])
    r["tier"] = m.get("scale_tier", r["tier"])
    r["lang"] = m.get("primary_language", r["lang"])
    r["manifest"].append(len(data.get("file_manifest") or []))

print(f"{'Repo':<18}{'PRs':>5}{'LOC':>10}{'Stars':>9}{'Contrib':>9}  {'Lang':<12}{'Tier':<10}{'Avg manifest':>14}")
print("-" * 90)
total_prs = 0
tier_counts = defaultdict(int)
lang_counts = defaultdict(int)
for name, r in sorted(by_repo.items()):
    avg_man = sum(r["manifest"]) / len(r["manifest"])
    print(f"{name:<18}{r['prs']:>5}{r['loc']:>10}{r['stars']:>9}{r['contrib']:>9}  {r['lang']:<12}{r['tier']:<10}{avg_man:>14.0f}")
    total_prs += r["prs"]
    tier_counts[r["tier"]] += r["prs"]
    lang_counts[r["lang"] or "unknown"] += r["prs"]

print("-" * 90)
print(f"{'TOTAL':<18}{total_prs:>5}   ({len(by_repo)} repos)")
print()
print("Scale-tier distribution (by PRs):")
for tier, n in sorted(tier_counts.items(), key=lambda x: -x[1]):
    print(f"  {tier:<12}{n:>5}  ({n/total_prs*100:.1f}%)")
print("Language distribution (by PRs):")
for lang, n in sorted(lang_counts.items(), key=lambda x: -x[1]):
    print(f"  {lang:<14}{n:>5}  ({n/total_prs*100:.1f}%)")

# CSV
with open(OUT / "dataset_stats.csv", "w") as f:
    f.write("repo,prs,loc,stars,contributors,language,tier,avg_manifest\n")
    for name, r in sorted(by_repo.items()):
        avg_man = sum(r["manifest"]) / len(r["manifest"])
        f.write(f"{name},{r['prs']},{r['loc']},{r['stars']},{r['contrib']},{r['lang']},{r['tier']},{avg_man:.0f}\n")
print(f"\nWrote {OUT/'dataset_stats.csv'}")

# Fig 4: two-panel breakdown
TIER_ORDER = ["hobby", "startup", "growth", "enterprise"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Languages
langs_sorted = sorted(lang_counts.items(), key=lambda x: -x[1])
ax = axes[0]
ax.bar([l for l, _ in langs_sorted], [n for _, n in langs_sorted], color="C0")
ax.set_ylabel("Commit pairs")
ax.set_title(f"(a) Primary language ({total_prs} PRs / {len(by_repo)} repos)")
for i, (_, n) in enumerate(langs_sorted):
    ax.text(i, n + 1, str(n), ha="center", fontsize=9)
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)

# Scale tiers
tiers_present = [t for t in TIER_ORDER if t in tier_counts] + \
                [t for t in tier_counts if t not in TIER_ORDER]
ax = axes[1]
colors = {"hobby": "C2", "startup": "C0", "growth": "C1", "enterprise": "C3"}
ax.bar(tiers_present,
       [tier_counts[t] for t in tiers_present],
       color=[colors.get(t, "gray") for t in tiers_present])
ax.set_ylabel("Commit pairs")
ax.set_title("(b) Scale tier distribution")
for i, t in enumerate(tiers_present):
    ax.text(i, tier_counts[t] + 1, str(tier_counts[t]), ha="center", fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "fig4_dataset.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig4_dataset.png", dpi=200, bbox_inches="tight")
print(f"Wrote {OUT/'fig4_dataset.pdf'}")
