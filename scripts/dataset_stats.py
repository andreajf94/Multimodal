"""Generate per-repo dataset stats and scale-tier distribution (Appendices B & H)."""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("data/commit_pairs_production")
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
for name, r in sorted(by_repo.items()):
    avg_man = sum(r["manifest"]) / len(r["manifest"])
    print(f"{name:<18}{r['prs']:>5}{r['loc']:>10}{r['stars']:>9}{r['contrib']:>9}  {r['lang']:<12}{r['tier']:<10}{avg_man:>14.0f}")
    total_prs += r["prs"]
    tier_counts[r["tier"]] += r["prs"]

print("-" * 90)
print(f"{'TOTAL':<18}{total_prs:>5}   ({len(by_repo)} repos)")
print()
print("Scale-tier distribution (by PRs):")
for tier, n in sorted(tier_counts.items(), key=lambda x: -x[1]):
    print(f"  {tier:<12}{n:>5}  ({n/total_prs*100:.1f}%)")

# write LaTeX-ready CSV
with open("figures/dataset_stats.csv", "w") as f:
    f.write("repo,prs,loc,stars,contributors,language,tier,avg_manifest\n")
    for name, r in sorted(by_repo.items()):
        avg_man = sum(r["manifest"]) / len(r["manifest"])
        f.write(f"{name},{r['prs']},{r['loc']},{r['stars']},{r['contrib']},{r['lang']},{r['tier']},{avg_man:.0f}\n")
print("\nWrote figures/dataset_stats.csv")
