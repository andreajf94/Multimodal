"""Paper Fig 3: structural hierarchy of the multi-component reward function.

Shows the 3-tier format gate that prevents reward signal collapse:
  Tier 1 (gate):  parse JSON  ──> if fail: only format_partial (0.25)
  Tier 2 (form):  schema OK   ──> +format_compliance (0.50) +structural_quality (0.50)
  Tier 3 (content): grounded   ──> +existing_file (1.5) +created_file (1.5) +semantic (3.0)

Total max = 7.25.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("figures"); OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

def box(x, y, w, h, text, color, fc="white", fontsize=10, weight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                       linewidth=1.6, edgecolor=color, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color="black", weight=weight)

def arrow(x1, y1, x2, y2, color="gray", style="-|>", label=None, lx=None, ly=None):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=14, color=color, lw=1.4)
    ax.add_patch(a)
    if label:
        ax.text(lx, ly, label, fontsize=8.5, color=color, ha="center")

# Top: model output
box(3.5, 6.0, 3.0, 0.7, "Model output (raw text)", "C0", fc="#eef4ff", weight="bold")

# Tier 1 — Gate
arrow(5.0, 6.0, 5.0, 5.45)
box(3.5, 4.8, 3.0, 0.65, "Tier 1: JSON parse gate", "#444", fc="#f4f4f4", weight="bold")

# Branch left: parse fail
arrow(3.5, 5.12, 1.6, 4.4, color="C3", label="parse fails", lx=2.0, ly=4.85)
box(0.2, 3.7, 2.8, 0.7, "format_partial\n0.25", "C3", fc="#ffecec", fontsize=10)
arrow(1.6, 3.7, 1.6, 0.65, color="C3")
ax.text(1.6, 0.35, "STOP — gradient floor", fontsize=8.5, color="C3", ha="center")

# Branch right: parse OK
arrow(6.5, 5.12, 8.0, 4.55, color="C2", label="parse OK", lx=7.5, ly=4.95)

# Tier 2 — Form (schema/structure)
box(6.7, 3.8, 3.1, 0.75, "Tier 2: Schema + structural", "#1a7", fc="#eaffea", weight="bold")
arrow(8.25, 3.80, 8.25, 3.30, color="#1a7")
box(6.7, 2.55, 3.1, 0.75, "format_compliance  0.50\nstructural_quality  0.50", "#1a7",
    fc="#f3fff3", fontsize=9)

# Tier 3 — Content (grounding)
arrow(8.25, 2.55, 8.25, 2.05, color="C0")
box(6.7, 1.30, 3.1, 0.75, "Tier 3: Grounded content", "C0", fc="#eef4ff", weight="bold")
arrow(8.25, 1.30, 8.25, 0.80, color="C0")
box(6.5, 0.05, 3.4, 0.75,
    "existing_file 1.5  ·  created_file 1.5\nsemantic_similarity 3.0",
    "C0", fc="#f7faff", fontsize=9)

# Total tag
ax.text(5.0, 0.3, "Σ  total reward, max = 7.25",
        ha="center", fontsize=11, weight="bold", color="#222",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff7d6",
                  edgecolor="#caa600", lw=1.2))

ax.set_title("Multi-component reward: 3-tier format gate prevents signal collapse",
             fontsize=12, weight="bold", pad=10)

fig.tight_layout()
fig.savefig(OUT / "fig3_reward_hierarchy.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig3_reward_hierarchy.png", dpi=200, bbox_inches="tight")
print(f"Wrote {OUT/'fig3_reward_hierarchy.pdf'}")
