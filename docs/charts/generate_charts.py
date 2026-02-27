"""
Generate 3 paper-quality comparison charts:
  Chart 1 — Relevance metrics: Baseline vs Re-ranking (MPD test set)
  Chart 2 — Beyond-accuracy metrics + cross-domain ILD comparison
  Chart 3 — Sensitivity analysis: NDCG@10 and ILD as a function of λ

Output: docs/charts/chart1_relevance.png
        docs/charts/chart2_diversity.png
        docs/charts/chart3_sensitivity.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────────────────
C_BASE  = "#4C72B0"   # blue  — baseline
C_RERANK = "#DD8452"  # orange — re-ranking

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Relevance Metrics: Baseline vs Re-ranking (MPD, 59,071 playlists)
# ══════════════════════════════════════════════════════════════════════════════

metrics    = ["NDCG@10", "Precision@10", "Recall@10", "MRR@10", "HIT@10", "R-Precision"]
baseline   = [0.3851,    0.1898,         0.0413,       0.3436,   0.1906,   0.0529]
reranking  = [0.3697,    0.1666,         0.0360,       0.3347,   0.1674,   0.0472]

x = np.arange(len(metrics))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars_b = ax.bar(x - w/2, baseline,  w, color=C_BASE,   label="Baseline",    zorder=3)
bars_r = ax.bar(x + w/2, reranking, w, color=C_RERANK, label="Re-ranking",  zorder=3)

# Δ annotations above the re-ranking bars
for i, (b, r) in enumerate(zip(baseline, reranking)):
    delta = r - b
    sign  = "−" if delta < 0 else "+"
    ax.text(x[i] + w/2, r + 0.005, f"{sign}{abs(delta):.4f}",
            ha="center", va="bottom", fontsize=8, color="#AA0000")

ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=15, ha="right")
ax.set_ylabel("Score")
ax.set_title("Chart 1 — Relevance Metrics: Baseline vs Re-ranking\n"
             "(MPD test set, 59,071 playlists, α=0.41, β=0.57, γ=0.01)")
ax.set_ylim(0, 0.48)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right")

fig.tight_layout()
path1 = OUT_DIR / "chart1_relevance.png"
fig.savefig(path1)
plt.close(fig)
print(f"Saved → {path1}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Beyond-Accuracy Metrics + Cross-Domain ILD
# ══════════════════════════════════════════════════════════════════════════════
# Two sub-plots side by side:
#   Left : ILD and Genre-Diversity change on MPD (grouped bars)
#   Right: ILD comparison across MPD and ThirtyMusic (cross-domain)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

# ── Left: MPD beyond-accuracy metrics ─────────────────────────────────────────
ba_metrics   = ["ILD@10", "Genre Diversity"]
ba_baseline  = [0.6051,   15.57]
ba_reranking = [0.8255,   17.24]

# Normalize Genre Diversity to same 0-1 scale for visual clarity
# (show raw values via y-axis ticks instead; use secondary axis trick with twin)
x2 = np.arange(len(ba_metrics))
w2 = 0.35

bars_b2 = ax_left.bar(x2 - w2/2, ba_baseline,  w2, color=C_BASE,   label="Baseline",   zorder=3)
bars_r2 = ax_left.bar(x2 + w2/2, ba_reranking, w2, color=C_RERANK, label="Re-ranking", zorder=3)

for i, (b, r) in enumerate(zip(ba_baseline, ba_reranking)):
    pct = (r - b) / b * 100
    ax_left.text(x2[i] + w2/2, r + 0.3, f"+{pct:.0f}%",
                 ha="center", va="bottom", fontsize=9.5, color="#006600", fontweight="bold")

ax_left.set_xticks(x2)
ax_left.set_xticklabels(ba_metrics)
ax_left.set_ylabel("Score / Count")
ax_left.set_title("Beyond-Accuracy Metrics (MPD)\nILD and Genre Diversity")
ax_left.set_ylim(0, 22)
ax_left.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax_left.set_axisbelow(True)
ax_left.legend(loc="upper left")

# ── Right: Cross-domain ILD comparison ────────────────────────────────────────
datasets      = ["MPD\n(in-domain)", "ThirtyMusic\n(cross-domain)"]
ild_base_cd   = [0.6051, 0.6069]
ild_impr_cd   = [0.8255, 0.8344]
ild_delta_cd  = [0.2204, 0.2275]

x3 = np.arange(len(datasets))
bars_b3 = ax_right.bar(x3 - w2/2, ild_base_cd, w2, color=C_BASE,   label="Baseline",   zorder=3)
bars_r3 = ax_right.bar(x3 + w2/2, ild_impr_cd, w2, color=C_RERANK, label="Re-ranking", zorder=3)

for i, (b, r, d) in enumerate(zip(ild_base_cd, ild_impr_cd, ild_delta_cd)):
    pct = d / b * 100
    ax_right.text(x3[i] + w2/2, r + 0.008, f"+{pct:.0f}%",
                  ha="center", va="bottom", fontsize=9.5, color="#006600", fontweight="bold")

ax_right.set_xticks(x3)
ax_right.set_xticklabels(datasets)
ax_right.set_ylabel("Intra-List Diversity (ILD@10)")
ax_right.set_title("Cross-Domain ILD Comparison\nMPD vs ThirtyMusic")
ax_right.set_ylim(0, 1.05)
ax_right.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax_right.set_axisbelow(True)
ax_right.legend(loc="upper left")

fig.suptitle("Chart 2 — Diversity Gains: In-Domain and Cross-Domain",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
path2 = OUT_DIR / "chart2_diversity.png"
fig.savefig(path2)
plt.close(fig)
print(f"Saved → {path2}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Sensitivity Analysis: NDCG@10 and ILD vs λ
# ══════════════════════════════════════════════════════════════════════════════

lambdas  = [0.0,    0.1,    0.2,    0.3]
ndcg_vals = [0.3851, 0.3784, 0.3701, 0.3701]
ild_vals  = [0.6051, 0.7503, 0.8221, 0.8221]

fig, ax1 = plt.subplots(figsize=(7, 5))

color_ndcg = C_BASE
color_ild  = C_RERANK

l1, = ax1.plot(lambdas, ndcg_vals, "o-", color=color_ndcg, lw=2.2, ms=8, label="NDCG@10 (left axis)")
ax1.set_xlabel("λ  (diversity incentive in objective)")
ax1.set_ylabel("NDCG@10", color=color_ndcg)
ax1.tick_params(axis="y", labelcolor=color_ndcg)
ax1.set_ylim(0.34, 0.42)
ax1.axhline(ndcg_vals[0], color=color_ndcg, lw=1, ls="--", alpha=0.5)

ax2 = ax1.twinx()
l2, = ax2.plot(lambdas, ild_vals, "s-", color=color_ild, lw=2.2, ms=8, label="ILD@10 (right axis)")
ax2.set_ylabel("Intra-List Diversity (ILD@10)", color=color_ild)
ax2.tick_params(axis="y", labelcolor=color_ild)
ax2.set_ylim(0.55, 0.90)
ax2.axhline(ild_vals[0], color=color_ild, lw=1, ls="--", alpha=0.5)

# Annotate the selected operating point
ax1.axvline(0.2, color="grey", lw=1.2, ls=":", zorder=0)
ax1.text(0.21, 0.354, "λ = 0.2\n(selected)", color="grey", fontsize=9)

# Annotate plateau
ax2.annotate("plateau", xy=(0.3, ild_vals[3]), xytext=(0.27, 0.845),
             fontsize=9, color=color_ild,
             arrowprops=dict(arrowstyle="->", color=color_ild, lw=1))

ax1.set_xticks(lambdas)
ax1.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax1.set_axisbelow(True)

lines  = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center right")

ax1.set_title("Chart 3 — Sensitivity Analysis: Effect of λ on NDCG@10 and ILD@10\n"
              "(MPD test set, 59,071 playlists)")

fig.tight_layout()
path3 = OUT_DIR / "chart3_sensitivity.png"
fig.savefig(path3)
plt.close(fig)
print(f"Saved → {path3}")

print("\nAll 3 charts saved to docs/charts/")
