#!/usr/bin/env python
"""Plot per-option visual-token overlap for MMBench (overlap_subset_mmbench.py output).

Headline metric: normalized_overlap = (observed pairwise top-k overlap - chance),
normalized. < 0 means answer options attend to LESS-shared visual tokens than
random selection would give (answer-dependent / "repulsive"); > 0 means shared.
"""
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

IN = sys.argv[1] if len(sys.argv) > 1 else "results_overlap_subset_mmbench_200.json"
OUT = os.path.splitext(IN)[0] + ".png"

# --- design-system palette (dataviz skill: diverging blue<->red, gray midpoint) ---
BLUE = "#2a78d6"   # below chance (answer-dependent)
RED  = "#e34948"   # above chance (shared)
INK        = "#0b0b0b"
INK_SECOND = "#52514e"
MUTED      = "#898781"
GRID       = "#e1e0d9"
SURFACE    = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_SECOND, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False,
})

d = json.load(open(IN))
no = np.array([x["normalized_overlap"] for x in d])
tasks = [x["task"] for x in d]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5),
                               gridspec_kw={"width_ratios": [1, 1.15]})

# ---- Panel A: distribution of normalized_overlap, colored by sign ----
bins = np.linspace(no.min(), no.max(), 26)
counts, edges = np.histogram(no, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2
colors = [BLUE if c < 0 else RED for c in centers]
axA.bar(centers, counts, width=np.diff(edges) * 0.9, color=colors,
        edgecolor=SURFACE, linewidth=0.8)
axA.axvline(0, color=INK, lw=1.5, ls="-")
axA.axvline(no.mean(), color=INK_SECOND, lw=1.5, ls="--")
axA.text(0, axA.get_ylim()[1] * 0.98, " chance", color=INK,
         va="top", ha="left", fontsize=9)
axA.text(no.mean(), axA.get_ylim()[1] * 0.88,
         f" mean = {no.mean():.3f} ", color=INK_SECOND, va="top",
         ha="right" if no.mean() < 0 else "left", fontsize=9)
frac_neg = (no < 0).mean()
axA.set_title("Per-question token overlap vs. chance", loc="left", weight="bold")
axA.set_xlabel("normalized overlap  (< 0 = fewer shared tokens than chance)")
axA.set_ylabel("questions")
axA.legend(handles=[Patch(color=BLUE, label=f"below chance ({frac_neg*100:.0f}%)"),
                    Patch(color=RED,  label=f"above chance ({(1-frac_neg)*100:.0f}%)")],
           frameon=False, loc="upper right", fontsize=9)

# ---- Panel B: normalized_overlap by task, sorted by mean ----
order = sorted(set(tasks), key=lambda t: np.mean([x["normalized_overlap"]
                                                  for x in d if x["task"] == t]))
rng = np.random.default_rng(0)
for i, t in enumerate(order):
    vals = np.array([x["normalized_overlap"] for x in d if x["task"] == t])
    jit = rng.uniform(-0.16, 0.16, size=len(vals))
    pcolors = [BLUE if v < 0 else RED for v in vals]
    axB.scatter(vals, i + jit, s=22, c=pcolors, alpha=0.55,
                edgecolors="none", zorder=2)
    axB.scatter(vals.mean(), i, marker="D", s=70, color=INK, zorder=3)
    axB.text(vals.mean(), i + 0.34, f"{vals.mean():.3f}", color=INK,
             ha="center", va="bottom", fontsize=8)

axB.axvline(0, color=INK, lw=1.5)
axB.set_yticks(range(len(order)))
axB.set_yticklabels([t.replace("finegrained_perception", "fg_perception")
                     for t in order], fontsize=9)
axB.set_ylim(-0.6, len(order) - 0.2)
axB.set_xlabel("normalized overlap")
axB.set_title("By task  (◆ = task mean)", loc="left", weight="bold")
axB.grid(axis="x", color=GRID, lw=0.8, zorder=0)

fig.suptitle(f"MMBench (dev, n={len(d)}) — do answer options select the same visual tokens?",
             x=0.01, ha="left", weight="bold", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
print("wrote", OUT)
