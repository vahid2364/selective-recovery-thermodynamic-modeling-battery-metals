"""
KDE plot of LHS input compositions (log10 scale) for all sulfate salts + NaOH dose.

Usage (run from repo root):
    python src/plot_lhs_kde.py                  # defaults to UP3
    UP_DIR=paper_SIT/Leach_recovery_UP2 python src/plot_lhs_kde.py

Outputs → results/<version>/fig_lhs_kde.png
          <UP_DIR>/lhs_input/kdePlot_all.png    (replaces in-place copy)
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── configure dataset ──────────────────────────────────────────────────────────
UP_DIR  = Path(os.environ.get("UP_DIR", "paper_SIT/Leach_recovery_UP3"))
VERSION = UP_DIR.name
OUTDIR  = Path("results") / VERSION
OUTDIR.mkdir(parents=True, exist_ok=True)

CSV = UP_DIR / "lhs_input" / "hydrolysis_then_NaOHmix_samples.csv"

print(f"Dataset : {UP_DIR}")
print(f"Output  : {OUTDIR}")

samples = pd.read_csv(CSV)

# ── display names ──────────────────────────────────────────────────────────────
label_map = {
    "Al_sulfate_mol": "Al$_2$(SO$_4$)$_3$",
    "Fe_sulfate_mol": "FeSO$_4$·7H$_2$O",
    "Cu_sulfate_mol": "CuSO$_4$",
    "Co_sulfate_mol": "CoSO$_4$·7H$_2$O",
    "Ni_sulfate_mol": "NiSO$_4$·6H$_2$O",
    "Mn_sulfate_mol": "MnSO$_4$·H$_2$O",
    "NaOH_mole":      "NaOH dose",
}

# column order: metals first (by increasing typical input conc), then NaOH
COL_ORDER = [
    "Fe_sulfate_mol",
    "Al_sulfate_mol",
    "Cu_sulfate_mol",
    "Ni_sulfate_mol",
    "Co_sulfate_mol",
    "Mn_sulfate_mol",
    "NaOH_mole",
]

# ── style cycle: Okabe-Ito + line style variants ───────────────────────────────
COLORS = [
    "#E69F00",   # orange      → Fe
    "#56B4E9",   # sky blue    → Al
    "#009E73",   # green       → Cu
    "#F0E442",   # yellow      → Ni
    "#0072B2",   # blue        → Co
    "#D55E00",   # vermillion  → Mn
    "#CC79A7",   # pink        → NaOH
]
LINESTYLES = ["-", "-", "-", "--", "--", "--", ":"]
LINEWIDTHS = [2.0, 2.0, 2.0, 2.2, 2.0, 2.0, 2.0]

# ── Nature-style rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          16,
    "axes.linewidth":     0.9,
    "xtick.major.width":  0.9,  "ytick.major.width":  0.9,
    "xtick.minor.width":  0.6,  "ytick.minor.width":  0.6,
    "xtick.direction":    "out", "ytick.direction":    "out",
    #"xtick.top":          True,  "ytick.right":        True,
})

fig, ax = plt.subplots(figsize=(7, 5.0))

for col, color, ls, lw in zip(COL_ORDER, COLORS, LINESTYLES, LINEWIDTHS):
    if col not in samples.columns:
        continue
    data = samples[col].dropna()
    #data = data[data > 0]
    if len(data) == 0:
        continue
    log_data = np.log10(data)
    sns.kdeplot(
        log_data,
        ax=ax,
        label=label_map.get(col, col),
        color=color,
        linestyle=ls,
        linewidth=lw,
        fill=False,
    )

ax.set_xlabel("log$_{10}$(Concentration / mol)", fontsize=18)
ax.set_ylabel("Density", fontsize=18)
ax.tick_params(axis="both", labelsize=11)
ax.legend(fontsize=12, frameon=True, handlelength=2.5,
          loc="upper left", bbox_to_anchor=(0.01, 0.99))
ax.minorticks_on()
ax.tick_params(axis='both', labelsize=16)

plt.tight_layout()

# save to results/ folder
out1 = OUTDIR / "fig_lhs_kde.png"
plt.savefig(out1, dpi=300, bbox_inches="tight")
print(f"Saved {out1}")

# also update the in-place copy used by the paper
out2 = UP_DIR / "lhs_input" / "kdePlot_all.png"
plt.savefig(out2, dpi=300, bbox_inches="tight")
print(f"Saved {out2}")

plt.close()
plt.rcParams.update(plt.rcParamsDefault)
