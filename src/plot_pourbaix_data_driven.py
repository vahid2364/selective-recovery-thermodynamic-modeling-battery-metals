"""
Data-driven Pourbaix diagrams (pH vs pe) for six metals.
Points coloured by SI value of the primary solid phase for each metal.
Diverging colormap centred at SI=0: blue = undersaturated, red = supersaturated.

Usage (run from repo root):
    python src/plot_pourbaix_data_driven.py                  # defaults to UP2
    UP_DIR=paper_SIT/Leach_recovery_UP3 python src/plot_pourbaix_data_driven.py

Outputs → results/<version>/fig_pourbaix_data_driven.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ── configure dataset ──────────────────────────────────────────────────────────
UP_DIR  = Path(os.environ.get("UP_DIR", "paper_SIT/Leach_recovery_UP2"))
VERSION = UP_DIR.name
BASE    = UP_DIR / "speciation_results_merged"
OUTDIR  = Path("results") / VERSION
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset : {UP_DIR}")
print(f"Output  : {OUTDIR}")

CSV = BASE / "DOE_merged_mix2_eq_react.csv"

# ── metal → SI column & display label ─────────────────────────────────────────
METALS = [
    ("Al",  "si_Gibbsite",      "Al",  "Gibbsite"),
    ("Fe",  "si_Fe(OH)3(s)",    "Fe",  "Fe(OH)₃(s)"),
    ("Cu",  "si_Cu(OH)2(s)",    "Cu",  "Cu(OH)₂(s)"),
    ("Ni",  "si_Ni(OH)2(s)",    "Ni",  "Ni(OH)₂(s)"),
    ("Co",  "si_Co(OH)2(s)",    "Co",  "Co(OH)₂(s)"),
    ("Mn",  "si_Mn(OH)2(s)",    "Mn",  "Mn(OH)₂(s)"),
]

METAL_COLORS = {
    "Al": "#e86e00",
    "Fe": "#1a8c1a",
    "Cu": "#6b3a2a",
    "Ni": "#9aaa00",
    "Co": "#cc44a0",
    "Mn": "#111111",
}

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df.columns = df.columns.str.strip()

ph = df["pH"].astype(float).to_numpy()
pe = df["pe"].astype(float).to_numpy()

# ── figure layout ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=False, sharey=False)
fig.suptitle("Data-driven Pourbaix diagrams (phase-equilibrium enforced)",
             fontsize=14, fontweight="bold", y=1.01)

SI_CLIP = 15

for ax, (metal, si_col, symbol, solid_label) in zip(axes.flat, METALS):

    si = df[si_col].astype(float).to_numpy() if si_col in df.columns else np.full(len(ph), np.nan)
    si_clipped = np.clip(si, -SI_CLIP, SI_CLIP)

    norm  = mcolors.TwoSlopeNorm(vmin=-SI_CLIP, vcenter=0, vmax=SI_CLIP)
    cmap  = plt.cm.RdBu_r

    sc = ax.scatter(ph, pe, c=si_clipped, cmap=cmap, norm=norm,
                    s=8, alpha=0.6, linewidths=0)

    sup_mask = np.isfinite(si) & (si >= 0)
    if sup_mask.any():
        ax.scatter(ph[sup_mask], pe[sup_mask], c="none",
                   edgecolors="k", linewidths=0.3, s=10, alpha=0.4, zorder=2)

    pH_line = np.linspace(0, 14, 200)
    ax.plot(pH_line, -pH_line + 20.75, "k--", lw=0.8, alpha=0.5, label="O₂/H₂O")
    ax.plot(pH_line, -pH_line,         "k:",  lw=0.8, alpha=0.5, label="H₂/H₂O")

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("SI", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_title(f"{symbol}  [{solid_label}]",
                 fontsize=12, fontweight="bold",
                 color=METAL_COLORS[metal])
    ax.set_xlabel("pH", fontsize=11)
    ax.set_ylabel("pe", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.set_xlim(1, 14)

    ax.text(0.03, 0.97, "SI ≥ 0 : supersaturated\n(black edge)",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

plt.tight_layout()
out = OUTDIR / "fig_pourbaix_data_driven.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
