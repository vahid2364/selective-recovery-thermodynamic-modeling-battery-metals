"""
Combined SI vs pH figure:
  Main panel  : with phase equilibrium   (DOE_merged_mix2_eq_react.csv)
  Inset panel : without phase equilibrium (DOE_merged_mix2_react.csv)

Usage (run from repo root):
    python src/plot_SI_vs_pH_combined.py                  # defaults to UP2
    UP_DIR=paper_SIT/Leach_recovery_UP3 python src/plot_SI_vs_pH_combined.py

Outputs → results/<version>/fig_SI_vs_pH_combined.png
          <UP_DIR>/SI_index_calculation.png  (for paper reference)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
from pathlib import Path

# ── configure dataset ──────────────────────────────────────────────────────────
UP_DIR  = Path(os.environ.get("UP_DIR", "paper_SIT/Leach_recovery_UP2"))
VERSION = UP_DIR.name
BASE    = UP_DIR / "speciation_results_merged"
OUTDIR  = Path("results") / VERSION
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset : {UP_DIR}")
print(f"Output  : {OUTDIR}")

CSV_EQ   = BASE / "DOE_merged_mix2_eq_react.csv"
CSV_NOEQ = BASE / "DOE_merged_mix2_react.csv"

# ── colour palette (Okabe-Ito) ─────────────────────────────────────────────────
CBF = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#000000",  # black
    "#56B4E9",  # light blue
]

# ── phases to show (in order) ──────────────────────────────────────────────────
PHASE_DEFS = [
    ("FeAl$_2$O$_4$(s)",  "si_FeAl2O4"),
    ("Al(OH)$_3$(s)",      "si_Gibbsite"),
    ("Fe(OH)$_3$(s)",     "si_Fe(OH)3(s)"),
    ("Cu(OH)$_2$(s)",     "si_Cu(OH)2(s)"),
    ("Co(OH)$_2$(s)",     "si_Co(OH)2(s)"),
    ("Ni(OH)$_2$(s)",     "si_Ni(OH)2(s)"),
    ("Mn(OH)$_2$(s)",     "si_Mn(OH)2(s)"),
]


def find_col(df, candidates):
    """Case-insensitive column lookup."""
    low = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.strip().lower() in low:
            return low[cand.strip().lower()]
    return None


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def plot_si_panel(ax, df, ms, alpha, ph_col):
    """Plot SI vs pH: circles for SI<0, squares for SI>=0."""
    x = df[ph_col].astype(float).to_numpy()
    legend_entries, legend_labels = [], []

    for i, (label, si_col) in enumerate(PHASE_DEFS):
        col = find_col(df, [si_col, si_col + ".1"])
        if col is None:
            for c in df.columns:
                if si_col.lower().replace("si_", "") in c.lower():
                    col = c
                    break
        if col is None:
            continue

        y     = df[col].astype(float).to_numpy()
        color = CBF[i % len(CBF)]
        fin   = np.isfinite(y)

        neg    = fin & (y <  0)
        nonneg = fin & (y >= 0)

        h_neg = h_pos = None
        if neg.any():
            h_neg, = ax.plot(x[neg], y[neg], linestyle="None",
                             marker="o", ms=ms, color=color, alpha=alpha,
                             markeredgecolor="k", markeredgewidth=0.3)
        if nonneg.any():
            h_pos, = ax.plot(x[nonneg], y[nonneg], linestyle="None",
                             marker="s", ms=ms, color=color, alpha=alpha,
                             markeredgecolor="k", markeredgewidth=0.3)

        pair = tuple(h for h in [h_neg, h_pos] if h is not None)
        if pair:
            legend_entries.append(pair)
            legend_labels.append(label)

    return legend_entries, legend_labels


# ── load data ──────────────────────────────────────────────────────────────────
df_eq   = load(CSV_EQ)
df_noeq = load(CSV_NOEQ)

ph_eq   = find_col(df_eq,   ["pH", "ph"])
ph_noeq = find_col(df_noeq, ["pH", "ph"])

# ── main figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

handles, labels = plot_si_panel(ax, df_eq, ms=5, alpha=0.65, ph_col=ph_eq)

ax.axhline(0.0, ls="--", lw=1.2, c="k")
ax.set_yscale("symlog", linthresh=1)
ax.set_ylim(-1e3, 1e3)
ax.set_xlim(1, 14)
ax.set_xlabel("pH", fontsize=19.2)
ax.set_ylabel("Saturation Index (SI)", fontsize=19.2)
ax.tick_params(labelsize=16.8)

ax.text(1.1,  0.18, "supersaturation", fontsize=13, color="k", va="bottom")
ax.text(1.1, -0.28, "undersaturation", fontsize=13, color="k", va="top")

# phase legend — lower right
phase_leg = ax.legend(handles, labels, title="Phases", fontsize=12, title_fontsize=12,
                      handler_map={tuple: HandlerTuple(ndivide=None)},
                      frameon=True, loc="lower right")
ax.add_artist(phase_leg)

# marker-meaning legend — bottom left
mk_neg = mlines.Line2D([], [], color="gray", marker="o", linestyle="None", ms=7,
                        markeredgecolor="k", markeredgewidth=0.5, label="SI < 0")
mk_pos = mlines.Line2D([], [], color="gray", marker="s", linestyle="None", ms=7,
                        markeredgecolor="k", markeredgewidth=0.5, label="SI ≥ 0")
ax.legend(handles=[mk_neg, mk_pos], fontsize=12, frameon=True, loc="lower left")

ax.text(0.02, 0.97, "With phase equilibrium",
        transform=ax.transAxes, fontsize=14,
        color="#0072B2", fontweight="bold", va="top")

# ── inset ──────────────────────────────────────────────────────────────────────
ax_in = ax.inset_axes([0.480, 0.571, 0.520, 0.429])

h_in, l_in = plot_si_panel(ax_in, df_noeq, ms=3.5, alpha=0.6, ph_col=ph_noeq)

ax_in.axhline(0.0, ls="--", lw=1.0, c="k")
ax_in.set_yscale("symlog", linthresh=1)
ax_in.set_xlim(2.5, 14)
ax_in.set_xlabel("pH", fontsize=12)
ax_in.set_ylabel("Saturation Index (SI)", fontsize=10.8)
ax_in.tick_params(labelsize=9.6)

# legends removed from inset — shared with main axes

ax_in.set_title("Without phase equilibrium", fontsize=11,
                color="#0072B2", fontweight="bold", loc="left")

ax_in.patch.set_edgecolor("black")
ax_in.patch.set_linewidth(1.5)
for spine in ax_in.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.5)

# ── save ───────────────────────────────────────────────────────────────────────
plt.tight_layout()
out = OUTDIR / "fig_SI_vs_pH_combined.png"
fig.savefig(out, dpi=220, bbox_inches="tight")

# also copy to UP_DIR so the paper reference stays current
si_out = UP_DIR / "SI_index_calculation.png"
fig.savefig(si_out, dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
print(f"Saved {si_out}")
