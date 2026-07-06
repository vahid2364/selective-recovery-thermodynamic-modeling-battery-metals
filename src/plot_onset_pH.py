"""
Precipitation onset pH distributions and selectivity windows from PHREEQC output.

Two datasets:
  - DOE_merged_mix2_react.csv    : phase equilibrium SUPPRESSED (SI builds freely)
  - DOE_merged_mix2_eq_react.csv : phase equilibrium ENFORCED   (SI pinned at 0)

Usage (run from repo root):
    python src/plot_onset_pH.py                  # defaults to UP2
    UP_DIR=paper_SIT/Leach_recovery_UP3 python src/plot_onset_pH.py

Outputs → results/<version>/fig8_*.png, fig9_*.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── configure dataset ──────────────────────────────────────────────────────────
UP_DIR  = Path(os.environ.get("UP_DIR", "paper_SIT/Leach_recovery_UP3"))
VERSION = UP_DIR.name                                   # e.g. "Leach_recovery_UP2"
BASE    = UP_DIR / "speciation_results_merged"
OUTDIR  = Path("results") / VERSION
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset : {UP_DIR}")
print(f"Output  : {OUTDIR}")

# ── datasets ───────────────────────────────────────────────────────────────────
DATASETS = {
    "noeq": {
        "file":      BASE / "DOE_merged_mix2_react.csv",
        "threshold": 0.0,
        "label":     "Without phase equilibrium (SI ≥ 0)",
    },
    "eq": {
        "file":      BASE / "DOE_merged_mix2_eq_react.csv",
        "threshold": -0.05,
        "label":     "With phase equilibrium (SI ≥ −0.05)",
    },
}

SI_MAP = {
    "Fe(OH)$_3$(s)":  "si_Fe(OH)3(s)",
    "Al(OH)$_3$(s)":  "si_Gibbsite",
    "Cu(OH)$_2$(s)":  "si_Cu(OH)2(s)",
    "Ni(OH)$_2$(s)":  "si_Ni(OH)2(s)",
    "Co(OH)$_2$(s)":  "si_Co(OH)2(s)",
    "Mn(OH)$_2$(s)":  "si_Mn(OH)2(s)",
}

COLORS = {
    "Fe(OH)$_3$(s)":  "#1a8c1a",
    "Al(OH)$_3$(s)":  "#e86e00",
    "Cu(OH)$_2$(s)":  "#6b3a2a",
    "Ni(OH)$_2$(s)":  "#9aaa00",
    "Co(OH)$_2$(s)":  "#cc44a0",
    "Mn(OH)$_2$(s)":  "#111111",
}

# Ordered by median onset pH (ascending) so ΔpH gaps are always positive
# and bracket annotations cover every consecutive pair.
# Al(OH)₃(s) (3.78) → Fe (5.48) → Cu (5.62) → Co (7.67) → Ni (8.01) → Mn (10.06)
PHASE_ORDER = [
    "Al(OH)$_3$(s)",
    "Fe(OH)$_3$(s)",
    "Cu(OH)$_2$(s)",
    "Co(OH)$_2$(s)",
    "Ni(OH)$_2$(s)",
    "Mn(OH)$_2$(s)",
]

FIG_SIZE = (5.5, 4.2)
FS_TICK  = 13
FS_LABEL = 14
FS_ANNOT = 11
FS_LEGEND= 11
FS_TITLE = 11


def derive_onset(df, threshold):
    onset = {}
    for label, col in SI_MAP.items():
        if col in df.columns:
            onset[label] = df.loc[df[col] >= threshold, "pH"].values
        else:
            onset[label] = np.array([])
    return onset


def plot_boxplot(onset, title, outpath):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    data = [onset[p] for p in PHASE_ORDER]
    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.4),
                    capprops=dict(linewidth=1.4),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, phase in zip(bp["boxes"], PHASE_ORDER):
        patch.set_facecolor(COLORS[phase])
        patch.set_alpha(0.78)
    ax.set_yticks(range(1, len(PHASE_ORDER) + 1))
    ax.set_yticklabels(PHASE_ORDER, fontsize=FS_TICK)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlabel("Precipitation onset pH", fontsize=FS_LABEL)
    ax.axvline(x=7, color="gray", linestyle="--", linewidth=0.9, alpha=0.6, label="pH = 7")
    ax.set_xlim(2, 14)
    ax.legend(fontsize=FS_LEGEND)
    ax.set_title(title, fontsize=FS_TITLE)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}")


def plot_selectivity(onset, title, outpath):
    phases_ok = [p for p in PHASE_ORDER if len(onset[p]) > 0]
    medians = {p: np.median(onset[p]) for p in phases_ok}
    q25     = {p: np.percentile(onset[p], 25) for p in phases_ok}
    q75     = {p: np.percentile(onset[p], 75) for p in phases_ok}

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    y_pos = np.arange(len(phases_ok))

    for i, phase in enumerate(phases_ok):
        ax.barh(y_pos[i], q75[phase] - q25[phase], left=q25[phase],
                height=0.52, color=COLORS[phase], alpha=0.78,
                edgecolor="black", linewidth=0.7)
        ax.plot(medians[phase], y_pos[i], "k|", markersize=14, markeredgewidth=2.8)

    for i in range(len(phases_ok) - 1):
        g_lo  = medians[phases_ok[i]]
        g_hi  = medians[phases_ok[i + 1]]
        gap   = g_hi - g_lo
        mid   = (g_lo + g_hi) / 2
        y_mid = (y_pos[i] + y_pos[i + 1]) / 2
        if gap > 0.3:
            ax.annotate(f"$\\Delta$pH = {gap:.1f}",
                        xy=(mid, y_mid), fontsize=FS_ANNOT,
                        ha="center", va="center", color="dimgray",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(phases_ok, fontsize=FS_TICK)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlabel("pH", fontsize=FS_LABEL)
    ax.set_xlim(2, 14)
    ax.axvline(x=7, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_title(title, fontsize=FS_TITLE)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}")


def plot_combined(onset, title, outpath):
    """
    Single boxplot with ΔpH gap annotations between consecutive medians.
    Eliminates the redundant IQR-only panel — the boxplot already contains
    Q1/Q3; the only unique content of the selectivity window was the ΔpH labels,
    which are now drawn directly on the boxplot as bracketed annotations.
    Original fig8 and fig9 outputs are unchanged.
    """
    phases_ok = [p for p in PHASE_ORDER if len(onset[p]) > 0]
    medians = {p: np.median(onset[p]) for p in phases_ok}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    data = [onset[p] for p in phases_ok]
    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.4),
                    capprops=dict(linewidth=1.4),
                    flierprops=dict(marker="o", markersize=3, alpha=0.35))
    for patch, phase in zip(bp["boxes"], phases_ok):
        patch.set_facecolor(COLORS[phase])
        patch.set_alpha(0.78)

    ax.set_yticks(range(1, len(phases_ok) + 1))
    ax.set_yticklabels(phases_ok, fontsize=FS_TICK)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlabel("Precipitation onset pH", fontsize=FS_LABEL)
    ax.axvline(x=7, color="gray", linestyle="--", linewidth=0.9,
               alpha=0.6, label="pH = 7")
    ax.set_xlim(2, 16)
    ax.legend(fontsize=FS_LEGEND, loc="upper left")

    # ── ΔpH bracket annotations between ALL consecutive medians ──────────────
    # PHASE_ORDER is sorted by median onset pH so all gaps are positive.
    y_positions = list(range(1, len(phases_ok) + 1))

    # stagger bracket x-positions so adjacent labels don't collide
    x_base   = 13.6
    x_step   = 0.55
    n_gaps   = len(phases_ok) - 1

    for i in range(n_gaps):
        gap  = medians[phases_ok[i + 1]] - medians[phases_ok[i]]
        y_lo = y_positions[i]
        y_hi = y_positions[i + 1]
        y_mid = (y_lo + y_hi) / 2
        x_br  = x_base + (i % 2) * x_step   # alternate columns for readability

        ax.annotate("", xy=(x_br, y_hi), xytext=(x_br, y_lo),
                    arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.2),
                    annotation_clip=False)
        ax.text(x_br + 0.12, y_mid, f"$\\Delta$pH\n={gap:.1f}",
                fontsize=FS_ANNOT - 2, va="center", color="dimgray",
                clip_on=False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}")


def print_summary(tag, onset):
    print(f"\n--- {tag} ---")
    for p in PHASE_ORDER:
        vals = onset[p]
        if len(vals):
            print(f"  {p:25s}  n={len(vals):4d}  "
                  f"median={np.median(vals):.2f}  "
                  f"Q1={np.percentile(vals,25):.2f}  "
                  f"Q3={np.percentile(vals,75):.2f}")
        else:
            print(f"  {p:25s}  no supersaturated samples")


# ── run ────────────────────────────────────────────────────────────────────────
for tag, cfg in DATASETS.items():
    print(f"\nProcessing: {tag}")
    df    = pd.read_csv(cfg["file"])
    onset = derive_onset(df, cfg["threshold"])
    print_summary(tag, onset)
    plot_boxplot(
        onset,
        title=f"Precipitation onset pH distribution ({cfg['label']})",
        outpath=OUTDIR / f"fig8_onset_pH_boxplot_{tag}.png",
    )
    plot_selectivity(
        onset,
        title=f"Selective precipitation windows ({cfg['label']})",
        outpath=OUTDIR / f"fig9_selectivity_windows_{tag}.png",
    )
    plot_combined(
        onset,
        title=f"Precipitation onset & selectivity windows ({cfg['label']})",
        outpath=OUTDIR / f"fig10_combined_onset_selectivity_{tag}.png",
    )

print("\nDone.")
