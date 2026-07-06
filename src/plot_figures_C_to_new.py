"""
Generate four figure sets from PHREEQC output CSVs:
  C  - Aqueous speciation transformation
  N1 - Co-precipitation risk matrix
  N2 - Recovery curves with uncertainty bands
  N3 - Free ion activity suppression profile

Usage (run from repo root):
    python src/plot_figures_C_to_new.py                  # defaults to UP2
    UP_DIR=paper_SIT/Leach_recovery_UP3 python src/plot_figures_C_to_new.py

Outputs → results/<version>/figC_*.png, figN1_*.png, figN2_*.png, figN3_*.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── configure dataset ──────────────────────────────────────────────────────────
UP_DIR  = Path(os.environ.get("UP_DIR", "paper_SIT/Leach_recovery_UP3"))
VERSION = UP_DIR.name
BASE    = UP_DIR / "speciation_results_merged"
OUTDIR  = Path("results") / VERSION
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset : {UP_DIR}")
print(f"Output  : {OUTDIR}")

NOEQ = BASE / "DOE_merged_mix2_react.csv"
EQ   = BASE / "DOE_merged_mix2_eq_react.csv"

_COL_ALIASES = {"m_Fe+2": "m_Fe2+", "m_Fe+3": "m_Fe3+",
                "a_Fe+2": "a_Fe2+", "a_Fe+3": "a_Fe3+"}

def _load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Only rename alias → canonical when the canonical name doesn't already exist
    # (avoids duplicate columns in datasets that carry both forms)
    rename_map = {k: v for k, v in _COL_ALIASES.items()
                  if k in df.columns and v not in df.columns}
    df.rename(columns=rename_map, inplace=True)
    df.replace(1e-99,    np.nan, inplace=True)
    df.replace(-999.999, np.nan, inplace=True)
    return df

df_noeq = _load(NOEQ)
df_eq   = _load(EQ)

SI_COLS = {
    "Fe(OH)$_3$": "si_Fe(OH)3(s)",
    "Gibbsite":   "si_Gibbsite",
    "Cu(OH)$_2$": "si_Cu(OH)2(s)",
    "Ni(OH)$_2$": "si_Ni(OH)2(s)",
    "Co(OH)$_2$": "si_Co(OH)2(s)",
    "Mn(OH)$_2$": "si_Mn(OH)2(s)",
}

METAL_COLORS = {
    "Fe": "#E69F00",  # orange        ┐
    "Al": "#56B4E9",  # sky blue      │
    "Cu": "#009E73",  # bluish green  │ Okabe-Ito
    "Ni": "#F0E442",  # yellow        │ colorblind-safe
    "Co": "#0072B2",  # blue          │
    "Mn": "#D55E00",  # vermillion    ┘
}

# ── C: Aqueous speciation ──────────────────────────────────────────────────────
SPECIES_MAP = {
    "Al": {"Free Al$^{3+}$": "m_Al+3", "AlOH$^{2+}$": "m_AlOH2+",
           "Al(OH)$_2^+$": "m_Al(OH)2+",
           "AlSO$_4^+$": "m_AlSO4+", "Al(SO$_4$)$_2^-$": "m_Al(SO4)2-",
           "AlO$_2^-$": "m_AlO2-", "Polymeric Al": "m_Al13O4(OH)24+7"},
    "Fe": {"Free Fe$^{2+}$": "m_Fe2+", "Free Fe$^{3+}$": "m_Fe3+",
           "FeSO$_4$": "m_FeSO4", "FeSO$_4^+$": "m_FeSO4+",
           "FeOH$^{2+}$": "m_FeOH+", "Fe(OH)$_2$": "m_Fe(OH)2",
           "Fe(OH)$_3^-$": "m_Fe(OH)3-", "Fe(OH)$_4^-$": "m_Fe(OH)4-"},
    "Cu": {"Free Cu$^{2+}$": "m_Cu+2", "Free Cu$^+$": "m_Cu+",
           "CuOH$^+$": "m_CuOH+", "CuSO$_4$": "m_CuSO4"},
    "Ni": {"Free Ni$^{2+}$": "m_Ni+2", "NiOH$^+$": "m_NiOH+",
           "NiSO$_4$": "m_NiSO4",
           "Ni(OH)$_2$": "m_Ni(OH)2", "Ni(OH)$_3^-$": "m_Ni(OH)3-",
           "Ni$_4$(OH)$_4^{4+}$": "m_Ni4(OH)4+4"},
    "Co": {"Free Co$^{2+}$": "m_Co+2", "CoOH$^+$": "m_CoOH+",
           "CoSO$_4$": "m_CoSO4", "Co(OH)$_4^{2-}$": "m_Co(OH)4-2"},
    "Mn": {"Free Mn$^{2+}$": "m_Mn+2", "MnOH$^+$": "m_MnOH+",
           "MnSO$_4$": "m_MnSO4",
           "Mn(OH)$_2$": "m_Mn(OH)2", "Mn(OH)$_3^-$": "m_Mn(OH)3-",
           "Mn(OH)$_4^{2-}$": "m_Mn(OH)4-2",
           "Mn$_2$(OH)$_3^+$": "m_Mn2(OH)3+"},
}

# Stoichiometric factors: moles of metal per mole of species
# (only needed for polynuclear species; all others default to 1)
SPECIES_STOICH = {
    "m_Ni4(OH)4+4":    4,
    "m_Al13O4(OH)24+7": 13,
    "m_Al2(OH)2+4":    2,
    "m_Al3(OH)4+5":    3,
    "m_Fe2(OH)2+4":    2,
    "m_Fe3(OH)4+5":    3,
    "m_Mn2(OH)3+":     2,
}

# Total dissolved metal column from PHREEQC -totals flag
TOTALS_COL = {"Al": "Al", "Fe": "Fe", "Cu": "Cu",
              "Ni": "Ni", "Co": "Co", "Mn": "Mn"}

SPEC_PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#bcbd22"]


def plot_speciation(df, tag):
    metals  = list(SPECIES_MAP.keys())
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=False)
    axes = axes.flatten()
    pH_min = np.floor(df["pH"].min() * 10) / 10    # round down to 1 d.p.
    pH_max = np.ceil(df["pH"].max()  * 10) / 10    # round up   to 1 d.p.
    bin_width = 0.15                                 # refined binning (was 0.25)
    pH_bins = np.arange(pH_min, pH_max + bin_width, bin_width)
    pH_mid  = 0.5 * (pH_bins[:-1] + pH_bins[1:])

    for ax, metal in zip(axes, metals):
        sp = SPECIES_MAP[metal]
        valid = [(l, c) for l, c in sp.items() if c in df.columns]
        if not valid:
            continue
        labels, cols = zip(*valid)
        fracs = []
        for lo, hi in zip(pH_bins[:-1], pH_bins[1:]):
            sub = df[(df["pH"] >= lo) & (df["pH"] < hi)][list(cols)].fillna(0)
            if len(sub) == 0:
                fracs.append([np.nan] * len(cols)); continue
            totals = sub.sum(axis=1)
            fracs.append((sub.div(totals, axis=0)).mean().values)
        fracs  = np.array(fracs)

        # only plot/label species that contribute non-negligible fractions
        has_data = np.nanmax(fracs, axis=0) > 0.005

        bottom = np.zeros(len(pH_mid))
        color_idx = 0
        for i, label in enumerate(labels):
            if not has_data[i]:
                continue
            ax.fill_between(pH_mid, bottom, bottom + np.nan_to_num(fracs[:, i]),
                            label=label, color=SPEC_PALETTE[color_idx % len(SPEC_PALETTE)],
                            alpha=0.82, step="mid")
            bottom += np.nan_to_num(fracs[:, i])
            color_idx += 1
        ax.set_xlim(pH_min, 10); ax.set_ylim(0, 1)
        ax.set_title(metal, fontsize=12, fontweight="bold", color=METAL_COLORS[metal])
        ax.set_xlabel("pH", fontsize=10.5); ax.set_ylabel("Species fraction", fontsize=10.5)
        ax.legend(fontsize=11, loc="lower left", framealpha=0.6)
        ax.axvline(7, color="gray", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    out = OUTDIR / f"figC_speciation_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


plot_speciation(df_noeq, "noeq")
plot_speciation(df_eq,   "eq")


# ── C2: Total metal partitioning (aqueous species + solid phase) ───────────────
# Solid phase columns and stoichiometry correction factors
# Al_sulfate_mol = mol of Al₂(SO₄)₃  → 2 Al atoms per formula unit
SOLID_MAP = {
    "Al": ("Gibbsite",    2),
    "Fe": ("Fe(OH)3(s)", 1),
    "Cu": ("Cu(OH)2(s)", 1),
    "Ni": ("Ni(OH)2(s)", 1),
    "Co": ("Co(OH)2(s)", 1),
    "Mn": ("Mn(OH)2(s)", 1),
}
# Display labels for solid phases (replaces mineral names with chemical formulae)
SOLID_LABEL = {
    "Gibbsite":    "Al(OH)$_3$(s)",
    "Fe(OH)3(s)":  "Fe(OH)$_3$(s)",
    "Cu(OH)2(s)":  "Cu(OH)$_2$(s)",
    "Ni(OH)2(s)":  "Ni(OH)$_2$(s)",
    "Co(OH)2(s)":  "Co(OH)$_2$(s)",
    "Mn(OH)2(s)":  "Mn(OH)$_2$(s)",
}
SOLID_COLOR = "#444444"   # dark gray for solid phase band


def plot_speciation_with_solid(df, tag):
    """
    Total metal partitioning: fraction of each metal in identified aqueous
    species, unresolved dissolved forms ('Other dissolved'), and solid phase.

    Normalisation uses the PHREEQC -totals column (all dissolved forms) +
    solid phase column, so fractions are true fractions of the total metal
    budget and sum to 1 at every pH bin.

    Stoichiometric factors in SPECIES_STOICH convert species molality to
    moles of metal (e.g. Ni4(OH)4^4+ contributes 4× to Ni budget).
    """
    metals = list(SPECIES_MAP.keys())
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=False)
    axes = axes.flatten()

    pH_min      = np.floor(df["pH"].min() * 10) / 10
    pH_max      = np.ceil(df["pH"].max()  * 10) / 10
    pH_SWITCH1  = 7.0    # fine bins below this
    pH_SWITCH2  = 9.75   # medium bins below this, wide bins above
    BIN_FINE    = 0.05
    BIN_MEDIUM  = 0.14
    BIN_WIDE    = 0.50
    pH_bins = np.concatenate([
        np.arange(pH_min,                  pH_SWITCH1 + BIN_FINE,   BIN_FINE),
        np.arange(pH_SWITCH1 + BIN_MEDIUM, pH_SWITCH2 + BIN_MEDIUM, BIN_MEDIUM),
        np.arange(pH_SWITCH2 + BIN_WIDE,   pH_max     + BIN_WIDE,   BIN_WIDE),
    ])
    pH_bins = np.unique(np.round(pH_bins, 6))   # remove duplicates at junction
    pH_mid  = 0.5 * (pH_bins[:-1] + pH_bins[1:])

    for ax, metal in zip(axes, metals):
        sp        = SPECIES_MAP[metal]
        solid_col, _ = SOLID_MAP[metal]
        tot_col   = TOTALS_COL[metal]

        valid = [(l, c) for l, c in sp.items() if c in df.columns]
        if not valid:
            continue
        labels, aq_cols = zip(*valid)
        aq_cols = list(aq_cols)

        solid_present = solid_col in df.columns
        tot_present   = tot_col   in df.columns

        aq_fracs   = []   # (n_bins, n_aq_species) — fraction of total metal
        solid_frac = []   # (n_bins,)
        other_frac = []   # (n_bins,) — unresolved dissolved

        for lo, hi in zip(pH_bins[:-1], pH_bins[1:]):
            mask = (df["pH"] >= lo) & (df["pH"] < hi)
            sub  = df[mask]
            n    = len(sub)
            if n == 0:
                aq_fracs.append([np.nan] * len(aq_cols))
                solid_frac.append(np.nan); other_frac.append(np.nan)
                continue

            sol = sub[solid_col].fillna(0) if solid_present \
                  else pd.Series(np.zeros(n), index=sub.index)

            # total dissolved from -totals (all forms, mol/kgw)
            tot_diss = sub[tot_col].fillna(0) if tot_present \
                       else pd.Series(np.zeros(n), index=sub.index)

            total = (tot_diss + sol).replace(0, np.nan)   # full metal budget

            # metal contribution from each mapped aqueous species (stoich-corrected)
            aq_metal = pd.DataFrame(index=sub.index)
            for col in aq_cols:
                stoich = SPECIES_STOICH.get(col, 1)
                aq_metal[col] = sub[col].fillna(0) * stoich

            aq_f   = aq_metal.div(total, axis=0).mean().values
            sol_f  = (sol / total).mean()

            # unresolved dissolved = total_dissolved − sum of mapped metal contributions
            mapped_metal_sum = aq_metal.sum(axis=1)
            other = (tot_diss - mapped_metal_sum).clip(lower=0)
            oth_f = (other / total).mean()

            aq_fracs.append(aq_f)
            solid_frac.append(sol_f)
            other_frac.append(oth_f)

        aq_fracs   = np.array(aq_fracs)
        solid_frac = np.array(solid_frac)
        other_frac = np.array(other_frac)

        has_data = np.nanmax(aq_fracs, axis=0) > 0.005

        # ── stacked identified aqueous bands ──────────────────────────────────
        bottom = np.zeros(len(pH_mid))
        color_idx = 0
        for i, label in enumerate(labels):
            if not has_data[i]:
                continue
            vals = np.nan_to_num(aq_fracs[:, i])
            ax.fill_between(pH_mid, bottom, bottom + vals,
                            label=label,
                            color=SPEC_PALETTE[color_idx % len(SPEC_PALETTE)],
                            alpha=0.82, step="mid", linewidth=0.8, edgecolor="black")
            bottom += vals
            color_idx += 1

        # ── "Other dissolved" band (light gray, no hatch) ─────────────────────
        other_vals = np.nan_to_num(other_frac)
        if other_vals.max() > 0.005:
            ax.fill_between(pH_mid, bottom, bottom + other_vals,
                            label="Other dissolved",
                            color="#aaaaaa", alpha=0.55, step="mid",
                            linewidth=0.8, edgecolor="black")
            bottom += other_vals

        # ── solid phase band (dark gray fill, near-white hatch, black border) ──
        if solid_present:
            sol_vals = np.nan_to_num(solid_frac)
            ax.fill_between(pH_mid, bottom, bottom + sol_vals,
                            label=f"{SOLID_LABEL.get(solid_col, solid_col)} (solid)",
                            color="#555555", alpha=0.85, step="mid",
                            hatch="///", linewidth=0, edgecolor="#e8e8e8")
            ax.fill_between(pH_mid, bottom, bottom + sol_vals,
                            facecolor="none", step="mid",
                            linewidth=0.8, edgecolor="black")

        ax.set_xlim(pH_min, 10); ax.set_ylim(0, 1)
        ax.set_title(metal, fontsize=12, fontweight="bold", color=METAL_COLORS[metal])
        ax.set_xlabel("pH", fontsize=10.5)
        ax.set_ylabel("Fraction of total metal", fontsize=10.5)
        leg_loc = "upper left" if metal in ("Ni", "Co", "Mn") else "upper right"
        ax.legend(fontsize=10, loc=leg_loc, framealpha=0.6)
        ax.axvline(7, color="gray", linestyle="--", linewidth=0.7)

    plt.tight_layout()
    out = OUTDIR / f"figC_speciation_with_solid_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


plot_speciation_with_solid(df_noeq, "noeq")
plot_speciation_with_solid(df_eq,   "eq")



# ── N1: Co-precipitation risk matrix ──────────────────────────────────────────
def coprecip_matrix(df, threshold, tag, title):
    phases = list(SI_COLS.keys()); n = len(phases)
    matrix = np.zeros((n, n))
    for i, pi in enumerate(phases):
        sat_i = df[SI_COLS[pi]] >= threshold
        for j, pj in enumerate(phases):
            matrix[i, j] = (sat_i & (df[SI_COLS[pj]] >= threshold)).sum() / max(sat_i.sum(), 1)
    short = ["Fe(OH)$_3$","Gibbsite","Cu(OH)$_2$","Ni(OH)$_2$","Co(OH)$_2$","Mn(OH)$_2$"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Co-precipitation probability")
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i,j] > 0.6 or matrix[i,j] < 0.2 else "black")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    out = OUTDIR / f"figN1_coprecip_matrix_{tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


coprecip_matrix(df_noeq, 0.0,  "noeq", "Co-precipitation risk (suppressed equilibrium)")
coprecip_matrix(df_eq, -0.05,  "eq",   "Co-precipitation risk (phase equilibrium enforced)")


# ── N2: Recovery curves ────────────────────────────────────────────────────────
METAL_INPUT_COLS = {m: f"{m}_sulfate_mol" for m in ["Al","Fe","Cu","Ni","Co","Mn"]}
DISSOLVED_COLS   = {m: m for m in ["Al","Fe","Cu","Ni","Co","Mn"]}
GROUPS = {"group1": ["Al","Fe","Cu"], "group2": ["Ni","Co","Mn"]}


def _recovery_series(df, metal):
    pH_bins = np.arange(2, 12.5, 0.25)
    pH_mid  = 0.5 * (pH_bins[:-1] + pH_bins[1:])
    rec = (df[DISSOLVED_COLS[metal]] / df[METAL_INPUT_COLS[metal]].replace(0, np.nan)).clip(0, 1)
    tmp = df[["pH"]].copy(); tmp["rec"] = rec
    medians, q25s, q75s = [], [], []
    for lo, hi in zip(pH_bins[:-1], pH_bins[1:]):
        sub = tmp[(tmp["pH"] >= lo) & (tmp["pH"] < hi)]["rec"].dropna()
        if len(sub) < 3:
            medians.append(np.nan); q25s.append(np.nan); q75s.append(np.nan)
        else:
            medians.append(np.median(sub))
            q25s.append(np.percentile(sub, 25))
            q75s.append(np.percentile(sub, 75))
    medians = pd.Series(medians, index=pH_mid)
    q25s    = pd.Series(q25s,    index=pH_mid)
    q75s    = pd.Series(q75s,    index=pH_mid)
    has_data   = ~medians.isna()
    med_interp = medians.interpolate(method="linear", limit=3, limit_area="inside")
    q25_interp = q25s.interpolate(method="linear", limit=3, limit_area="inside")
    q75_interp = q75s.interpolate(method="linear", limit=3, limit_area="inside")
    gap_filled = ~has_data & ~med_interp.isna()
    return pH_mid, tmp, medians, q25_interp, q75_interp, has_data, gap_filled, med_interp


def plot_recovery(df, tag, title):
    """Two-panel version: early-precipitating (Al, Fe, Cu) | late-precipitating (Ni, Co, Mn)."""
    # Nature-style rcParams
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.direction": "out",  "ytick.direction": "out",
        "xtick.top": True,        "ytick.right": True,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    for ax, (gkey, metals), glabel, gtitle in zip(
            axes, GROUPS.items(), ["(a)","(b)"], ["Al · Fe · Cu", "Ni · Co · Mn"]):
        for metal in metals:
            if METAL_INPUT_COLS[metal] not in df.columns: continue
            pH_mid, tmp, medians, q25_interp, q75_interp, has_data, gap_filled, med_interp = \
                _recovery_series(df, metal)
            c = METAL_COLORS[metal]
            ax.scatter(tmp["pH"], tmp["rec"], color=c, s=2, alpha=0.04, linewidths=0, rasterized=True)
            ax.fill_between(pH_mid, q25_interp.values, q75_interp.values, color=c, alpha=0.15)
            ax.plot(pH_mid[has_data.values], med_interp.values[has_data.values],
                    color=c, linewidth=1.8, label=metal)
            ax.plot(pH_mid[gap_filled.values], med_interp.values[gap_filled.values],
                    color=c, linewidth=1.2, linestyle="--", alpha=0.55)
        ax.axhline(0.05, color="#c0392b", linestyle=":", linewidth=1.0, label="95% removal")
        ax.axvline(7, color="#7f7f7f", linestyle="--", linewidth=0.7)
        ax.set_xlim(2, 10); ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("pH", fontsize=9)
        ax.text(0.04, 0.96, glabel, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")
        leg_loc = "lower left" if gkey == "group2" else "upper right"
        ax.legend(fontsize=8, framealpha=0.8, frameon=True,
                  edgecolor="0.7", loc=leg_loc,
                  handlelength=1.6, handletextpad=0.5)
        ax.minorticks_on()
    axes[0].set_ylabel("Dissolved fraction remaining", fontsize=9)
    plt.tight_layout(pad=0.5, w_pad=0.8)
    out = OUTDIR / f"figN2_recovery_curves_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")
    plt.rcParams.update(plt.rcParamsDefault)


def plot_recovery_single(df, tag, title):
    """Single-panel version with all 6 metals, Nature journal style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.direction": "out",  "ytick.direction": "out",
        "xtick.top": True,        "ytick.right": True,
    })
    # Ordered to match precipitation sequence
    METALS_ALL = ["Fe", "Al", "Cu", "Ni", "Co", "Mn"]
    # Distinct line styles to aid black-and-white reproduction
    LINESTYLES = {"Fe": "-", "Al": "-", "Cu": "-", "Ni": "--", "Co": "--", "Mn": "--"}

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for metal in METALS_ALL:
        if METAL_INPUT_COLS[metal] not in df.columns:
            continue
        pH_mid, tmp, medians, q25_interp, q75_interp, has_data, gap_filled, med_interp = \
            _recovery_series(df, metal)
        c  = METAL_COLORS[metal]
        ls = LINESTYLES[metal]
        ax.scatter(tmp["pH"], tmp["rec"], color=c, s=1.5, alpha=0.13,
                   linewidths=0, rasterized=True)
        ax.fill_between(pH_mid, q25_interp.values, q75_interp.values,
                        color=c, alpha=0.12)
        ax.plot(pH_mid[has_data.values], med_interp.values[has_data.values],
                color=c, linewidth=1.8, linestyle=ls, label=metal)
        ax.plot(pH_mid[gap_filled.values], med_interp.values[gap_filled.values],
                color=c, linewidth=1.1, linestyle=":", alpha=0.75)

    ax.axhline(0.05, color="#c0392b", linestyle=":", linewidth=1.0,
               label="95% removal")
    ax.axvline(7, color="#7f7f7f", linestyle="--", linewidth=0.7, alpha=0.8)

    ax.set_xlim(3, 10)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("pH", fontsize=9)
    ax.set_ylabel("Dissolved fraction remaining", fontsize=9)
    ax.minorticks_on()

    # custom legend: metal lines + band proxies
    import matplotlib.patches as mpatches
    iqr_patch  = mpatches.Patch(color="gray", alpha=0.35, label="IQR (25–75%)")
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles=handles + [iqr_patch],
              labels=labels + ["IQR (25–75%)"],
              fontsize=7.5, framealpha=0.9, frameon=True, edgecolor="0.7",
              loc="upper right", handlelength=1.8, handletextpad=0.4, ncol=1)

    #leg = ax.legend(fontsize=8, framealpha=0.8, frameon=True,
    #                edgecolor="0.7", loc="lower right",
    #                handlelength=2.0, handletextpad=0.5,
    #                ncol=1)

    # Annotate precipitation tiers
    ##ax.annotate("Fe, Al, Cu", xy=(5.8, 0.5), fontsize=7.5, color="#555555",
    ##            ha="center", style="italic")
    ##ax.annotate("Ni, Co, Mn", xy=(9.0, 0.6), fontsize=7.5, color="#555555",
    ##            ha="center", style="italic")

    plt.tight_layout(pad=0.5)
    out = OUTDIR / f"figN2_recovery_curves_single_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    plt.rcParams.update(plt.rcParamsDefault)


def plot_recovery_single_pct(df, tag, title):
    """
    Single-panel % recovery version of plot_recovery_single.
    y-axis = (1 - dissolved fraction remaining) × 100  →  % precipitated / recovered.
    All other styling mirrors plot_recovery_single exactly.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        #"xtick.top": True,        "ytick.right": True,
    })
    METALS_ALL = ["Fe", "Al", "Cu", "Ni", "Co", "Mn"]
    LINESTYLES = {"Fe": "-", "Al": "-", "Cu": "-", "Ni": "--", "Co": "--", "Mn": "--"}

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for metal in METALS_ALL:
        if METAL_INPUT_COLS[metal] not in df.columns:
            continue
        pH_mid, tmp, medians, q25_interp, q75_interp, has_data, gap_filled, med_interp = \
            _recovery_series(df, metal)
        c  = METAL_COLORS[metal]
        ls = LINESTYLES[metal]

        # convert dissolved fraction → % recovery
        pct_scatter  = (1 - tmp["rec"]) * 100
        pct_med      = (1 - med_interp.values) * 100
        pct_q25      = (1 - q75_interp.values) * 100   # note: q75 dissolved → q25 recovered
        pct_q75      = (1 - q25_interp.values) * 100   # note: q25 dissolved → q75 recovered

        ax.scatter(tmp["pH"], pct_scatter, color=c, s=1.5, alpha=0.13,
                   linewidths=0, rasterized=True)
        ax.fill_between(pH_mid, pct_q25, pct_q75, color=c, alpha=0.12)
        ax.plot(pH_mid[has_data.values], pct_med[has_data.values],
                color=c, linewidth=1.8, linestyle=ls, label=metal)
        ax.plot(pH_mid[gap_filled.values], pct_med[gap_filled.values],
                color=c, linewidth=1.1, linestyle=":", alpha=0.75)

    # 95% recovery reference line
    ax.axhline(95, color="#c0392b", linestyle=":", linewidth=1.0, label="95% recovery")
    ax.axvline(7,  color="#7f7f7f", linestyle="--", linewidth=0.7, alpha=0.8)

    ax.set_xlim(3, 10)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("pH", fontsize=9)
    ax.set_ylabel("Recovery (%)", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.minorticks_on()

    import matplotlib.patches as mpatches
    iqr_patch = mpatches.Patch(color="gray", alpha=0.35, label="IQR (25–75%)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [iqr_patch],
              labels=labels + ["IQR (25–75%)"],
              fontsize=7.5, framealpha=0.9, frameon=True, edgecolor="0.7",
              loc="lower right", handlelength=1.8, handletextpad=0.4, ncol=1)

    plt.tight_layout(pad=0.5)
    out = OUTDIR / f"figN2_recovery_pct_single_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    plt.rcParams.update(plt.rcParamsDefault)


def _recovery_percentiles(df, metal, pH_bins):
    """Return pH_mid and dict of percentile arrays (5, 25, 50, 75, 95)."""
    pH_mid = 0.5 * (pH_bins[:-1] + pH_bins[1:])
    rec = (df[DISSOLVED_COLS[metal]] / df[METAL_INPUT_COLS[metal]].replace(0, np.nan)).clip(0, 1)
    tmp = df[["pH"]].copy(); tmp["rec"] = rec
    pcts = {p: [] for p in [5, 25, 50, 75, 95]}
    for lo, hi in zip(pH_bins[:-1], pH_bins[1:]):
        sub = tmp[(tmp["pH"] >= lo) & (tmp["pH"] < hi)]["rec"].dropna()
        for p in pcts:
            pcts[p].append(np.percentile(sub, p) if len(sub) >= 3 else np.nan)
    for p in pcts:
        s = pd.Series(pcts[p], index=pH_mid)
        pcts[p] = s.interpolate(method="linear", limit=3, limit_area="inside").values
    return pH_mid, pcts


def plot_recovery_percentile_stack(df, tag, title):
    """
    Version 3 – Percentile envelope stack (5–95 / 25–75 / median).
    Nested bands give a density-gradient effect; no raw scatter.
    """
    NATURE_RC = {
        "font.family": "sans-serif", "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.direction": "out",  "ytick.direction": "out",
        "xtick.top": True,        "ytick.right": True,
    }
    plt.rcParams.update(NATURE_RC)

    METALS_ALL = ["Fe", "Al", "Cu", "Ni", "Co", "Mn"]
    LINESTYLES = {"Fe": "-", "Al": "-", "Cu": "-", "Ni": "--", "Co": "--", "Mn": "--"}
    pH_bins = np.arange(2, 12.5, 0.25)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for metal in METALS_ALL:
        if METAL_INPUT_COLS[metal] not in df.columns:
            continue
        c  = METAL_COLORS[metal]
        ls = LINESTYLES[metal]
        pH_mid, pcts = _recovery_percentiles(df, metal, pH_bins)

        # outer band: 5–95th percentile (lightest)
        ax.fill_between(pH_mid, pcts[5], pcts[95], color=c, alpha=0.10, linewidth=0)
        # inner band: 25–75th percentile (darker)
        ax.fill_between(pH_mid, pcts[25], pcts[75], color=c, alpha=0.22, linewidth=0)
        # median line
        ax.plot(pH_mid, pcts[50], color=c, linewidth=1.8, linestyle=ls, label=metal)

    ax.axhline(0.05, color="#c0392b", linestyle=":", linewidth=1.0, label="95% removal")
    ax.axvline(7,    color="#7f7f7f", linestyle="--", linewidth=0.7, alpha=0.8)

    ax.set_xlim(3, 10); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("pH", fontsize=9)
    ax.set_ylabel("Dissolved fraction remaining", fontsize=9)
    ax.minorticks_on()

    # custom legend: metal lines + band proxies
    import matplotlib.patches as mpatches
    handles, labels = ax.get_legend_handles_labels()
    iqr_patch  = mpatches.Patch(color="gray", alpha=0.35, label="IQR (25–75%)")
    tail_patch = mpatches.Patch(color="gray", alpha=0.15, label="5–95%")
    ax.legend(handles=handles + [iqr_patch, tail_patch],
              labels=labels + ["IQR (25–75%)", "5–95%"],
              fontsize=7.5, framealpha=0.9, frameon=True, edgecolor="0.7",
              loc="upper right", handlelength=1.8, handletextpad=0.4, ncol=1)

    # tier annotations
    ax.text(5.5, 0.48, "Fe, Al, Cu", fontsize=7.5, color="#444444",
            ha="center", style="italic")
    ax.text(9.2, 0.62, "Ni, Co, Mn", fontsize=7.5, color="#444444",
            ha="center", style="italic")

    plt.tight_layout(pad=0.5)
    out = OUTDIR / f"figN2_recovery_curves_pct_stack_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    plt.rcParams.update(plt.rcParamsDefault)


def plot_recovery_violin(df, tag, title):
    """
    Version 4 – Half-violin (ridge) at discrete pH positions per metal.
    Each violin shows the full distribution shape at that pH slice.
    Metals are offset vertically by a small amount so violins don't collide.
    A thin median line connects the median dots across pH.
    """
    NATURE_RC = {
        "font.family": "sans-serif", "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.direction": "out",  "ytick.direction": "out",
        "xtick.top": True,        "ytick.right": True,
    }
    plt.rcParams.update(NATURE_RC)

    METALS_ALL = ["Fe", "Al", "Cu", "Ni", "Co", "Mn"]
    LINESTYLES = {"Fe": "-", "Al": "-", "Cu": "-", "Ni": "--", "Co": "--", "Mn": "--"}

    # pH positions at which to draw violins (coarser than bins)
    pH_slots = np.array([3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0])
    half_width_pH = 0.28   # visual half-width of each violin in pH units
    violin_scale  = 0.55   # max height of violin in recovery-fraction units

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for metal in METALS_ALL:
        if METAL_INPUT_COLS[metal] not in df.columns:
            continue
        c  = METAL_COLORS[metal]
        ls = LINESTYLES[metal]
        rec = (df[DISSOLVED_COLS[metal]] /
               df[METAL_INPUT_COLS[metal]].replace(0, np.nan)).clip(0, 1)
        tmp = df[["pH"]].copy(); tmp["rec"] = rec

        medians_pts = []

        for pH_c in pH_slots:
            lo, hi = pH_c - 0.3, pH_c + 0.3
            sub = tmp[(tmp["pH"] >= lo) & (tmp["pH"] < hi)]["rec"].dropna().values
            if len(sub) < 5:
                continue

            # KDE over [0,1] — skip degenerate bins (zero variance)
            from scipy.stats import gaussian_kde
            if np.std(sub) < 1e-6:
                medians_pts.append((pH_c, np.median(sub)))
                continue
            kde = gaussian_kde(sub, bw_method=0.2)
            y_vals = np.linspace(0, 1, 200)
            density = kde(y_vals)
            density = density / density.max() * violin_scale * half_width_pH

            # draw right half-violin only (cleaner for overlapping metals)
            ax.fill_betweenx(y_vals,
                             pH_c, pH_c + density,
                             color=c, alpha=0.30, linewidth=0)
            ax.plot(pH_c + density, y_vals, color=c, linewidth=0.6, alpha=0.7)

            medians_pts.append((pH_c, np.median(sub)))

        if medians_pts:
            mx, my = zip(*medians_pts)
            # median dot
            ax.scatter(mx, my, color=c, s=16, zorder=5, linewidths=0.5,
                       edgecolors="white")
            # connecting median line
            ax.plot(mx, my, color=c, linewidth=1.4, linestyle=ls,
                    label=metal, alpha=0.9)

    ax.axhline(0.05, color="#c0392b", linestyle=":", linewidth=1.0, label="95% removal")
    ax.axvline(7,    color="#7f7f7f", linestyle="--", linewidth=0.7, alpha=0.8)

    ax.set_xlim(2, 10); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("pH", fontsize=9)
    ax.set_ylabel("Dissolved fraction remaining", fontsize=9)
    ax.minorticks_on()

    # proxy for violin shape in legend
    import matplotlib.patches as mpatches
    violin_proxy = mpatches.Patch(color="gray", alpha=0.35, label="Distribution (KDE)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [violin_proxy],
              labels=labels + ["KDE shape"],
              fontsize=7.5, framealpha=0.9, frameon=True, edgecolor="0.7",
              loc="upper right", handlelength=1.8, handletextpad=0.4, ncol=2)

    ax.text(5.5, 0.48, "Fe, Al, Cu", fontsize=7.5, color="#444444",
            ha="center", style="italic")
    ax.text(9.5, 0.65, "Ni, Co, Mn", fontsize=7.5, color="#444444",
            ha="center", style="italic")

    plt.tight_layout(pad=0.5)
    out = OUTDIR / f"figN2_recovery_curves_violin_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    plt.rcParams.update(plt.rcParamsDefault)


def plot_recovery_method_comparison(df, tag):
    """
    6-panel figure (one per metal) comparing two % recovery methods:
      - Indirect : (1 − dissolved/input) × 100
      - Direct   : precipitated_phase / input × 100  (with correct stoichiometry)

    Al stoichiometry note: Al_sulfate_mol = moles of Al₂(SO₄)₃ → actual Al input = 2×Al_sulfate_mol.
    Gibbsite has 1 Al per formula unit, so direct Al recovery = Gibbsite / (2×Al_sulfate_mol).

    Residual panel per metal shows (indirect − direct) to highlight where dissolved
    complexes make the two methods diverge.
    """
    NATURE_RC = {
        "font.family": "sans-serif", "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.direction": "out",  "ytick.direction": "out",
        "xtick.top": True,        "ytick.right": True,
    }
    plt.rcParams.update(NATURE_RC)

    # Metal → (dissolved_col, input_col, phase_col, input_stoich_factor)
    # stoich_factor: multiply input_col by this to get actual moles of the metal
    METALS_CFG = {
        "Fe": ("Fe", "Fe_sulfate_mol", "Fe(OH)3(s)",  1),
        "Al": ("Al", "Al_sulfate_mol", "Gibbsite",     2),   # Al2(SO4)3 → 2 Al
        "Cu": ("Cu", "Cu_sulfate_mol", "Cu(OH)2(s)",  1),
        "Ni": ("Ni", "Ni_sulfate_mol", "Ni(OH)2(s)",  1),
        "Co": ("Co", "Co_sulfate_mol", "Co(OH)2(s)",  1),
        "Mn": ("Mn", "Mn_sulfate_mol", "Mn(OH)2(s)",  1),
    }

    pH_bins = np.arange(2, 12.5, 0.3)
    pH_mid  = 0.5 * (pH_bins[:-1] + pH_bins[1:])

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5), sharey=False)
    axes = axes.flatten()

    for ax, (metal, (diss_col, inp_col, phase_col, stoich)) in zip(axes, METALS_CFG.items()):
        c = METAL_COLORS[metal]

        # ── compute per-bin medians ──────────────────────────────────────────
        inp   = df[inp_col].replace(0, np.nan) * stoich          # actual metal moles input
        diss  = df[diss_col].fillna(0)
        phase = df[phase_col].fillna(0) if phase_col in df.columns else pd.Series(0, index=df.index)

        pct_indirect = ((1 - diss / inp) * 100).clip(0, 100)
        pct_direct   = (phase / inp * 100).clip(0, 100)
        residual     = pct_indirect - pct_direct

        tmp = pd.DataFrame({"pH": df["pH"], "ind": pct_indirect,
                            "dir": pct_direct, "res": residual})

        med_ind, med_dir, med_res = [], [], []
        q25_ind, q75_ind = [], []
        q25_dir, q75_dir = [], []

        for lo, hi in zip(pH_bins[:-1], pH_bins[1:]):
            sub = tmp[(tmp["pH"] >= lo) & (tmp["pH"] < hi)].dropna()
            if len(sub) < 3:
                for lst in [med_ind, med_dir, med_res, q25_ind, q75_ind, q25_dir, q75_dir]:
                    lst.append(np.nan)
            else:
                med_ind.append(np.median(sub["ind"]))
                med_dir.append(np.median(sub["dir"]))
                med_res.append(np.median(sub["res"]))
                q25_ind.append(np.percentile(sub["ind"], 25))
                q75_ind.append(np.percentile(sub["ind"], 75))
                q25_dir.append(np.percentile(sub["dir"], 25))
                q75_dir.append(np.percentile(sub["dir"], 75))

        med_ind = np.array(med_ind); med_dir = np.array(med_dir)
        med_res = np.array(med_res)
        q25_ind = np.array(q25_ind); q75_ind = np.array(q75_ind)
        q25_dir = np.array(q25_dir); q75_dir = np.array(q75_dir)

        valid = np.isfinite(med_ind)

        # ── main recovery curves ─────────────────────────────────────────────
        ax.fill_between(pH_mid[valid], q25_ind[valid], q75_ind[valid],
                        color=c, alpha=0.15, linewidth=0)
        ax.fill_between(pH_mid[valid], q25_dir[valid], q75_dir[valid],
                        color="steelblue", alpha=0.13, linewidth=0)
        ax.plot(pH_mid[valid], med_ind[valid], color=c,
                linewidth=1.8, linestyle="-", label="Indirect")
        ax.plot(pH_mid[valid], med_dir[valid], color="steelblue",
                linewidth=1.5, linestyle="--", label="Direct (phase)")

        # ── residual as thin line on secondary feel — shade gap ──────────────
        ax.fill_between(pH_mid[valid], med_dir[valid], med_ind[valid],
                        where=med_ind[valid] >= med_dir[valid],
                        color="salmon", alpha=0.25, linewidth=0,
                        label="Gap (complexes)")

        ax.axhline(95, color="#c0392b", linestyle=":", linewidth=0.9)
        ax.axvline(7,  color="#7f7f7f", linestyle="--", linewidth=0.7, alpha=0.7)

        ax.set_xlim(2, 10); ax.set_ylim(-2, 102)
        ax.set_title(metal, fontsize=10, fontweight="bold", color=c, pad=3)
        ax.set_xlabel("pH", fontsize=8)
        ax.set_ylabel("Recovery (%)", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.minorticks_on()
        ax.legend(fontsize=6.5, framealpha=0.85, frameon=True,
                  edgecolor="0.7", loc="lower right",
                  handlelength=1.6, handletextpad=0.4)

        # stoich note for Al
        if stoich == 2:
            ax.text(0.03, 0.55, "Al₂(SO₄)₃\n÷2 applied",
                    transform=ax.transAxes, fontsize=6, color="gray", va="top")

    fig.suptitle(
        f"Recovery comparison: indirect (dissolved) vs direct (phase)\n"
        f"Median ± IQR across 5,001 LHS samples — {tag}",
        fontsize=9, y=1.01)
    plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=1.0)
    out = OUTDIR / f"figN2_recovery_method_comparison_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    plt.rcParams.update(plt.rcParamsDefault)


def plot_recovery_direct_pct(df, tag):
    """
    Single-panel % recovery from direct phase precipitation data.
    Mirrors plot_recovery_single_pct exactly in style (scatter, IQR band,
    solid/dotted median line, Nature rcParams, 300 dpi) but computes:

        % recovery = phase_precipitated / actual_metal_input × 100

    Stoichiometry corrections applied:
      - Al: Al_sulfate_mol = moles of Al₂(SO₄)₃  → actual Al = 2 × Al_sulfate_mol
      - Fe, Cu, Ni, Co, Mn: 1:1 (MSO₄ salts)
    """
    NATURE_RC = {
        "font.family": "sans-serif", "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.direction": "out",  "ytick.direction": "out",
        #"xtick.top": True,        "ytick.right": True,
    }
    plt.rcParams.update(NATURE_RC)

    # (input_col, phase_col, stoich_factor, legend_label)
    METALS_CFG = {
        "Fe": ("Fe_sulfate_mol", "Fe(OH)3(s)",  1, "Fe(OH)$_3$(s)"),
        "Al": ("Al_sulfate_mol", "Gibbsite",     2, "Al(OH)$_3$(s)"),
        "Cu": ("Cu_sulfate_mol", "Cu(OH)2(s)",  1, "Cu(OH)$_2$(s)"),
        "Ni": ("Ni_sulfate_mol", "Ni(OH)2(s)",  1, "Ni(OH)$_2$(s)"),
        "Co": ("Co_sulfate_mol", "Co(OH)2(s)",  1, "Co(OH)$_2$(s)"),
        "Mn": ("Mn_sulfate_mol", "Mn(OH)2(s)",  1, "Mn(OH)$_2$(s)"),
    }
    METALS_ALL = ["Fe", "Al", "Cu", "Ni", "Co", "Mn"]
    LINESTYLES = {"Fe": "-", "Al": "-", "Cu": "-", "Ni": "--", "Co": "--", "Mn": "--"}

    pH_bins = np.arange(2, 12.5, 0.25)
    pH_mid  = 0.5 * (pH_bins[:-1] + pH_bins[1:])

    fig, ax = plt.subplots(figsize=(7, 5))

    for metal in METALS_ALL:
        inp_col, phase_col, stoich, phase_label = METALS_CFG[metal]
        if inp_col not in df.columns or phase_col not in df.columns:
            continue

        c  = METAL_COLORS[metal]
        ls = LINESTYLES[metal]

        inp   = df[inp_col].replace(0, np.nan) * stoich
        phase = df[phase_col].fillna(0)
        pct   = (phase / inp * 100).clip(0, 100)

        tmp = pd.DataFrame({"pH": df["pH"], "pct": pct})

        medians, q25s, q75s = [], [], []
        for lo, hi in zip(pH_bins[:-1], pH_bins[1:]):
            sub = tmp[(tmp["pH"] >= lo) & (tmp["pH"] < hi)]["pct"].dropna()
            if len(sub) < 3:
                medians.append(np.nan); q25s.append(np.nan); q75s.append(np.nan)
            else:
                medians.append(np.median(sub))
                q25s.append(np.percentile(sub, 25))
                q75s.append(np.percentile(sub, 75))

        medians = pd.Series(medians, index=pH_mid)
        q25s    = pd.Series(q25s,    index=pH_mid)
        q75s    = pd.Series(q75s,    index=pH_mid)
        has_data   = ~medians.isna()
        med_interp = medians.interpolate(method="linear", limit=3, limit_area="inside")
        q25_interp = q25s.interpolate(method="linear", limit=3, limit_area="inside")
        q75_interp = q75s.interpolate(method="linear", limit=3, limit_area="inside")
        gap_filled = ~has_data & ~med_interp.isna()

        # raw scatter
        ax.scatter(tmp["pH"], tmp["pct"], color=c, s=2.5, alpha=0.15,
                   linewidths=0, rasterized=True)
        # IQR band
        ax.fill_between(pH_mid, q25_interp.values, q75_interp.values,
                        color=c, alpha=0.15)
        # median solid
        ax.plot(pH_mid[has_data.values], med_interp.values[has_data.values],
                color=c, linewidth=1.8, linestyle=ls, label=phase_label)
        # interpolated gap dotted
        ax.plot(pH_mid[gap_filled.values], med_interp.values[gap_filled.values],
                color=c, linewidth=2.0, linestyle=":", alpha=0.75)

    ax.axhline(95, color="#c0392b", linestyle=":", linewidth=1.0, label="95% recovery")
    ax.axvline(7,  color="#7f7f7f", linestyle="--", linewidth=0.7, alpha=0.8)

    ax.set_xlim(3, 10)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("pH", fontsize=18)
    ax.set_ylabel("Recovery (%)", fontsize=18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.minorticks_on()
    ax.tick_params(axis='both', labelsize=16)
  

    import matplotlib.patches as mpatches
    iqr_patch = mpatches.Patch(color="gray", alpha=0.35, label="IQR (25–75%)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [iqr_patch],
              labels=labels + ["IQR (25–75%)"],
              fontsize=12, framealpha=0.9, frameon=True, edgecolor="0.7",
              loc="lower right", handlelength=1.8, handletextpad=0.4, ncol=1)

    plt.tight_layout()
    out = OUTDIR / f"figN2_recovery_direct_pct_{tag}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    plt.rcParams.update(plt.rcParamsDefault)


plot_recovery(df_noeq, "noeq", "Model-predicted dissolved metal fraction vs. pH (suppressed equilibrium)")
plot_recovery(df_eq,   "eq",   "Model-predicted dissolved metal fraction vs. pH (phase equilibrium)")
plot_recovery_single(df_noeq, "noeq", "Dissolved metal fraction vs. pH (suppressed equilibrium)")
plot_recovery_single(df_eq,   "eq",   "Dissolved metal fraction vs. pH (phase equilibrium)")
plot_recovery_single_pct(df_noeq, "noeq", "Metal recovery vs. pH (suppressed equilibrium)")
plot_recovery_single_pct(df_eq,   "eq",   "Metal recovery vs. pH (phase equilibrium)")
plot_recovery_percentile_stack(df_noeq, "noeq", "Dissolved metal fraction vs. pH (suppressed equilibrium)")
plot_recovery_percentile_stack(df_eq,   "eq",   "Dissolved metal fraction vs. pH (phase equilibrium)")
plot_recovery_violin(df_noeq, "noeq", "Dissolved metal fraction vs. pH (suppressed equilibrium)")
plot_recovery_violin(df_eq,   "eq",   "Dissolved metal fraction vs. pH (phase equilibrium)")
plot_recovery_method_comparison(df_noeq, "noeq")
plot_recovery_method_comparison(df_eq,   "eq")
plot_recovery_direct_pct(df_noeq, "noeq")
plot_recovery_direct_pct(df_eq,   "eq")


# ── N3: Free ion activity ──────────────────────────────────────────────────────
# Dominant free-ion species per metal. Fe²⁺ is read from a_Fe2+ when available
# (UP5+); for older datasets lacking that column it is derived from a_Fe3+ and
# pe via the Nernst equation (pe° = 13.03 for Fe³⁺/Fe²⁺ at 25 °C).
# ── Free ion activity: species definitions ────────────────────────────────────
# Each entry: label → (a_col, derived_key, color, linestyle, linewidth, iqr_shade)
# derived_key "fe2" = use a_Fe2+ directly, fall back to Nernst from a_Fe3+
_FE_PE_STD = 13.03   # pe° for Fe³⁺/Fe²⁺ at 25 °C

# Element base colors — distinct, colorblind-friendly palette
_ELEM_COLORS = {
    "Al": "#E69F00",   # amber
    "Fe": "#56B4E9",   # sky blue
    "Cu": "#009E73",   # teal green
    "Ni": "#F0E442",   # yellow
    "Co": "#0072B2",   # deep blue
    "Mn": "#D55E00",   # vermillion
}

# (col, derived, element, linestyle, linewidth, marker, iqr_shade)
ACT_SPECIES = [
    # label             col          derived  elem   ls         lw    shade
    ("Al$^{3+}$",      "a_Al3+",    None,   "Al",  "-",       2.2,  True ),
    ("Fe$^{2+}$",      "a_Fe2+",    "fe2",  "Fe",  "-",       2.2,  True ),
    ("Fe$^{3+}$",      "a_Fe3+",    None,   "Fe",  "--",      1.6,  False),
    ("Cu$^{2+}$",      "a_Cu2+",    None,   "Cu",  "-",       2.2,  True ),
    ("Cu$^{+}$",       "a_Cu+",     None,   "Cu",  (0,(4,2)), 1.6,  False),
    ("Ni$^{2+}$",      "a_Ni2+",    None,   "Ni",  "-",       2.2,  True ),
    ("Co$^{2+}$",      "a_Co2+",    None,   "Co",  "-",       2.2,  True ),
    ("Co$^{3+}$",      "a_Co3+",    None,   "Co",  "--",      1.6,  False),
    ("Mn$^{2+}$",      "a_Mn2+",    None,   "Mn",  "-",       2.2,  True ),
]


def _bin_log_activity(series, pH_series, pH_bins, min_n=1):
    log_a = np.log10(series.replace(0, np.nan).clip(lower=1e-20))
    tmp = pd.DataFrame({"pH": pH_series, "log_a": log_a})
    medians, q25s, q75s, p10s, p90s = [], [], [], [], []
    for lo, hi in zip(pH_bins[:-1], pH_bins[1:]):
        sub = tmp[(tmp["pH"] >= lo) & (tmp["pH"] < hi)]["log_a"].dropna()
        if len(sub) < min_n:
            medians.append(np.nan); q25s.append(np.nan); q75s.append(np.nan)
            p10s.append(np.nan);    p90s.append(np.nan)
        else:
            medians.append(np.median(sub))
            q25s.append(np.percentile(sub, 25) if len(sub) >= 3 else np.nan)
            q75s.append(np.percentile(sub, 75) if len(sub) >= 3 else np.nan)
            p10s.append(np.percentile(sub, 10) if len(sub) >= 5 else np.nan)
            p90s.append(np.percentile(sub, 90) if len(sub) >= 5 else np.nan)
    return medians, q25s, q75s, p10s, p90s


def plot_free_ion(df, tag, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    pH_bins = np.arange(2, 14.1, 0.3)
    pH_mid  = 0.5 * (pH_bins[:-1] + pH_bins[1:])

    for label, col, derived, elem, ls, lw, shade in ACT_SPECIES:
        color = _ELEM_COLORS[elem]

        if derived == "fe2":
            if "a_Fe2+" in df.columns:
                series = df["a_Fe2+"].replace(0, np.nan)
            elif "a_Fe3+" in df.columns and "pe" in df.columns:
                series = df["a_Fe3+"].replace(0, np.nan) * 10**(_FE_PE_STD - df["pe"])
            else:
                continue
        else:
            if col not in df.columns:
                continue
            series = df[col].replace(0, np.nan)

        medians, q25s, q75s, p10s, p90s = _bin_log_activity(series, df["pH"], pH_bins)

        ax.plot(pH_mid, medians, color=color, linewidth=lw,
                linestyle=ls, label=label, solid_capstyle="round")
        if shade:
            # outer band: 10th–90th percentile
            ax.fill_between(pH_mid, p10s, p90s, color=color, alpha=0.08)
            # inner band: IQR (Q1–Q3)
            ax.fill_between(pH_mid, q25s, q75s, color=color, alpha=0.18)

    # LOD line
    ax.axhline(-8, color="black", linestyle="--", linewidth=1.2,
               label="ICP-OES LOD ($\\approx10^{-8}$ mol/L)")
    # pH 7 marker
    ax.axvline(7, color="gray", linestyle=":", linewidth=0.9)
    # sub-LOD shading
    ax.fill_between([3, 10], -18, -8, color="lightgray", alpha=0.25, zorder=0)

    ax.set_xlim(3, 10); ax.set_ylim(-18, 0)
    ax.set_xlabel("pH", fontsize=12)
    ax.set_ylabel("log$_{10}$(free ion activity / mol L$^{-1}$)", fontsize=12)
    ax.set_title("", fontsize=12, pad=8)
    ax.tick_params(labelsize=10)

    # Add shading-band legend entries
    import matplotlib.patches as mpatches
    patch_iqr = mpatches.Patch(color="gray", alpha=0.35, label="IQR (Q1–Q3)")
    patch_90  = mpatches.Patch(color="gray", alpha=0.15, label="10th–90th pct.")
    handles, labels_leg = ax.get_legend_handles_labels()
    handles += [patch_iqr, patch_90]
    labels_leg += ["IQR (Q1–Q3)", "10th–90th pct."]
    ax.legend(handles, labels_leg, fontsize=9.5, ncol=2,
              framealpha=0.85, edgecolor="0.7",
              loc="lower left", bbox_to_anchor=(0.01, 0.01))

    plt.tight_layout()
    out = OUTDIR / f"figN3_free_ion_activity_{tag}.png"
    plt.savefig(out, dpi=220, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


plot_free_ion(df_noeq, "noeq", "Free ion activity suppression (suppressed equilibrium)")
plot_free_ion(df_eq,   "eq",   "Free ion activity suppression (phase equilibrium)")

print("\nAll figures done.")
