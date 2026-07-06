"""
Data-driven Pourbaix-style speciation diagrams (pH vs pe).
For each simulation point, the dominant species (highest molality among all
aqueous species + solid if SI >= 0) is identified. Points are coloured by
dominant species and a filled-contour background shows the inferred stability
regions across the pH-pe space.

Usage (run from repo root):
    python src/plot_pourbaix_speciation.py                  # defaults to UP2
    UP_DIR=paper_SIT/Leach_recovery_UP3 python src/plot_pourbaix_speciation.py

Outputs → results/<version>/fig_pourbaix_speciation.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import griddata
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

df = pd.read_csv(CSV)
df.columns = df.columns.str.strip()
_aliases = {"m_Fe+2": "m_Fe2+", "m_Fe+3": "m_Fe3+",
            "a_Fe+2": "a_Fe2+", "a_Fe+3": "a_Fe3+"}
df.rename(columns={k: v for k, v in _aliases.items()
                   if k in df.columns and v not in df.columns}, inplace=True)
df.replace(1e-99, np.nan, inplace=True)
df.replace(-999.999, np.nan, inplace=True)

ph = df["pH"].astype(float).to_numpy()
pe = df["pe"].astype(float).to_numpy()

# ── per-metal species definitions ──────────────────────────────────────────────
# Each entry: (display_label, m_col_or_None, si_col_or_None, is_solid)
METAL_SPECIES = {
    "Al": [
        ("Al³⁺(aq)",       "m_Al+3",          None,          False),
        ("AlOH²⁺",         "m_AlOH2+",        None,          False),
        ("Al(OH)₂⁺",       "m_Al(OH)2+",      None,          False),
        ("AlO₂⁻",          "m_AlO2-",         None,          False),
        ("AlSO₄⁺",         "m_AlSO4+",        None,          False),
        ("Polymeric Al",    "m_Al2(OH)2+4",    None,          False),
        ("Gibbsite(s)",     None,              "si_Gibbsite", True),
    ],
    "Fe": [
        ("Fe²⁺(aq)",        "m_Fe2+",          None,          False),
        ("Fe³⁺(aq)",        "m_Fe3+",          None,          False),
        ("FeOH²⁺",          "m_FeOH+",         None,          False),
        ("Fe(OH)₂(aq)",     "m_Fe(OH)2",       None,          False),
        ("Fe(OH)₃⁻",        "m_Fe(OH)3-",      None,          False),
        ("Fe(OH)₄⁻",        "m_Fe(OH)4-",      None,          False),
        ("FeSO₄⁺",          "m_FeSO4+",        None,          False),
        ("Fe(OH)₃(s)",      None,              "si_Fe(OH)3(s)", True),
    ],
    "Cu": [
        ("Cu²⁺(aq)",        "m_Cu+2",          None,          False),
        ("CuOH⁺",           "m_CuOH+",         None,          False),
        ("CuSO₄(aq)",       "m_CuSO4",         None,          False),
        ("Cu(OH)₂(s)",      None,              "si_Cu(OH)2(s)", True),
    ],
    "Ni": [
        ("Ni²⁺(aq)",        "m_Ni+2",          None,          False),
        ("NiOH⁺",           "m_NiOH+",         None,          False),
        ("Ni(OH)₂(aq)",     "m_Ni(OH)2",       None,          False),
        ("NiSO₄(aq)",       "m_NiSO4",         None,          False),
        ("Ni(OH)₂(s)",      None,              "si_Ni(OH)2(s)", True),
    ],
    "Co": [
        ("Co²⁺(aq)",        "m_Co+2",          None,          False),
        ("CoOH⁺",           "m_CoOH+",         None,          False),
        ("CoSO₄(aq)",       "m_CoSO4",         None,          False),
        ("Co(OH)₂(s)",      None,              "si_Co(OH)2(s)", True),
    ],
    "Mn": [
        ("Mn²⁺(aq)",        "m_Mn+2",          None,          False),
        ("MnOH⁺",           "m_MnOH+",         None,          False),
        ("Mn(OH)₂(aq)",     "m_Mn(OH)2",       None,          False),
        ("MnSO₄(aq)",       "m_MnSO4",         None,          False),
        ("Mn(OH)₂(s)",      None,              "si_Mn(OH)2(s)", True),
    ],
}

# ── filter to only species with actual data ────────────────────────────────────
for metal in METAL_SPECIES:
    filtered = []
    for entry in METAL_SPECIES[metal]:
        label, m_col, si_col, is_solid = entry
        if is_solid:
            if si_col and si_col in df.columns and df[si_col].notna().any():
                filtered.append(entry)
        else:
            if m_col and m_col in df.columns and df[m_col].max(skipna=True) > 1e-10:
                filtered.append(entry)
    METAL_SPECIES[metal] = filtered
    kept = [e[0] for e in filtered]
    print(f"{metal}: keeping {kept}")

METAL_TITLE_COLORS = {
    "Al": "#e86e00", "Fe": "#1a8c1a", "Cu": "#6b3a2a",
    "Ni": "#9aaa00", "Co": "#cc44a0", "Mn": "#333333",
}

PALETTES = [
    ["#4477AA","#66CCEE","#228833","#CCBB44","#EE6677","#AA3377","#BBBBBB","#000000"],
    ["#0077BB","#33BBEE","#009988","#EE7733","#CC3311","#EE3377","#BBBBBB","#AA4499"],
    ["#332288","#117733","#44AA99","#88CCEE","#DDCC77","#CC6677","#AA4499","#882255"],
]


def dominant_species(df, species_list):
    """Return integer array of dominant species index for each row."""
    n = len(df)
    scores = np.full((n, len(species_list)), -np.inf)

    for j, (label, m_col, si_col, is_solid) in enumerate(species_list):
        if is_solid:
            if si_col and si_col in df.columns:
                si = df[si_col].astype(float).to_numpy()
                scores[:, j] = np.where(si >= 0, 1e6 + si, -np.inf)
        else:
            if m_col and m_col in df.columns:
                m = df[m_col].astype(float).to_numpy()
                scores[:, j] = np.where(m > 0, np.log10(np.where(m > 0, m, 1e-300)), -np.inf)

    return np.argmax(scores, axis=1)


# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

pH_grid = np.linspace(1, 14, 300)
pe_grid = np.linspace(pe.min() - 1, pe.max() + 1, 300)
PHgrid, PEgrid = np.meshgrid(pH_grid, pe_grid)

for ax, (metal, species_list) in zip(axes.flat, METAL_SPECIES.items()):

    palette = PALETTES[list(METAL_SPECIES.keys()).index(metal) % len(PALETTES)]
    colors  = palette[:len(species_list)]
    cmap    = ListedColormap(colors)
    bounds  = np.arange(-0.5, len(species_list) + 0.5, 1)
    norm    = BoundaryNorm(bounds, cmap.N)

    dom = dominant_species(df, species_list).astype(float)

    try:
        dom_grid = griddata(
            (ph, pe), dom,
            (PHgrid, PEgrid),
            method="nearest"
        )
        ax.pcolormesh(PHgrid, PEgrid, dom_grid, cmap=cmap, norm=norm,
                      alpha=0.35, shading="auto", zorder=0)
    except Exception:
        pass

    ax.scatter(ph, pe, c=dom, cmap=cmap, norm=norm,
               s=10, alpha=0.8, linewidths=0, zorder=2)

    pH_l = np.linspace(1, 14, 200)
    ax.plot(pH_l, -pH_l + 20.75, "k--", lw=0.9, alpha=0.6)
    ax.plot(pH_l, -pH_l,         "k:",  lw=0.9, alpha=0.6)

    patches = [mpatches.Patch(color=colors[j], label=sp[0])
               for j, sp in enumerate(species_list)]
    ax.legend(handles=patches, fontsize=7, loc="upper right",
              framealpha=0.85, title="Dominant species", title_fontsize=7)

    ax.set_title(metal, fontsize=13, fontweight="bold",
                 color=METAL_TITLE_COLORS[metal])
    ax.set_xlabel("pH", fontsize=11)
    ax.set_ylabel("pe", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_xlim(1, 14)

plt.tight_layout()
out = OUTDIR / "fig_pourbaix_speciation.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
