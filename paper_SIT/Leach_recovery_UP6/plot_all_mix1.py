# -*- coding: utf-8 -*-
"""
Created: 2025

Project   : Battery Recycling - Hydrometallurgical Process

@author: Vahid Attari
@email : vahid.attari@nrcan-rncan.gc.ca
Affiliation: Natural Resources Canada (NRCan)
Description:
    Visualization of auto-generated PHREEQC solution blocks using Latin Hypercube Sampling (LHS)
    while keeping all other sections unchanged.

License:
    Creative Commons Attribution 4.0 International (CC BY 4.0)
    https://creativecommons.org/licenses/by/4.0/    
"""

import os
import re
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.legend_handler import HandlerTuple
from scipy.interpolate import griddata

# ---------------- CONFIG ----------------
SAVE_PDF = False

PH_RANGE = (3.0, 5.0)     # x-axis window for pH in many plots
EPS_SI   = 0.10         # |SI| threshold for "incipient precipitation"
TEMP_SLICE = None       # e.g., 80 for single-T panels; None to use all points

# Which variable to contour on pH–T panel: "pe" or "mu"
CONTOUR_VAR = "mu"

# Color-blind friendly palette (Okabe–Ito)
CBF_COLORS = [
    "#0072B2",  # blue (Okabe-Ito)
    "#E69F00",  # orange (Okabe-Ito)
    "#009E73",  # bluish green (Okabe-Ito)
    "#D55E00",  # vermillion (Okabe-Ito)
    "#CC79A7",  # reddish purple (Okabe-Ito)
    "#F0E442",  # yellow (Okabe-Ito)
    "#000000",  # black (Okabe-Ito)

    # ---- Additional color-blind-safe colors ----
    "#56B4E9",  # light blue (Okabe–Ito extension)
    "#0099CC",  # strong cyan
    "#33CC33",  # vivid green (Tol)
    "#CC3311",  # dark reddish orange (Tol)
    "#EE3377",  # rosy magenta (Tol)
    "#0077BB",  # deep blue (Tol)
    "#BBBBBB",  # grey (Tol)
    "#AA4499"   # purple (Tol)
]


def is_constant(series, tol=1e-12):
    # Skip non-numeric columns
    if not np.issubdtype(series.dtype, np.number):
        return False
    return (series.max() - series.min()) < tol



def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path, sep=",", engine="python")
    except Exception:
        df = pd.read_csv(csv_path, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_col(frame: pd.DataFrame, names, contains=None) -> str:
    cols = list(frame.columns)
    for n in names:
        for c in cols:
            if c.strip().lower() == str(n).strip().lower():
                return c
    if contains:
        for c in cols:
            if contains.lower() in c.strip().lower():
                return c
    for c in cols:
        for n in names:
            if str(n).strip().lower() in c.strip().lower():
                return c
    raise KeyError(f"Missing any of {names} (or contains='{contains}') in:\n{cols}")

def optional_col(frame: pd.DataFrame, hint: str):
    for k in frame.columns:
        if k.strip().lower() == hint.strip().lower():
            return k
    for k in frame.columns:
        if hint.strip().lower() in k.strip().lower():
            return k
    return None

# ------------- PHASE & METAL DISCOVERY -------------
PHASE_RENAME = {
    "Gibbsite": "Al(OH)3(s)",
}

def latexify_label(label: str) -> str:
    # Apply display-name overrides before latexification
    label = PHASE_RENAME.get(label, label)
    # Make (s) typeset and digits subscripts
    lab = label
    lab = lab.replace("(s)", "$(s)$")
    lab = re.sub(r"(\d+)", r"$_\1$", lab)
    return lab

def discover_phases(df: pd.DataFrame, name: str = "si_"):
    """Return PHASES list with label, si_col, marker, color from all si_* columns (dedup .1 etc.)."""
    phase_cols = [c for c in df.columns if c.lower().startswith(name)]
    if not phase_cols:
        return []

    def clean(col):
        name = re.sub(r"^si_", "", col, flags=re.IGNORECASE)
        name = re.sub(r"\.?\d*$", "", name).strip()
        return name

    # map clean name -> first seen raw col
    phase_map = {}
    for c in phase_cols:
        cc = clean(c)
        if cc not in phase_map:
            phase_map[cc] = c

    # Build colors/markers
    base_colors = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors) + list(plt.cm.Dark2.colors)
    markers     = ["o","s","^","D","P","X","*","x","+","v","<",">","h","H","p","d"]

    PHASES = []
    for idx, (clean_name, raw_col) in enumerate(phase_map.items()):
        PHASES.append(dict(
            label=latexify_label(clean_name),
            si_col=raw_col,
            marker=markers[idx % len(markers)],
            color=base_colors[idx % len(base_colors)],
        ))
    return PHASES

def discover_metal_free_cols(df: pd.DataFrame):
    """Return list of dicts for any *_free_% columns found (Ni/Co/Mn/Li if present)."""
    candidates = ["Ni_free_%", "Co_free_%", "Mn_free_%", "Li_free_%"]
    found = []
    markers = {"Ni_free_%":"o", "Co_free_%":"s", "Mn_free_%":"^", "Li_free_%":"D"}
    names   = {"Ni_free_%":r"Ni$^{2+}$", "Co_free_%":r"Co$^{2+}$", "Mn_free_%":r"Mn$^{2+}$", "Li_free_%":r"Li$^{+}$"}
    for c in candidates:
        col = optional_col(df, c)
        if col is not None:
            found.append(dict(name=names[c], free_col=col, marker=markers[c]))
    return found

# ------------- PLOTTING UTILITIES -------------
def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def savefig(fig, name: str, OUT_DIR):
    png = OUT_DIR / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    if SAVE_PDF:
        pdf = OUT_DIR / f"{name}.pdf"
        fig.savefig(pdf, bbox_inches="tight")

def add_eh_axis(ax, factor=0.05916):
    ax2 = ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim(ymin*factor, ymax*factor)
    ax2.set_ylabel("Eh (V vs SHE)")
    return ax2

def dedupe_legend(ax, fontsize=9):
    h, l = ax.get_legend_handles_labels()
    by = dict(zip(l, h))
    if by:
        ax.legend(by.values(), by.keys(), fontsize=fontsize, frameon=True, loc='best')

# ------------- PLOTS -------------
def plot_pourbaix(dfr, PHASES, METALS, ph_col, pe_col, t_col):
    pH = dfr[ph_col].astype(float).to_numpy()
    pe = dfr[pe_col].astype(float).to_numpy()
    T  = dfr[t_col].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8.4,6.2))
    sc = ax.scatter(pH, pe, c=T, s=90, edgecolor='k')
    cb = plt.colorbar(sc, ax=ax, pad=0.15)
    cb.set_label("Temperature (°C)")

    # Water stability (25 °C approximate guides)
    gx = np.linspace(PH_RANGE[0], PH_RANGE[1], 400)
    ax.plot(gx, -gx + 7.0, '--', lw=1, label='H$_2$/H$_2$O (~25 °C)')
    ax.plot(gx, 20.75 - gx, '--', lw=1, label='O$_2$/H$_2$O (~25 °C)')

    # Free-ion dominance (≥50%)
    for m in METALS:
        col = m["free_col"]
        if col in dfr.columns:
            pct = dfr[col].astype(float).to_numpy()
            mask = np.isfinite(pct) & (pct >= 50.0)
            if np.any(mask):
                ax.scatter(pH[mask], pe[mask], marker=m["marker"], s=140,
                           facecolors='none', edgecolors='black', linewidths=1.6,
                           label=f'{m["name"]} (free ≥50%)')

    # SI≈0 marks and SI≥0 soft fills
    for phs in PHASES:
        si_col = phs["si_col"]
        if si_col not in dfr.columns: 
            continue
        vals = dfr[si_col].astype(float).to_numpy()
        # Boundary
        #mask_b = np.isfinite(vals) & (np.abs(vals) < EPS_SI)
        #if np.any(mask_b):
        #    ax.scatter(pH[mask_b], pe[mask_b], marker=phs["marker"], s=120,
        #               facecolors='none', edgecolors='black', linewidths=1.4,
        #               label=f'{phs["label"]} (SI≈0)')
        # Oversat region (dot cloud)
        mask_os = np.isfinite(vals) & (vals >= -0.1)
        if np.any(mask_os):
            ax.scatter(pH[mask_os], pe[mask_os], s=36, alpha=0.18, marker='o',
                       facecolors=phs["color"], edgecolors='none',
                       label=f'{phs["label"]} (SI≥0)')

    ax.set_xlabel("pH")
    ax.set_ylabel("pe")
    ax.set_xlim(PH_RANGE)
    #ax.set_title("pe–pH (react rows) with SI & free-ion overlays")
    add_eh_axis(ax)
    ax.grid(True, linestyle='None', lw=0.3, alpha=0.4)
    dedupe_legend(ax)
    return fig


def plot_si_vs_pH(dfr, PHASES, ph_col):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = dfr[ph_col].astype(float).to_numpy()
    legend_entries, legend_labels = [], []

    for i, phs in enumerate(PHASES):
        col = phs["si_col"]
        if col not in dfr.columns:
            continue
        y = dfr[col].astype(float).to_numpy()
        mask_pos = y >= 0
        mask_nonpos = ~mask_pos

        color = CBF_COLORS[i % len(CBF_COLORS)]
        line_neg = line_pos = None

        if mask_nonpos.any():
            line_neg = ax.plot(
                x[mask_nonpos], y[mask_nonpos],
                linestyle='None', marker='o', ms=4, lw=0.8,
                markeredgecolor='k', markeredgewidth=0.5,
                color=color,
                alpha=0.7
            )

        if mask_pos.any():
            line_pos = ax.plot(
                x[mask_pos], y[mask_pos],
                linestyle='None', marker='s', ms=4, lw=0.8,
                markeredgecolor='k', markeredgewidth=0.5,
                color=color,
                alpha=0.7
            )

        handles = tuple(filter(None, [line_neg[0] if line_neg else None,
                                      line_pos[0] if line_pos else None]))
        if handles:
            legend_entries.append(handles)
            legend_labels.append(phs["label"])

    ax.axhline(0.0, ls='--', lw=1, c='k')
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel("pH", fontsize=18)
    ax.set_ylabel("Saturation Index (SI)", fontsize=18)
    ax.set_xlim(PH_RANGE)
    ax.tick_params(axis='both', labelsize=16)

    ax.text(PH_RANGE[0]*1.1, +0.15, "supersaturation")
    ax.text(PH_RANGE[0]*1.1, -0.2, "undersaturation")


    ax.legend(legend_entries, legend_labels,
              handler_map={tuple: HandlerTuple(ndivide=None)},
              frameon=True, loc='lower left', title='Phases')

    return fig


def plot_si_vs_T(dfr, PHASES, t_col):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = dfr[t_col].astype(float).to_numpy()
    legend_entries, legend_labels = [], []

    for i, phs in enumerate(PHASES):
        col = phs["si_col"]
        if col not in dfr.columns:
            continue
        y = dfr[col].astype(float).to_numpy()

        mask_pos = y >= 0
        mask_nonpos = ~mask_pos

        color = CBF_COLORS[i % len(CBF_COLORS)]
        line_neg = line_pos = None

        # Plot SI ≤ 0 (circle markers)
        if mask_nonpos.any():
            line_neg = ax.plot(
                x[mask_nonpos], y[mask_nonpos],
                linestyle='None', marker='o', ms=4, lw=0.8,
                markeredgecolor='k', markeredgewidth=0.5,
                color=color
            )

        # Plot SI > 0 (square markers)
        if mask_pos.any():
            line_pos = ax.plot(
                x[mask_pos], y[mask_pos],
                linestyle='None', marker='s', ms=4, lw=0.8,
                markeredgecolor='k', markeredgewidth=0.5,
                color=color
            )

        # Combine handles for unified legend entry
        handles = tuple(filter(None, [
            line_neg[0] if line_neg else None,
            line_pos[0] if line_pos else None
        ]))
        if handles:
            legend_entries.append(handles)
            legend_labels.append(phs["label"])

    # Reference line at SI = 0
    ax.axhline(0.0, ls='--', lw=1, c='k')

    # Axis formatting
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Saturation Index (SI)")
    ax.grid(True, ls='--', lw=0.4, alpha=0.6)
    ax.set_xlim(np.nanmin(x)*0.5, np.nanmax(x)*1.5)

    # Legend: automatically combine circle/square per phase
    ax.legend(
        legend_entries, legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False, loc='upper right', title='Phases'
    )

    return fig



def plot_free_vs_pH(dfr, METALS, ph_col):
    metals_present = [m for m in METALS if m["free_col"] in dfr.columns]
    if not metals_present:
        return None
    fig, ax = plt.subplots(figsize=(7.2,5.0))
    x = dfr[ph_col].astype(float).to_numpy()
    for m in metals_present:
        y = dfr[m["free_col"]].astype(float).to_numpy()
        ax.plot(x, y, linestyle='None', marker=m["marker"], ms=4, label=m["name"])
    ax.axhline(50.0, ls=':', lw=1, c='k')
    ax.set_xlabel("pH")
    ax.set_ylabel("Free-ion (%)")
    ax.set_xlim(PH_RANGE)
    ax.set_ylim(0, 100)
    #ax.set_title("Free-ion percentage vs pH")
    dedupe_legend(ax)
    return fig

def plot_mu_vs_pH(dfr, ph_col):
    mu_col = optional_col(dfr, "mu")
    if mu_col is None:
        return None

    t_col = optional_col(dfr, "temp") or optional_col(dfr, "temperature")
    if t_col is None:
        return None

    markers = {
        "NMC111": "o",
        "NMC523": "s",
        "NMC532": "^",
        "NMC622": "D",
        "NMC811": "P",
        "NCA":    "X",
        "LCO":    "*"
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    sc_ref = None

    for chem_class, group in dfr.groupby("chem"):

        x = group[ph_col].astype(float).to_numpy()
        y = group[mu_col].astype(float).to_numpy()
        t = group["pe"].astype(float).to_numpy()

        is_sdl = chem_class == "NMC523 - SDL"

        marker = '*' if is_sdl else markers.get(chem_class, "x")

        sc = ax.scatter(
            x, y,
            c=("gold" if is_sdl else t),
            cmap=None if is_sdl else "viridis",
            s=120 if is_sdl else 25,
            alpha=1.0 if is_sdl else 0.7,
            edgecolors="black" if is_sdl else "none",
            linewidths=1.2 if is_sdl else 0,
            marker=marker,
            zorder=5 if is_sdl else 1
        )

        if not is_sdl and sc_ref is None:
            sc_ref = sc  # keep one valid mappable for colorbar

    if sc_ref is not None:
        cbar = plt.colorbar(sc_ref, ax=ax, pad=0.07)
        cbar.set_label("pe")

    ax.set_xlabel("pH")
    ax.set_ylabel("Ionic strength (μ)")
    ax.set_xlim(PH_RANGE)

    handles = []

    for chem_class in dfr["chem"].unique():
        is_sdl = chem_class == "NMC523 - SDL"

        marker = '*' if is_sdl else markers.get(chem_class, "x")

        handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="None",
                markersize=10 if is_sdl else 6,
                markerfacecolor="gold" if is_sdl else "lightgray",
                markeredgecolor="black" if is_sdl else "none",
                markeredgewidth=1.2 if is_sdl else 0,
                label=chem_class
            )
        )

    ax.legend(
        handles=handles,
        title="Chemistry",
        loc="best",
        fontsize=8,
        title_fontsize=10
    )

    return fig


def plot_pH_T_contour(df, x_axis, t_col, var="mu", grid_res=200):
    x = df[x_axis].astype(float)
    y = df[t_col].astype(float)
    z = df[var].astype(float)

    # 1 – Create structured grid
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    XI, YI = np.meshgrid(xi, yi)

    # 2 – Interpolate
    ZI = griddata((x, y), z, (XI, YI), method='linear')

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # --- Scatter-first approach ---
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=z.min(), vmax=z.max())

    sc = ax.scatter(x, y, c=z, cmap=cmap, norm=norm,
                    edgecolor="k", s=25)

    # Use scatter’s colormap for contour
    #cf = ax.contourf(XI, YI, ZI,
    #                 levels=20,
    #                 cmap=cmap,
    #                 norm=norm,
    #                 alpha=0.7
    #                 )

    #ax.contour(XI, YI, ZI, colors='k', linewidths=0.1, alpha=0.75)

    cbar = plt.colorbar(sc)   # or cf — identical either way
    cbar.set_label(var)

    ax.set_xlabel(x_axis)
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)

    return fig


def plot_2Dcontour(
    df,
    x_axis,
    y_axis,
    var="mu",
    *,
    xscale="linear",
    yscale="linear",
    x_range=None,
    y_range=None,
    cscale="linear",
    c_range=None,
    grid_res=200
):
    """
    xscale / yscale: "linear", "log", "symlog"
    cscale (color scale): "linear", "log", "symlog"
    c_range: (vmin, vmax) for colorbar; x_range, y_range to override data bounds
    """
    x = df[x_axis].astype(float).values
    y = df[y_axis].astype(float).values
    z = df[var].astype(float).values

    # Axis ranges
    x_min = x_range[0] if x_range else x.min()
    x_max = x_range[1] if x_range else x.max()
    y_min = y_range[0] if y_range else y.min()
    y_max = y_range[1] if y_range else y.max()

    # Create grid
    xi = np.linspace(x_min, x_max, grid_res)
    yi = np.linspace(y_min, y_max, grid_res)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate
    ZI = griddata((x, y), z, (XI, YI), method="nearest") # linear, nearest, cubic

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # -----------------------------
    # Color normalization
    # -----------------------------
    if c_range:
        vmin, vmax = c_range
    else:
        vmin, vmax = np.nanmin(z), np.nanmax(z)

    if cscale == "log":
        norm = LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
    elif cscale == "symlog":
        norm = SymLogNorm(linthresh=1e-3, vmin=vmin, vmax=vmax)
    else:
        norm = None

    # -----------------------------
    # Scatter plot (points)
    # -----------------------------
    sc = ax.scatter(
        x, y,
        c=z,
        cmap="viridis",
        s=25,
        edgecolor="k",
        norm=norm,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax
    )

    # -----------------------------
    # Contour overlay
    # -----------------------------
    if ZI is not None:
        # Filled contour
        ax.contourf(
            XI, YI, ZI,
            levels=50,
            cmap="viridis",
            norm=norm,
            alpha=0.5
        )

        # --- Add contour lines ---
        contour_levels = [-0.5, 0, 0.5]
        cs = ax.contour(
            XI, YI, ZI,
            levels=contour_levels,
            colors=['red', 'white', 'yellow'],  # choose readable colors
            linewidths=1.0,
            norm=norm
        )

        # Add labels directly on the contour lines
        ax.clabel(cs, fmt='%1.2f', inline=True, fontsize=11)

    # -----------------------------
    # Colorbar
    # -----------------------------
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(var)

    # -----------------------------
    # Axis scaling
    # -----------------------------
    ax.set_xscale(xscale.lower())
    ax.set_yscale(yscale.lower())

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Optional inversion (kept from your original)
    if yscale == "linear":
        ax.invert_yaxis()

    ax.set_xlabel(x_axis, fontsize=14)
    ax.set_ylabel(y_axis, fontsize=14)
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)

    return fig


   



def plot_correlation_heatmap(dfr, core_cols):
    # numeric subset
    num = dfr[core_cols].apply(pd.to_numeric, errors='coerce')
    corr = num.corr()

    fig, ax = plt.subplots(figsize=(10.0, 9.2))
    im = ax.imshow(corr.values, origin='lower', cmap='coolwarm', norm=Normalize(-1, 1))

    # Axis labels
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    # Add correlation values on squares
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha='center', va='center',
                    color='black' if abs(val) < 0.5 else 'white',
                    fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Correlation")

    fig.tight_layout()
    return fig


def plot_upper_triangle_heatmap(dfr, core_cols, pretty_labels=None):
    """Upper-triangle correlation heatmap matching the paper style."""
    num  = dfr[core_cols].apply(pd.to_numeric, errors='coerce')
    corr = num.corr().values
    n    = len(core_cols)
    labels = pretty_labels if pretty_labels else core_cols

    # mask: show only upper triangle (i < j); diagonal left blank
    display = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            display[i, j] = corr[i, j]

    fig, ax = plt.subplots(figsize=(10.5, 7.42))

    cmap = plt.cm.coolwarm
    norm = Normalize(-1, 1)

    for i in range(n):
        for j in range(n):
            val = display[i, j]
            if np.isnan(val):
                continue
            color = cmap(norm(val))
            rect  = plt.Rectangle([j - 0.5, i - 0.5], 1, 1,
                                   facecolor=color, edgecolor='white', lw=1.0)
            ax.add_patch(rect)
            txt_color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=14, color=txt_color, fontweight='bold')

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=15)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=15)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(length=0, pad=6)

    # black frame around the axes area
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.8)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Correlation", fontsize=15)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(1.5)

    fig.tight_layout()
    return fig


def sns_pairplot_decision_maps(df, features, target,
                               n_points=150,
                               save_path="figs/fig8_pairplot_matrix_seaborn.png"):


    df_plot = (df.dropna(axis=1)
                 .sample(n=n_points, random_state=42))
    df_plot = df_plot[df_plot[target] >= -0.5]
    values = df_plot[target].values
    norm = Normalize(values.min(), values.max())

    sns.set_theme(style="ticks", context="notebook", font_scale=1.1)

    g = sns.pairplot(
        df_plot,
        vars=features,
        corner=True,
        diag_kind="kde",
        plot_kws=dict(
            alpha=0.7,
            s=35,
            c=values,
            cmap="viridis",
            norm=norm,
            edgecolor="k",
            linewidth=0.4
        )
    )

    plt.suptitle("Pairplot Decision Maps", fontsize=20, y=1.02)
    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])

    cbar = g.fig.colorbar(sm, ax=g.fig.axes, label=target,
                   fraction=0.02, pad=0.02)
    cbar.ax.set_ylabel(target, fontsize=30)
    cbar.ax.tick_params(labelsize=20)   # fontsize for numbers

    g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def get_top_correlations(dfr, cols, top_n=40, save_path=None):
    """
    Compute the top-N absolute correlations among the given columns.

    Parameters
    ----------
    dfr : pandas.DataFrame
        Input dataframe containing the numeric data.
    cols : list
        List of column names to include in the correlation ranking.
    top_n : int
        Number of top absolute correlations to return.
    save_path : str or Path, optional
        If provided, saves the CSV of results.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: [Var1, Var2, AbsCorr].
    """

    # numeric subset
    num = dfr[cols].apply(pd.to_numeric, errors="coerce")
    corr = num.corr()

    # convert to long format and remove self-correlations
    corr_long = (
        corr.abs()                                     # use absolute correlations
            .where(~np.eye(corr.shape[0], dtype=bool)) # remove diagonal entries
            .stack()                                   # flatten
            .reset_index()
    )
    corr_long.columns = ["Var1", "Var2", "AbsCorr"]

    # sort and keep top N
    top_df = corr_long.sort_values(by="AbsCorr", ascending=False).head(top_n)

    # Save if requested
    #os.makedirs('figs', exist_ok=True)
    if save_path:
        top_df.to_csv(save_path, index=False)
        print(f"[INFO] Saved top-{top_n} correlations to: {save_path}")

    return top_df

def objective_function(si_dict):
    """
    Compute objective:
        f = (SIFe(OH)3 + SIAl(OH)3 + SICu(OH)2)
            - (SIMn(OH)2 + SINi(OH)2 + SICo(OH)2)

    Parameters
    ----------
    si_dict : dict
        Keys must include:
            'SI_Fe(OH)3', 'SI_Al(OH)3', 'SI_Cu(OH)2',
            'SI_Mn(OH)2', 'SI_Ni(OH)2', 'SI_Co(OH)2'

    Returns
    -------
    float
        Objective function value.
    """

    positive_block = (
        si_dict["SI_Fe(OH)3"]
        + si_dict["SI_Al(OH)3"]
        + si_dict["SI_Cu(OH)2"]
    )

    negative_block = (
        si_dict["SI_Mn(OH)2"]
        + si_dict["SI_Ni(OH)2"]
        + si_dict["SI_Co(OH)2"]
    )

    return positive_block - negative_block

def naoh_moles_to_volume(n_moles, stock_conc_M, unit="mL"):
    """
    Convert NaOH moles to volume.

    Parameters
    ----------
    n_moles : float or array-like
        NaOH amount in moles
    stock_conc_M : float
        NaOH stock concentration in mol/L
    unit : str
        'L', 'mL', or 'uL'

    Returns
    -------
    volume : float or ndarray
        NaOH volume in requested units
    """
    if stock_conc_M <= 0:
        raise ValueError("Stock concentration must be positive")

    vol_L = n_moles / stock_conc_M

    scale = {"L": 1.0, "mL": 1e3, "uL": 1e6}
    return vol_L * scale[unit]


# ------------- MAIN -------------
def main():

    #CSV_PATH = Path("DOE_merged_mix2_react.csv")  # change if needed
    
    CSV_PATHs = [Path("speciation_results_merged/DOE_merged_mix1_react.csv")] #, Path("DOE_merged_mix1_react.csv")];  # change if needed
    OUT_DIRs  = [Path("figs_mix1")]

    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------")

    for CSV_PATH,OUT_DIR in zip(CSV_PATHs,OUT_DIRs):

        print(f"\n\nProcessing {CSV_PATH} ...")
        print(f"Output directory: {OUT_DIR}")

        ensure_outdir(OUT_DIR)

        dfr = load_results(CSV_PATH)
        dfr = dfr.dropna()
        # Filter react rows if present
        #if "state" in df.columns:
        #    dfr = df[df["state"].astype(str).str.strip().str.lower() == "react"].copy()
        #    if dfr.empty:
        #        dfr = df.copy()
        #else:
        #    dfr = df.copy()

        NaOH_mole = find_col(dfr, ["NaOH_mole","NaOH_moles","NaOH_mol"])
        NaOH_stock = find_col(dfr, ["NaOH_stock"])
        BattLeachate_volume = find_col(dfr, ["BattLeachate_volume"])
        ph_col   = find_col(dfr, ["pH","ph"])
        pe_col   = find_col(dfr, ["pe"])
        t_col    = find_col(dfr, ["T_C","temp","temperature"], contains="temp")
        mol_cols = discover_phases(dfr, "_mol")
        PHASES = discover_phases(dfr)
        METALS = discover_metal_free_cols(dfr)
        si_cols = dfr.filter(like="si_").columns
        m_cols = dfr.filter(like="m_").columns
        d_cols = dfr.filter(like="d_").columns
        phase_cols = [name[2:] for name in d_cols]

        C_stock = float(dfr[NaOH_stock].iloc[0])

        dfr["NaOH_volume"] = naoh_moles_to_volume(
            dfr[NaOH_mole].astype(float),
            C_stock,
            unit="mL"
        )

        print(dfr["NaOH_volume"])

        # Optional temperature slice
        #if TEMP_SLICE is not None:
        #    mask = np.isclose(dfr[t_col].astype(float).to_numpy(), float(TEMP_SLICE), atol=1.0)
        #    if mask.any():
        #        dfr = dfr[mask].copy()

        # 1) Pourbaix-style panel
        fig = plot_pourbaix(dfr, PHASES, METALS, ph_col, pe_col, t_col)
        savefig(fig, "fig1_pe_pH_pourbaix", OUT_DIR)
        plt.close(fig)

        # 2) SI vs pH
        fig = plot_si_vs_pH(dfr, PHASES, ph_col)
        savefig(fig, "fig2_SI_vs_pH", OUT_DIR)
        plt.close(fig)

        ## 3) SI vs Temperature
        # fig = plot_si_vs_T(dfr, PHASES, t_col)
        # savefig(fig, "fig3_SI_vs_T", OUT_DIR)
        # plt.close(fig)

        # 4) Free-ion % vs pH
        fig = plot_free_vs_pH(dfr, METALS, ph_col)
        if fig is not None:
            savefig(fig, "fig4_free_percent_vs_pH", OUT_DIR)
            plt.close(fig)

        # 5) Ionic strength vs pH
        fig = plot_mu_vs_pH(dfr, ph_col)
        if fig is not None:
            savefig(fig, "fig5_mu_vs_pH", OUT_DIR)
            plt.close(fig)

        # 6) pH–T contour of pe (or μ)
        #fig = plot_pH_T_contour(dfr, ph_col, t_col, var=CONTOUR_VAR)
        #if fig is not None:
        #    savefig(fig, f"fig6_{CONTOUR_VAR}_contour_pH_T")
        #    plt.close(fig)

        # 7) Correlation heatmap — fixed 9-variable upper-triangle
        HEATMAP_COLS = [
            "Al_sulfate_mol", "Fe_sulfate_mol", "Cu_sulfate_mol",
            "Ni_sulfate_mol", "Co_sulfate_mol", "Mn_sulfate_mol",
            ph_col, pe_col, optional_col(dfr, "mu"),
            "m_SO4-2",
        ]
        HEATMAP_LABELS = [
            "Al sulfate mol", "Fe sulfate mol", "Cu sulfate mol",
            "Ni sulfate mol", "Co sulfate mol", "Mn sulfate mol",
            "pH", "pe", "Ionic Strength",
            "SO₄²⁻ (free)",
        ]
        # keep only columns that actually exist
        hm_cols   = [c for c in HEATMAP_COLS if c and c in dfr.columns]
        hm_labels = [HEATMAP_LABELS[i] for i, c in enumerate(HEATMAP_COLS)
                     if c and c in dfr.columns]

        # original full heatmap
        fig = plot_correlation_heatmap(dfr, hm_cols)
        savefig(fig, "fig7_correlation_heatmap", OUT_DIR)
        plt.close(fig)

        # upper-triangle styled heatmap
        fig = plot_upper_triangle_heatmap(dfr, hm_cols, pretty_labels=hm_labels)
        savefig(fig, "fig7b_correlation_heatmap_upper", OUT_DIR)
        plt.close(fig)

        # also keep full correlation list for analysis
        core_cols = hm_cols[:]

        top40 = get_top_correlations(dfr, core_cols, top_n=40,
                                save_path=f"{OUT_DIR}/top40_correlations.csv")

        print("\n===== TOP 40 ABSOLUTE CORRELATIONS =====\n")
        print(top40.to_string(index=False))

        X = core_cols
        y = "si_Ni(OH)2(s)"   #"si_Al(OH)3(am)" #,si_Fe(OH)3(a),si_Cu(OH)2(s),si_Co(OH)2(s),si_Ni(OH)2(s),si_Mn(OH)2(s),Solution_ID,a_Ni2+,m_Ni2+,a_Co2+,m_Co2+,a_Mn2+,m_Mn2+,a_Li+,m_Li+,Description
        # 8) Pairplot decision maps using seaborn
        #df_for_kde = dfr.loc[:, ~dfr.apply(is_constant)]
        #X_filtered = [col for col in X if col in df_for_kde.columns]
        #y_filtered = [col for col in y if col in df_for_kde.columns]
        #sns_pairplot_decision_maps(df_for_kde, X_filtered, y_filtered,
        #                           n_points=300,
        #                           save_path="figs/fig8_pairplot_matrix_seaborn.png")

        print(f"Done. Figures saved under: {OUT_DIR.resolve()}")

        print(dfr.columns)
        dfr["OS_Fe"] = dfr["si_Fe(OH)3(s)"] >= -0.02
        dfr["OS_Al"] = dfr["si_Gibbsite"] >= -0.02 
        dfr["OS_Ni"] = dfr["si_Ni(OH)2(s)"] >= -0.02
        dfr["OS_Mn"] = dfr["si_Mn(OH)2(s)"] >= -0.02
        dfr["OS_Co"] = dfr["si_Co(OH)2(s)"] >= -0.02
        dfr["OS_Cu"] = dfr["si_Cu(OH)2(s)"] >= -0.02

        print(dfr.columns)

        targets = ["OS_Fe","OS_Al","OS_Cu","OS_Co","OS_Ni","OS_Mn"]
        dfr["Joint"] = dfr[targets].sum(axis=1)












        dfr["objective1"] = (
            (dfr["d_Fe(OH)3(s)"] + dfr["d_Gibbsite"] + dfr["d_Cu(OH)2(s)"]) 
        )

        dfr["objective2"] = (
            (dfr["d_Fe(OH)3(s)"] + dfr["d_Gibbsite"] + dfr["d_Cu(OH)2(s)"]) / ( dfr["d_Mn(OH)2(s)"] + dfr["d_Ni(OH)2(s)"] + dfr["d_Co(OH)2(s)"] )
        )

        dfr["objective3"] = (
            dfr["si_Fe(OH)3(s)"] + dfr["si_Gibbsite"] + dfr["si_Cu(OH)2(s)"]
            - dfr["si_Mn(OH)2(s)"] - dfr["si_Ni(OH)2(s)"] - dfr["si_Co(OH)2(s)"]
        )



        fig = plt.figure(figsize=(6.2, 4.0))
        plt.plot(dfr["NaOH_volume"], dfr[ph_col], linestyle='None', marker='s', ms=2, lw=1,
                 markeredgecolor='k', color='blue')
        plt.xlabel("NaOH Volume (mL)")
        plt.ylabel("pH")
        savefig(fig, "fig16_NaOHVol_vs_pH", OUT_DIR)
        plt.close(fig)

        fig = plt.figure(figsize=(6.2, 4.0))
        for d_col in phase_cols:
            if d_col in dfr.columns:
                plt.semilogy(dfr["NaOH_volume"], dfr[d_col], linestyle='--', marker='s', ms=2, lw=2,
                             markeredgecolor='k', markeredgewidth=0.2, label=d_col)
        plt.xlabel("NaOH Volume (mL)")
        plt.ylabel("Mass (mol)")
        plt.legend()
        savefig(fig, "fig17_NaOHVol_vs_Precipitation_Moles", OUT_DIR)
        plt.close(fig)


        fig, ax1 = plt.subplots(figsize=(6.2, 4.0))

        # --- Left axis: pH ---
        ax1.plot(
            dfr["NaOH_volume"], dfr[ph_col],
            linestyle='None', marker='s', ms=2, lw=1,
            markeredgecolor='k', color='blue', label="pH"
        )
        ax1.set_xlabel("NaOH Volume (mL)")
        ax1.set_ylabel("pH", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # --- Right axis: Mass of phases ---
        ax2 = ax1.twinx()

        line_styles = ['-', '--', '-.', ':',
               (0, (1, 1)), (0, (3, 1, 1, 1)),
               (0, (5, 1)), (0, (3, 5, 1, 5)),
               (0, (5, 5)), (0, (1, 5))]

        for i, d_col in enumerate(d_cols):
            if d_col in dfr.columns:
                ax2.plot(
                    dfr["NaOH_volume"], dfr[d_col],
                    linestyle='None', #line_styles[i % len(line_styles)],
                    lw=2,
                    #markeredgecolor='k', markeredgewidth=0.2,
                    label=d_col
                )


        ax2.set_ylabel("Mass (mol)")
        ax2.set_yscale("log")

        # --- Combined legend ---
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=11)
        plt.tight_layout()
        savefig(fig, "fig18_NaOHVol_pH_and_PhaseMass", OUT_DIR)
        plt.close(fig)


        dfr_sorted = dfr.sort_values("NaOH_volume")
        dAlk_dV = np.diff(dfr_sorted['Alk']) / np.diff(dfr_sorted['NaOH_volume'])
        fig = plt.figure(figsize=(6.2, 4.0))
        plt.semilogy(
            dfr_sorted["NaOH_volume"].iloc[1:], dAlk_dV,
            linestyle='None', marker='s', ms=2, lw=2,
            markeredgecolor='k', markeredgewidth=0.2
        )
        plt.xlabel("NaOH Volume (mL)")
        plt.ylabel(r"$\Delta$(Alkalinity)/$\Delta$(V) (mol/L/mL)")
        plt.tight_layout()
        savefig(fig, "fig19_NaOHVol_vs_DeltaAlk", OUT_DIR)
        plt.close(fig)

        

        results = pd.DataFrame()
        metals = ["Ni", "Co", "Cu", "Mn", "Fe", "Al"]

        # stoichiometric factors from sulfate salts
        metal_factors = {
            "Al": 2,   # Al2(SO4)3
            "Fe": 2,   # Fe2(SO4)3
            "Cu": 1,   # CuSO4
            "Co": 1,   # CoSO4
            "Ni": 1,   # NiSO4
            "Mn": 1,   # MnSO4
        }

        for metal in metals:
            # convert sulfate input → actual metal moles
            initial = dfr[f"{metal}_sulfate_mol"] * metal_factors[metal]
            
            # totals are mol/kgw → convert to mol
            dissolved = dfr[metal] * dfr["mass_H2O"]
            percent = (dissolved / initial) * 100

            results[f"{metal}_initial"] = initial
            results[f"{metal}_dissolved"] = dissolved
            results[f"{metal}_pct"] = percent

        dfr = pd.concat([dfr, results], axis=1)

        # --- Global style ---
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
        })

        fig, ax = plt.subplots(figsize=(6.5, 5.5))  # more square for equal axes

        # --- Scatter plots (consistent marker style) ---
        scatter_kwargs = dict(s=45, alpha=0.6, linewidth=0.6)

        ax.scatter(dfr["Al_initial"], dfr["Al_dissolved"], label="Al", edgecolor="k", **scatter_kwargs)
        ax.scatter(dfr["Fe_initial"], dfr["Fe_dissolved"], label="Fe", edgecolor="b", **scatter_kwargs)
        ax.scatter(dfr["Cu_initial"], dfr["Cu_dissolved"], label="Cu", edgecolor="g", **scatter_kwargs)
        ax.scatter(dfr["Co_initial"], dfr["Co_dissolved"], label="Co", edgecolor="r", **scatter_kwargs)
        ax.scatter(dfr["Ni_initial"], dfr["Ni_dissolved"], label="Ni", edgecolor="m", **scatter_kwargs)
        ax.scatter(dfr["Mn_initial"], dfr["Mn_dissolved"], label="Mn", edgecolor="c", **scatter_kwargs)

        # --- Equal axes + same limits ---
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

        # --- 1:1 reference line ---
        ax.plot(lims, lims, linestyle="--", color="black", linewidth=1, alpha=0.7)

        # --- Labels ---
        ax.set_xlabel("Initial moles (from sulfate input)")
        ax.set_ylabel("Dissolved moles (from speciation output)")

        # --- Grid (subtle) ---
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

        # --- Legend (clean placement) ---
        ax.legend(frameon=False, ncol=2)

        # --- Tight layout ---
        plt.tight_layout()

        # --- Save (high-res + vector) ---
        plt.savefig(f"{OUT_DIR}/fig15_dissolved_vs_initial.png", dpi=300, bbox_inches="tight")
        #plt.savefig(f"{OUT_DIR}/fig15_dissolved_vs_initial.pdf", bbox_inches="tight")   
        plt.close(fig)

        


                
        #         # must contain metal name
        #         if metal not in col:
        #             continue
                
        #         # exclude solids (e.g., (s))
        #         if "(s)" in col:
        #             continue
                
        #         # exclude duplicate columns like .1
        #         if re.search(r"\.\d+$", col):
        #             continue
                
        #         cols.append(col)
            
        #     return cols


        # for metal in metals:
            
        #     # initial moles from sulfate input
        #     sulfate_col = f"{metal}_sulfate_mol"
            
        #     if sulfate_col not in dfr.columns:
        #         print(f"Warning: {sulfate_col} not found")
        #         continue
            
        #     initial = dfr[sulfate_col]
            
        #     # aqueous species
        #     aq_cols = get_aqueous_columns(dfr, metal)
            
        #     if len(aq_cols) == 0:
        #         print(f"Warning: No aqueous species found for {metal}")
        #         continue
            
        #     dissolved = dfr[aq_cols].sum(axis=1)
            
        #     # percentage in solution
        #     percent = (dissolved / initial) * 100
            
        #     results[f"{metal}_pct"] = percent

        # dfr = pd.concat([dfr, results], axis=1)
        
        
        print('dfr shape after filtering:', dfr.shape)
        print('dfr shape after filtering:', dfr['Al_pct'])
        print('dfr shape after filtering:', dfr['Fe_pct'])

        dfr["pH_bin"] = pd.cut(dfr["pH"], bins=4)

        summary = (
            dfr.groupby("pH_bin")
            .agg(
                pH_mid=("pH", "mean"),
                Al_min=("Al_pct", "min"),
                Al_max=("Al_pct", "max"),
                Al_p10=("Al_pct", lambda x: x.quantile(0.1)),
                Al_p90=("Al_pct", lambda x: x.quantile(0.9)),
                Al_med=("Al_pct", "median"),
            )
            .dropna()
            .sort_values("pH_mid")
        )

        # fig, ax = plt.subplots()

        # ax.fill_betweenx(summary["pH_mid"], summary["Al_min"], summary["Al_max"], alpha=0.2)
        # ax.fill_betweenx(summary["pH_mid"], summary["Al_p10"], summary["Al_p90"], alpha=0.5)
        # ax.plot(summary["Al_med"], summary["pH_mid"], color="k", lw=2)

        # ax.set_xlabel("Al dissolution (%)")
        # ax.set_ylabel("pH")
        # ax.invert_yaxis()
        # ax.legend(["Median", "10–90%", "Min–Max"])
        # plt.close(fig)

        ratio = dfr["Fe"] / (2 * dfr["Fe_sulfate_mol"])
        print(ratio.describe())

        df_long = dfr.melt(
            id_vars=["pH"],
            value_vars=["Al_pct", "Fe_pct", "Ni_pct", "Co_pct", "Cu_pct", "Mn_pct"],
            var_name="metal",
            value_name="pct"
        )
        df_long["pct_dev"] = df_long["pct"] - 100

        # clean metal names
        df_long["metal"] = df_long["metal"].str.replace("_pct", "")


        fig, ax = plt.subplots(figsize=(8, 5))

        sns.violinplot(
            data=df_long,
            x="metal",
            y="pH",
            inner="quartile",
            linewidth=1,
            linecolor="k",
            ax=ax
        )

        sns.pointplot(
            data=df_long,
            x="metal",
            y="pH",
            estimator="mean",
            color="black",
            markers="D",
            ax=ax
        )

        sns.stripplot(
            data=df_long,
            x="metal",
            y="pH",
            hue="pct_dev",
            palette="viridis",
            size=4,
            ax=ax
        )

        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/violin_plot.png", dpi=300, bbox_inches="tight")

        # # # overlay points colored by pct
        # sns.stripplot(
        #     data=df_long,
        #     x="metal",
        #     y="pct",
        #     #hue="pH",
        #     palette="viridis",
        #     size=6,
        #     ax=ax
        # )

        fig, ax = plt.subplots(figsize=(5, 5))

        metals = ["Al_pct", "Fe_pct", "Ni_pct", "Co_pct", "Cu_pct", "Mn_pct"]
        markers = ["o", "s", "^", "D", "v", "P"]  # different markers

        for m, mk in zip(metals, markers):
            ax.scatter(
                dfr["pH"],
                dfr[m],
                label=m.replace("_pct", ""),
                marker=mk,
                s=50,
                alpha=0.8
            )

        # reference line at 100%
        ax.axhline(100, linestyle="--", color="k", linewidth=1)

        # set y-range
        ax.set_ylim(99.5, 100.5)

        # labels
        ax.set_xlabel("pH",fontsize=18)
        ax.set_ylabel("Dissolution (%)",fontsize=18)
        ax.legend(title="Metal", fontsize=14, ncol=3,
                loc="upper center")
        ax.tick_params(axis='both', labelsize=16)

        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/metal_dissolution_vs_pH.png", dpi=300, bbox_inches="tight")
        #fig.tight_layout()




if __name__ == "__main__":
    main()


