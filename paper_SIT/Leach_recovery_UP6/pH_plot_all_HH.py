import pandas as pd
import glob
import os
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
from load_data import read_multisheet_excel,read_multisheet_with_metadata
from pathlib import Path

# ==============================================================

shutil.rmtree("HH_results_backup", ignore_errors=True)
fldr = "HH_results_backup"
os.makedirs(fldr, exist_ok=True)

# Example usage:
#BASE_DIR = Path(__file__).resolve().parent
#file_path = BASE_DIR / "../Telescope Data and PPTs/2025_HTE_sample_data_updated4.xlsx"
file_path = "../2025_HTE_sample_data_updated4.xlsx"
#print(file_path)
ExpData_dict, meta_df = read_multisheet_with_metadata(file_path)

# =====================================================

mpl.rcParams.update({
    # Global font
    "font.size": 14,

    # Frame (axes spines)
    # Axes
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "axes.linewidth": 1.5,

    # Ticks
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,

    # Minor ticks on by default
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    # Legend
    "legend.fontsize": 12,
})

# def total_metal_moles_from_icp(exp_mgL, volume_mL):
#     """
#     Convert ICP metal concentrations (mg/L) to total moles
#     in a vial of known volume.

#     Parameters
#     ----------
#     exp_mgL : dict
#         {metal_symbol: mg/L}
#     volume_mL : float
#         Sample volume in mL

#     Returns
#     -------
#     dict
#         {metal_symbol: total mol}
#     """

#     atomic_wt = {
#         "Al 396.152 nm ppm": 26.9815, # MW:342.15, #
#         "Co": 58.933,
#         "Cu": 63.546,
#         "Fe": 55.845,
#         "Li": 6.94,
#         "Mn": 54.938,
#         "Ni": 58.693,
#     }

#     volume_L = volume_mL / 1000.0

#     total_mol = {}

#     for metal, mgL in exp_mgL.items():
#         mol = (mgL * volume_L) / (atomic_wt[metal] * 1000)
#         total_mol[metal] = mol

#     return total_mol

def total_metal_moles_from_icp(exp_mgL, volume_mL):
    """
    Convert ICP metal concentrations (mg/L) to total moles
    in a vial of known volume.

    Parameters
    ----------
    exp_mgL : dict
        {metal_symbol: mg/L}
    volume_mL : float
        Sample volume in mL

    Returns
    -------
    dict
        {metal_symbol: total mol}
    """

    atomic_wt = {
        #"Al": 26.9815, # MW:342.15, #
        "Al 396.152 nm ppm": 26.9815, # MW:342.15, #
        "Co 238.892 nm ppm": 58.933,
        "Cu": 63.546,
        "Fe 238.204 nm ppm": 55.845,
        "Li": 6.94,
        "Mn 257.610 nm ppm": 54.938,
        "Ni": 58.693,
    }

    volume_L = volume_mL / 1000.0

    total_mol = {}

    for metal, mgL in exp_mgL.items():
        mol = (mgL * volume_L) / (atomic_wt[metal] * 1000)
        total_mol[metal] = mol

    return total_mol

# ------------- PHASE & METAL DISCOVERY -------------
def latexify_label(label: str) -> str:
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

# ==============================================================
# Function to robustly load simulation result CSVs
# ==============================================================
def load_results(pattern, prefix):
    all_files = glob.glob(os.path.join(results_path, pattern))
    frames = []

    for file in all_files:
        sample_id = len(frames) + 1

        try:
            # 🔹 Read space-delimited files with flexible spacing
            df = pd.read_csv(file, delim_whitespace=True, engine="python", encoding="utf-8-sig")
            df["SampleID"] = sample_id

            if df.empty:
                print(f"⚠️ Empty file: {file}")
                continue

            # 🔹 Normalize column names
            df.columns = (
                df.columns.str.strip()
                .str.replace('"', '', regex=False)
                .str.replace("'", '', regex=False)
            )

            # 🔹 Fix known header merge issue (PHREEQC spacing bug)
            df.columns = df.columns.str.replace(
                "si_Co\\(OH\\)2\\(s\\)si_Ni\\(OH\\)2\\(s\\)",
                "si_Co(OH)2(s) si_Ni(OH)2(s)",
                regex=True,
            )

            # 🔹 Verify and select rows
            if "state" in df.columns:
                #df = df[df["state"].astype(str).str.contains("react", case=False, na=False)]
                if df.empty:
                    df = df.tail(1)  # fallback to last row if no "react" found
            else:
                print(f"⚠️ 'state' not found after cleanup in: {os.path.basename(file)}")
                df = df.tail(1)

            frames.append(df)

        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            continue

    if not frames:
        print(f"❗ No valid files found for {pattern}")
        return pd.DataFrame()

    # Drop empty columns created by stray delimiters
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean whitespace in string values
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    print(f"✅ Loaded {len(frames)} files for pattern {pattern}")
    return pd.concat(frames, ignore_index=True)

# ==============================================================
# Function to calculate NaOH volume from moles and stock concentration
# ==============================================================
def naoh_moles_to_mL(n_moles, stock_conc_M):
    """
    Convert NaOH moles to volume in mL.

    Parameters
    ----------
    n_moles : float or array-like
        Amount of NaOH (mol)
    stock_conc_M : float
        NaOH stock concentration (mol/L)

    Returns
    -------
    volume_mL : float or ndarray
        NaOH volume in mL
    """
    if stock_conc_M <= 0:
        raise ValueError("NaOH stock concentration must be positive")

    return (n_moles / stock_conc_M) * 1e3

# ==============================================================
# Experiment folders
# ==============================================================
fls = [
    "speciation_results_merged/DOE_merged_mix2_react.csv",
    "speciation_results_merged/DOE_merged_mix2_eq_react.csv",
]

lbs =[
    "Seperate Beakers - precipitation now allowed", 
    "Seperate Beakers - precipitation allowed", 
]

# Style definitions
marker_styles = [
    "o",   # circle
    "s",   # square
    "^",   # triangle_up
    "v",   # triangle_down
    "<",   # triangle_left
    ">",   # triangle_right
    "D",   # diamond
    "d",   # thin_diamond
    "p",   # pentagon
    "h",   # hexagon1
    "H",   # hexagon2
    "*",   # star
    "X",   # filled_x
    "P",   # filled_plus
    "+",   # plus
    "x",   # x
    "1",   # tri_down (tick-style)
    "2",   # tri_up
    "3",   # tri_left
    "4",   # tri_right
    "|",   # vertical line
    "_",   # horizontal line
]
line_styles   = ["None", "-", "-", "-"] # first two: no line
line_colors = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
    "#999999",  # grey
    "#117733",  # dark green (high contrast)
]

# ==============================================================
# Primary system responses (what we measured)
# ==============================================================

plt.figure(figsize=(7, 5))
for i, fl in enumerate(fls):
    out_dir = os.path.join(fl)
    fname = out_dir #os.path.join(out_dir, "DOE_merged_mix2_eq_react.csv")

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    ph_col = find_col(df, ["pH","ph"])

    C_stock = float(df[NaOH_stock].iloc[0])

    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    df = df.dropna(subset=[NaOH_mole, ph_col, "Al_sulfate_mol"])

    plt.plot(
        df[NaOH_mole],
        df[ph_col],
        marker=marker_styles[i],
        linestyle='None', #line_styles[i],
        #c=line_colors[i],
        #c=df['Al_sulfate_mol'],#line_colors[i],
        #cmap='viridis',   # optional colormaps
        markersize=4,
        #s=20,
        label=lbs[i],
        alpha=0.5,
    )

    # Convert tick labels from mol → mL
    C_stock = float(df[NaOH_stock].iloc[0])  # already defined in your loop
    #print(C_stock)
    #pause

    # Top axis: mL
    def mol_to_mL(x):
        return naoh_moles_to_mL(x, C_stock)  #/1000

    def mL_to_mol(x):
        return x * 1e-3 * C_stock

    # Bottom axis
    ax = plt.gca()
    #ax.set_xscale("log")
    ax.set_xlabel("NaOH added (mole NaOH per L leachate)")
    # Top axis
    secax = ax.secondary_xaxis("top", functions=(mol_to_mL, mL_to_mol))
    #secax.set_xscale("log")
    secax.set_xlabel("NaOH added (mL NaOH per L leachate)")

    mol_ticks = ax.get_xticks()
    ml_ticks = naoh_moles_to_mL(mol_ticks, C_stock)

    #plt.yscale("log")
    #plt.xscale("log")

    #plt.xlabel("NaOH volume added (mole)")
    plt.ylabel("pH")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{fldr}/NaOH_vs_pH_all_experiments_{i}.png", dpi=300)
    plt.show()      



# ==============================================================
# ==============================================================

chem_list = df["chem"].unique().tolist()
#df["marker"] = df["chem"].map(chem_markers)

chem_markers = {
    "LCO": "o",     # circle
    "NCA": "D",     # diamond
    "NMC111": "v",  # triangle down
    "NMC523": "P",  # plus filled
    "NMC532": "X",  # x filled
    "NMC622": "s",  # square
    "NMC811": "^"   # triangle up
}

plt.figure(figsize=(7, 5))

for i, fl in enumerate(fls[1:]):
    fname = os.path.join(fl)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    # --- column detection ---
    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    ph_col = find_col(df, ["pH","ph"])

    # --- stock concentration ---
    C_stock = float(df[NaOH_stock].iloc[0])

    # --- convert to volume ---
    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    # --- clean + sort ---
    df = df.dropna(subset=[NaOH_mole, ph_col, "Al_sulfate_mol", "Gibbsite", "Cu(OH)2(s)"])
    df = df.sort_values(by=NaOH_mole)

    for col, color in zip(["Gibbsite", "Fe(OH)3(s)", "Cu(OH)2(s)"], ["red","blue","orange"]):
        if col not in df.columns:
            print(f"⚠️ Missing expected column '{col}' in {fname}")
            df[col] = 0.0  # add dummy column to avoid errors

        x = df[NaOH_mole].values
        y = df[col].values

        #color = f"C{i}"

        # =========================
        # Rolling min–max envelope
        # =========================
        window = 7   # adjust (5–15)

        y_series = pd.Series(y)

        y_min = y_series.rolling(window, center=True).min()
        y_max = y_series.rolling(window, center=True).max()

        # fill edges
        y_min = y_min.bfill().ffill()
        y_max = y_max.bfill().ffill()

        # =========================
        # Plot
        # =========================

        # shaded envelope
        #plt.fill_between(
        #    x,
        #    y_min,
        #    y_max,
        #    color=color,
        #    alpha=0.25,
        #    #label=col
        #)

        # raw markers on top
        for chem_type, subdf in df.groupby("chem"):

            is_sdl = "SDL" in str(chem_type)

            plt.plot(
                subdf[NaOH_mole],
                subdf[col],
                linestyle='None',
                #marker='*' if is_sdl else chem_markers.get(chem_type, "o"),
                marker='*' if is_sdl else chem_markers.get(str(chem_type).replace("-SDL", ""), "o"),
                markersize=14 if is_sdl else 4,
                color=color,  # keep same color class
                markeredgecolor='black' if is_sdl else None,
                markeredgewidth=1.1 if is_sdl else 0,
                alpha=1.0 if is_sdl else 0.6,
                zorder=5 if is_sdl else 1,
                label=f"{col} | {chem_type}"
            )

# =========================
# Axes (outside loop)
# =========================
ax = plt.gca()
ax.set_yscale("log")
ax.set_xlabel("NaOH added (mole NaOH per L leachate)")
ax.set_ylabel("Phase (mol/kgw)")

# --- secondary axis ---
def mol_to_mL(x):
    return naoh_moles_to_mL(x, C_stock)

def mL_to_mol(x):
    return x * 1e-3 * C_stock

secax = ax.secondary_xaxis("top", functions=(mol_to_mL, mL_to_mol))
secax.set_xlabel("NaOH added (mL NaOH per L leachate)")

# =========================
# Final formatting
# =========================
plt.legend(
    fontsize=7,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0
)
plt.grid(False)
plt.tight_layout()

plt.savefig(f"{fldr}/NaOH_vs_PhaseComp_all_experiments_garbage.png", dpi=300)
plt.show()


# ==============================================================
# ==============================================================




plt.figure(figsize=(7, 5))

for i, fl in enumerate(fls[1:]):
    fname = os.path.join(fl)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    # --- column detection ---
    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    ph_col = find_col(df, ["pH","ph"])

    # --- stock concentration ---
    C_stock = float(df[NaOH_stock].iloc[0])

    # --- convert to volume ---
    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    # --- clean + sort ---
    df = df.dropna(subset=[NaOH_mole, ph_col, "chem", "Al_sulfate_mol", "Gibbsite", "Cu(OH)2(s)"])
    df = df.sort_values(by=NaOH_mole)

    for col, color in zip(["Ni(OH)2(s)", "Mn(OH)2(s)", "Co(OH)2(s)"], ["red","blue","orange"]):
        if col not in df.columns:
            print(f"⚠️ Missing expected column '{col}' in {fname}")
            df[col] = 0.0  # add dummy column to avoid errors

        x = df[NaOH_mole].values
        y = df[col].values

        # =========================
        # Rolling min–max envelope
        # =========================
        window = 7   # adjust (5–15)

        y_series = pd.Series(y)

        y_min = y_series.rolling(window, center=True).min()
        y_max = y_series.rolling(window, center=True).max()

        # fill edges
        y_min = y_min.bfill().ffill()
        y_max = y_max.bfill().ffill()

        # =========================
        # Plot
        # =========================

        # shaded envelope
        #plt.fill_between(
        #    x,
        #    y_min,
        #    y_max,
        #    color=color,
        #    alpha=0.25,
        #    #label=col
        #)

        # raw markers on top
        for chem_type, subdf in df.groupby("chem"):

            is_sdl = "SDL" in str(chem_type)
            
            plt.plot(
                subdf[NaOH_mole],
                subdf[col],
                linestyle='None',
                marker='*' if is_sdl else chem_markers.get(chem_type, "o"),
                markersize=14 if is_sdl else 4,
                color=color,  # keep same color class
                markeredgecolor='black' if is_sdl else None,
                markeredgewidth=1.1 if is_sdl else 0,
                alpha=1.0 if is_sdl else 0.6,
                zorder=5 if is_sdl else 1,
                label=f"{col} | {chem_type}"
            )


# =========================
# Axes (outside loop)
# =========================
ax = plt.gca()
ax.set_yscale("log")
ax.set_xlabel("NaOH added (mole NaOH per L leachate)")
ax.set_ylabel("Phase (mol/kgw)")

# --- secondary axis ---
def mol_to_mL(x):
    return naoh_moles_to_mL(x, C_stock)

def mL_to_mol(x):
    return x * 1e-3 * C_stock

secax = ax.secondary_xaxis("top", functions=(mol_to_mL, mL_to_mol))
secax.set_xlabel("NaOH added (mL NaOH per L leachate)")

# =========================
# Final formatting
# =========================

plt.legend(
    fontsize=7,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0
)
plt.grid(False)
plt.tight_layout()

plt.savefig(f"{fldr}/NaOH_vs_PhaseComp_all_experiments_NMC.png", dpi=300)
plt.show()


# ==============================================================
# ==============================================================

plt.figure(figsize=(7, 5))

for i, fl in enumerate(fls[1:]):
    fname = os.path.join(fl)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    # --- column detection ---
    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    ph_col = find_col(df, ["pH","ph"])

    # --- stock concentration ---
    C_stock = float(df[NaOH_stock].iloc[0])

    # --- convert to volume ---
    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    # --- clean + sort ---
    df = df.dropna(subset=[NaOH_mole, ph_col, "Al_sulfate_mol", "Gibbsite", "Cu(OH)2(s)"])
    df = df.sort_values(by=NaOH_mole)

    for col, color in zip(["Al", "Fe", "Cu"], ["red","blue","orange"]):
        if col not in df.columns:
            print(f"⚠️ Missing expected column '{col}' in {fname}")
            df[col] = 0.0  # add dummy column to avoid errors

        x = df[NaOH_mole].values
        y = df[col].values

        #color = f"C{i}"

        # =========================
        # Rolling min–max envelope
        # =========================
        window = 7   # adjust (5–15)

        y_series = pd.Series(y)

        y_min = y_series.rolling(window, center=True).min()
        y_max = y_series.rolling(window, center=True).max()

        # fill edges
        y_min = y_min.bfill().ffill()
        y_max = y_max.bfill().ffill()

        # =========================
        # Plot
        # =========================

        # shaded envelope
        #plt.fill_between(
        #    x,
        #    y_min,
        #    y_max,
        #    color=color,
        #    alpha=0.25,
        #    #label=col
        #)

        # raw markers on top
        for chem_type, subdf in df.groupby("chem"):

            is_sdl = "SDL" in str(chem_type)

            plt.plot(
                subdf[NaOH_mole],
                subdf[col],
                linestyle='None',
                marker='*' if is_sdl else chem_markers.get(chem_type, "o"),
                markersize=14 if is_sdl else 4,
                color=color,  # keep same color class
                markeredgecolor='black' if is_sdl else None,
                markeredgewidth=1.1 if is_sdl else 0,
                alpha=1.0 if is_sdl else 0.6,
                zorder=5 if is_sdl else 1,
                label=f"{col} | {chem_type}"
            )


# =========================
# Axes (outside loop)
# =========================
ax = plt.gca()
ax.set_yscale("log")
ax.set_xlabel("NaOH added (mole NaOH per L leachate)")
ax.set_ylabel("Total Dissolved Metal Concentration (mol/kgw)")

# --- secondary axis ---
def mol_to_mL(x):
    return naoh_moles_to_mL(x, C_stock)

def mL_to_mol(x):
    return x * 1e-3 * C_stock

secax = ax.secondary_xaxis("top", functions=(mol_to_mL, mL_to_mol))
secax.set_xlabel("NaOH added (mL NaOH per L leachate)")

# =========================
# Final formatting
# =========================
plt.legend(
    fontsize=7,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0
)
plt.grid(False)
plt.tight_layout()

plt.savefig(f"{fldr}/NaOH_vs_TotalDissolvedMetalComp_all_experiments_garbage.png", dpi=300)
plt.show()


#####
#####
#####

plt.figure(figsize=(7, 5))

for i, fl in enumerate(fls[1:]):
    fname = os.path.join(fl)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    # --- column detection ---
    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    ph_col = find_col(df, ["pH","ph"])

    # --- stock concentration ---
    C_stock = float(df[NaOH_stock].iloc[0])

    # --- convert to volume ---
    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    # --- clean + sort ---
    df = df.dropna(subset=[NaOH_mole, ph_col, "chem", "Al_sulfate_mol", "Gibbsite", "Cu(OH)2(s)"])
    df = df.sort_values(by=NaOH_mole)

    for col, color in zip(["Ni", "Mn", "Co"], ["red","blue","orange"]):
        if col not in df.columns:
            print(f"⚠️ Missing expected column '{col}' in {fname}")
            df[col] = 0.0  # add dummy column to avoid errors

        x = df[NaOH_mole].values
        y = df[col].values

        # =========================
        # Rolling min–max envelope
        # =========================
        window = 7   # adjust (5–15)

        y_series = pd.Series(y)

        y_min = y_series.rolling(window, center=True).min()
        y_max = y_series.rolling(window, center=True).max()

        # fill edges
        y_min = y_min.bfill().ffill()
        y_max = y_max.bfill().ffill()

        # =========================
        # Plot
        # =========================

        # shaded envelope
        #plt.fill_between(
        #    x,
        #    y_min,
        #    y_max,
        #    color=color,
        #    alpha=0.25,
        #    #label=col
        #)

        # raw markers on top
        for chem_type, subdf in df.groupby("chem"):

            is_sdl = "SDL" in str(chem_type)

            plt.plot(
                subdf[NaOH_mole],
                subdf[col],
                linestyle='None',
                marker='*' if is_sdl else chem_markers.get(chem_type, "o"),
                markersize=14 if is_sdl else 4,
                color=color,  # keep same color class
                markeredgecolor='black' if is_sdl else None,
                markeredgewidth=1.1 if is_sdl else 0,
                alpha=1.0 if is_sdl else 0.6,
                zorder=5 if is_sdl else 1,
                label=f"{col} | {chem_type}"
            )


# =========================
# Axes (outside loop)
# =========================
ax = plt.gca()
ax.set_yscale("log")
ax.set_xlabel("NaOH added (mole NaOH per L leachate)")
ax.set_ylabel("Total Dissolved Metal Concentration (mol/kgw)")

# --- secondary axis ---
def mol_to_mL(x):
    return naoh_moles_to_mL(x, C_stock)

def mL_to_mol(x):
    return x * 1e-3 * C_stock

secax = ax.secondary_xaxis("top", functions=(mol_to_mL, mL_to_mol))
secax.set_xlabel("NaOH added (mL NaOH per L leachate)")

# =========================
# Final formatting
# =========================

plt.legend(
    fontsize=7,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0
)
plt.grid(False)
plt.tight_layout()

plt.savefig(f"{fldr}/NaOH_vs_TotalDissolvedMetalComp_all_experiments_NMC.png", dpi=300)
plt.show()

######
######
######



plt.figure(figsize=(7, 5))

for i, fl in enumerate(fls[1:]):
    fname = os.path.join(fl)
    fname2= "speciation_results_merged/DOE_merged_mix2_react.csv"

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)
    df2 = pd.read_csv(fname2)

    for col, color in zip(["Ni", "Co", "Mn"], ["red","blue","orange"]):
        if col not in df.columns:
            print(f"⚠️ Missing expected column '{col}' in {fname}")
            df[col] = 0.0  # add dummy column to avoid errors

        # raw markers on top
        # raw markers on top
        for chem_type, subdf in df.groupby("chem"):

            is_sdl = "SDL" in str(chem_type)

            # align df2 to the same indices as subdf
            subdf2 = df2.loc[subdf.index]

            plt.plot(
                subdf[col],          # x from df subset
                subdf2[col],         # y from df2 subset
                linestyle='None',
                marker='*' if is_sdl else chem_markers.get(chem_type, "o"),
                markersize=14 if is_sdl else 6,
                color=color,
                markeredgecolor='black' if is_sdl else None,
                markeredgewidth=1.1 if is_sdl else 0,
                alpha=1.0 if is_sdl else 0.6,
                zorder=5 if is_sdl else 1,
                label=f"{col} | {chem_type}"
            )


# =========================
# Axes (outside loop)
# =========================
ax = plt.gca()
ax.set_xscale("log")
ax.set_yscale("log")
lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1])
]

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.plot(lims, lims, 'k--', linewidth=1)  # y = x line
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel("Total Dissolved Metal Concentration (mol/kgw) - after eq", fontsize=11)
ax.set_ylabel("Total Dissolved Metal Concentration (mol/kgw) - before eq", fontsize=11)

# =========================
# Final formatting
# =========================

plt.legend(
    fontsize=7,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0
)
plt.grid(False)
plt.tight_layout()

plt.savefig(f"{fldr}/NaOH_vs_TotalDissolvedMetalComp_all_experiments_NMC_b_eq.png", dpi=300)
plt.show()


#plt.figure(figsize=(7, 5))

for i, fl in enumerate(fls[1:]):
    fname1 = os.path.join(fl)
    fname2= "speciation_results_merged/DOE_merged_mix2_react.csv"

    if not os.path.exists(fname1):
        print(f"⚠️ Missing file: {fname1}")
        continue

    df = pd.read_csv(fname1)
    df2 = pd.read_csv(fname2)

    for col, color in zip(["Al", "Fe", "Cu", "Co", "Ni", "Mn"], ["red","blue","orange", "green", "purple", "black"]):
        if col not in df.columns:
            print(f"⚠️ Missing expected column '{col}' in {fname1}")
            df[col] = 0.0  # add dummy column to avoid errors

        plt.figure(figsize=(7, 5))

        # raw markers on top
        # raw markers on top
        for chem_type, subdf in df.groupby("chem"):

            is_sdl = "SDL" in str(chem_type)

            # align df2 to the same indices as subdf
            subdf2 = df2.loc[subdf.index]

            plt.plot(
                subdf[col],          # x from df subset
                subdf2[col],         # y from df2 subset
                linestyle='None',
                marker='*' if is_sdl else chem_markers.get(chem_type, "o"),
                markersize=14 if is_sdl else 6,
                color=color,
                markeredgecolor='black' if is_sdl else None,
                markeredgewidth=1.1 if is_sdl else 0,
                alpha=1.0 if is_sdl else 0.6,
                zorder=5 if is_sdl else 1,
                label=f"{col} | {chem_type}"
            )


        # =========================
        # Axes (outside loop)
        # =========================
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")

        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims, lims, 'k--', linewidth=1)  # y = x line
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel("Total Dissolved Metal Concentration (mol/kgw) - after eq", fontsize=11)
        ax.set_ylabel("Total Dissolved Metal Concentration (mol/kgw) - before eq", fontsize=11)

        # =========================
        # Final formatting
        # =========================

        plt.legend(
            fontsize=7,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0
        )
        plt.grid(False)
        plt.tight_layout()

        plt.savefig(f"{fldr}/TotalDissolvedMetalComp_all_experiments_garbage_b_eq_{col}.png", dpi=300)
        plt.close()   

#####
#####
#####

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.figure(figsize=(7, 5))

# -------------------------
# Config
# -------------------------
metals = ["Al", "Fe", "Cu", "Co", "Ni", "Mn"]
colors = ["red", "blue", "orange", "green", "purple", "black"]

metal_stoich = {
    "Al": 2,
    "Fe": 2,
    "Cu": 1,
    "Co": 1,
    "Ni": 1,
    "Mn": 1
}

metal_solids = {
    "Al": ["Gibbsite", "Al(OH)3(am)", "Al(OH)3_metastable"],
    "Fe": ["Fe(OH)3(s)", "Fe(OH)3(a)", "Fe(OH)3(cr)"],
    "Ni": ["Ni(OH)2(s)"],
    "Cu": ["Cu(OH)2(s)"],
    "Co": ["Co(OH)2(s)"],
    "Mn": ["Mn(OH)2(s)"]
}

# -------------------------
# Plot loop
# -------------------------
chem_all = set()

for fl in fls[1:]:

    if not os.path.exists(fl):
        print(f"⚠️ Missing file: {fl}")
        continue

    df = pd.read_csv(fl)

    chem_all.update(df["chem"].unique())

    for metal, color in zip(metals, colors):

        solids = metal_solids[metal]

        # ensure columns exist
        for col in solids:
            if col not in df.columns:
                df[col] = 0.0

        # compute totals
        df[f"{metal}_solid_total"] = df[solids].sum(axis=1)
        df[f"{metal}_initial"] = df[f"{metal}_sulfate_mol"] * metal_stoich[metal]

        df[f"{metal}_recovery_pct"] = (
            df[f"{metal}_solid_total"] / df[f"{metal}_initial"]
        ) * 100

        # plot by chemistry
        for chem_type, subdf in df.groupby("chem"):

            is_sdl = "SDL" in str(chem_type)

            plt.plot(
                subdf["NaOH_mole"],
                subdf[f"{metal}_recovery_pct"],
                linestyle='None',
                marker='*' if is_sdl else chem_markers.get(chem_type, "o"),
                markersize=14 if is_sdl else 6,
                color=color,
                markeredgecolor='black' if is_sdl else None,
                markeredgewidth=1.1 if is_sdl else 0,
                alpha=1.0 if is_sdl else 0.6,
                zorder=5 if is_sdl else 1,
                label=None   # ✅ avoid duplicate legend entries
            )
    
    plt.ylim([-5, 125])        
    plt.xlim([-0.05, 1.8])        

# -------------------------
# Axes
# -------------------------
ax = plt.gca()

ax.set_xlabel("NaOH Concentration (mol)", fontsize=18)
ax.set_ylabel("% Recovery", fontsize=18)
ax.tick_params(axis='both', labelsize=16)

# -------------------------
# Legends (clean & separated)
# -------------------------

# Metal legend (colors)
metal_handles = [
    Line2D(
        [0], [0],
        marker='o',
        linestyle='None',
        markersize=8,
        markerfacecolor=c,
        markeredgecolor='none',
        label=m
    )
    for m, c in zip(metals, colors)
]

# Chemistry legend (markers)
chem_handles = []
for chem_type in sorted(chem_all):

    is_sdl = "SDL" in str(chem_type)

    chem_handles.append(
        Line2D(
            [0], [0],
            marker='*' if is_sdl else chem_markers.get(chem_type, "o"),
            linestyle='None',
            markersize=10 if is_sdl else 6,
            markerfacecolor="gold" if is_sdl else "gray",
            markeredgecolor="black" if is_sdl else "none",
            markeredgewidth=1.1 if is_sdl else 0,
            label=chem_type
        )
    )

# Draw legends
leg1 = plt.legend(
    handles=metal_handles,
    title="Metal",
    loc="lower left",
    bbox_to_anchor=(0.87, 0.01, 1, 0.1),
    fontsize=10,
    handletextpad=0.05,
    title_fontsize=12
)

leg2 = plt.legend(
    handles=chem_handles,
    title="Battery Chemistry",
    loc="upper left",
    bbox_to_anchor=(0, 0.90, 1, 0.1),  # full axes width
    mode="expand",                     # forces it to fit
    ncol=len(chem_handles),
    fontsize=10,
    title_fontsize=12,
    handletextpad=0.1,
    columnspacing=0.8,
    frameon=True
)

ax.add_artist(leg1)

# -------------------------
# Final formatting
# -------------------------
plt.grid(False)
plt.tight_layout()

plt.savefig(f"{fldr}/pct_recovery.png", dpi=300)
plt.show()

#######
#######
#######

for i, fl in enumerate(fls[1:]):
    fname1= "speciation_results_merged/DOE_merged_mix1_i_soln.csv" #os.path.join(fl)
    fname2= "speciation_results_merged/DOE_merged_mix1_react.csv"

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df1 = pd.read_csv(fname1)
    df2 = pd.read_csv(fname2)

    for col1, col2, color in zip(["Al_sulfate_mol", "Fe_sulfate_mol", "Cu_sulfate_mol","Co_sulfate_mol", "Ni_sulfate_mol", "Mn_sulfate_mol"], ["Al", "Fe", "Cu", "Co", "Ni", "Mn"], ["red", "blue", "orange", "green", "purple", "black"]):
        if col1 not in df1.columns:
            print(f"⚠️ Missing expected column '{col1}' in {fname1}")
            df1[col1] = 0.0  # add dummy column to avoid errors

        if col2 not in df2.columns:
            print(f"⚠️ Missing expected column '{col2}' in {fname2}")
            df2[col2] = 0.0  # add dummy column to avoid errors

        plt.figure(figsize=(7, 5))

        # raw markers on top
        for chem_type, subdf2 in df2.groupby("chem"):

            is_sdl = "SDL" in str(chem_type)

            # align df2 to the same indices as subdf
            #subdf1 = df1.loc[subdf2.index]

            plt.plot(
                subdf2[col2],         # after eq
                subdf2[col1],         # feedstock (before eq)
                linestyle='None',
                marker='*' if is_sdl else chem_markers.get(chem_type, "o"),
                markersize=14 if is_sdl else 6,
                color=color,
                markeredgecolor='black' if is_sdl else None,
                markeredgewidth=1.1 if is_sdl else 0,
                alpha=1.0 if is_sdl else 0.6,
                zorder=5 if is_sdl else 1,
                label=f"{col2} | {chem_type}"
            )


        # =========================
        # Axes (outside loop)
        # =========================
        ax = plt.gca()
        #ax.set_xscale("log")
        #ax.set_yscale("log")

        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims, lims, 'k--', linewidth=1)  # y = x line
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel("Dissolved Metal Concentration at Equilibrium (mol/kgw)", fontsize=11)
        ax.set_ylabel("Feedstock concentration (mol)", fontsize=11)

            # =========================
            # Final formatting
            # =========================

        plt.legend(
            fontsize=7,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0
        )
        plt.grid(False)
        plt.tight_layout()

        plt.savefig(f"{fldr}/TotalDissolvedMetalComp_all_experiments_garbage_MIX1_b_eq_{col1}_{col2}.png", dpi=300)
        plt.show()      


#######
#######
#######

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------------
# Inputs
# -------------------------
fname1 = "speciation_results_merged/DOE_merged_mix1_i_soln.csv"
fname2 = "speciation_results_merged/DOE_merged_mix2_eq_react.csv"

if not os.path.exists(fname1) or not os.path.exists(fname2):
    raise FileNotFoundError(f"Missing file: {fname1} or {fname2}")

df1 = pd.read_csv(fname1)
df2 = pd.read_csv(fname2)

# -------------------------
# Metal mapping
# -------------------------
metal_map = {
    "Al_sulfate_mol": "Al",
    "Fe_sulfate_mol": "Fe",
    "Cu_sulfate_mol": "Cu",
    "Co_sulfate_mol": "Co",
    "Ni_sulfate_mol": "Ni",
    "Mn_sulfate_mol": "Mn"
}

colors = ["red", "blue", "orange", "green", "purple", "black"]

# -------------------------
# Loop over metals
# -------------------------
for (col1, col2), color in zip(metal_map.items(), colors):

    # ensure columns exist
    if col1 not in df1.columns:
        print(f"⚠️ Missing {col1} in df1 → filling zeros")
        df1[col1] = 0.0

    if col2 not in df2.columns:
        print(f"⚠️ Missing {col2} in df2 → filling zeros")
        df2[col2] = 0.0

    # dynamic legend trackers
    metal_colors_used = {}
    chem_markers_used = {}

    plt.figure(figsize=(7, 5))

    # -------------------------
    # Plot
    # -------------------------
    for chem_type, subdf2 in df2.groupby("chem"):

        # align df1 with df2 indices  ✅ CRITICAL FIX
        subdf1 = df1.loc[subdf2.index]

        is_sdl = "SDL" in str(chem_type)
        marker = '*' if is_sdl else chem_markers.get(chem_type, "o")

        # track legend entries dynamically
        metal_colors_used[col2] = color
        chem_markers_used[chem_type] = marker

        plt.plot(
            subdf2[col2],        # equilibrium
            subdf1[col1],        # feedstock
            linestyle='None',
            marker=marker,
            markersize=14 if is_sdl else 6,
            color=color,
            markeredgecolor='black' if is_sdl else None,
            markeredgewidth=1.1 if is_sdl else 0,
            alpha=1.0 if is_sdl else 0.6,
            zorder=5 if is_sdl else 1,
            label=None
        )

    # -------------------------
    # Axes
    # -------------------------
    ax = plt.gca()

    lims = [
        np.nanmin([*ax.get_xlim(), *ax.get_ylim()]),
        np.nanmax([*ax.get_xlim(), *ax.get_ylim()])
    ]

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, 'k--', linewidth=1)

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("Dissolved Metal Concentration at Equilibrium (mol/kgw)", fontsize=11)
    ax.set_ylabel("Feedstock concentration (mol)", fontsize=11)

    # -------------------------
    # Legends (dynamic)
    # -------------------------
    metal_handles = [
        Line2D([0], [0], color=c, lw=2, label=m)
        for m, c in metal_colors_used.items()
    ]

    chem_handles = [
        Line2D([0], [0],
               marker=mk,
               color='gray',
               linestyle='None',
               markersize=7,
               label=chem)
        for chem, mk in chem_markers_used.items()
    ]

    leg1 = plt.legend(
        handles=metal_handles,
        title="Metal",
        loc="upper right",
        fontsize=8,
        #bbox_to_anchor=(1.02, 1)
    )

    leg2 = plt.legend(
        handles=chem_handles,
        title="Chemistry",
        loc="lower right",
        fontsize=8,
        #bbox_to_anchor=(1.02, 0)
    )

    plt.gca().add_artist(leg1)

    # -------------------------
    # Final formatting
    # -------------------------
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(
        f"{fldr}/TotalDissolvedMetalComp_MIX2_eq_{col2}.png",
        dpi=300
    )

    plt.show()



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------------
# Inputs
# -------------------------
fname1 = "speciation_results_merged/DOE_merged_mix1_i_soln.csv"
fname2 = "speciation_results_merged/DOE_merged_mix2_eq_react.csv"

if not os.path.exists(fname1) or not os.path.exists(fname2):
    raise FileNotFoundError(f"Missing file: {fname1} or {fname2}")

df1 = pd.read_csv(fname1)
df2 = pd.read_csv(fname2)

# -------------------------
# Stoichiometry (CRITICAL FIX)
# -------------------------
metal_map = {
    "Al_sulfate_mol": ("Al", 2),  # Al2 → 2 Al
    "Fe_sulfate_mol": ("Fe", 2),  # Fe2 → 2 Fe
    "Cu_sulfate_mol": ("Cu", 1),
    "Co_sulfate_mol": ("Co", 1),
    "Ni_sulfate_mol": ("Ni", 1),
    "Mn_sulfate_mol": ("Mn", 1)
}

colors = ["red", "blue", "orange", "green", "purple", "black"]

# -------------------------
# Loop over metals
# -------------------------
for (col1, (col2, stoich)), color in zip(metal_map.items(), colors):

    # ensure columns exist
    if col1 not in df1.columns:
        print(f"⚠️ Missing {col1} in df1 → filling zeros")
        df1[col1] = 0.0

    if col2 not in df2.columns:
        print(f"⚠️ Missing {col2} in df2 → filling zeros")
        df2[col2] = 0.0

    # -------------------------
    # Compute TRUE initial metal (mol)
    # -------------------------
    df1[f"{col2}_initial_mol"] = df1[col1] * stoich

    metal_colors_used = {}
    chem_markers_used = {}

    plt.figure(figsize=(7, 5))

    # -------------------------
    # Plot
    # -------------------------
    for chem_type, subdf2 in df2.groupby("chem"):

        # align rows
        subdf1 = df1.loc[subdf2.index]

        is_sdl = "SDL" in str(chem_type)
        marker = '*' if is_sdl else chem_markers.get(chem_type, "o")

        metal_colors_used[col2] = color
        chem_markers_used[chem_type] = marker

        # -------------------------
        # Plot (mol vs mol)
        # -------------------------
        x = subdf2[col2]                    # mol/kgw ≈ mol (since water ~1 kg)
        y = subdf1[f"{col2}_initial_mol"]   # TRUE mol

        plt.plot(
            x,
            y,
            linestyle='None',
            marker=marker,
            markersize=14 if is_sdl else 6,
            color=color,
            markeredgecolor='black' if is_sdl else None,
            markeredgewidth=1.1 if is_sdl else 0,
            alpha=1.0 if is_sdl else 0.6,
            zorder=5 if is_sdl else 1,
            label=None
        )

    # -------------------------
    # Axes
    # -------------------------
    ax = plt.gca()

    lims = [
        np.nanmin([*ax.get_xlim(), *ax.get_ylim()]),
        np.nanmax([*ax.get_xlim(), *ax.get_ylim()])
    ]

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, 'k--', linewidth=1)

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("Dissolved Metal at Equilibrium (mol)", fontsize=11)
    ax.set_ylabel("Initial Metal (mol)", fontsize=11)

    # -------------------------
    # Legends (dynamic)
    # -------------------------
    metal_handles = [
        Line2D([0], [0], color=c, lw=2, label=m)
        for m, c in metal_colors_used.items()
    ]

    chem_handles = [
        Line2D([0], [0],
               marker=mk,
               color='gray',
               linestyle='None',
               markersize=7,
               label=chem)
        for chem, mk in chem_markers_used.items()
    ]

    combined_handles = metal_handles + chem_handles

    plt.legend(
        handles=combined_handles,
        title="Metal / Chemistry",
        loc="lower right",
        fontsize=9,
        title_fontsize=10    # title ← this is what you want
    )

    # -------------------------
    # Final formatting
    # -------------------------
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(
        f"{fldr}/TotalDissolvedMetalComp_MIX2_eq_{col2}_updated.png",
        dpi=300
    )

    plt.show()

    # -------------------------
    # 🔍 Sanity check
    # -------------------------
    violations = (x > y).sum()
    print(f"{col2}: {violations} mass-balance violations")


pause
# ==============================================================
# ==============================================================

plt.figure(figsize=(7, 5))
for i, fl in enumerate(fls):
    out_dir = os.path.join(fl)
    fname = os.path.join(out_dir, "DOE_merged_mix2_eq_react.csv")
    #print(fname)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    BattLeachate_volume = find_col(df, ["BattLeachate_volume"])
    ph_col   = find_col(df, ["pH","ph"])
    pe_col   = find_col(df, ["pe"])
    t_col    = find_col(df, ["T_C","temp","temperature"], contains="temp")
    mol_cols = discover_phases(df, "_mol")
    PHASES = discover_phases(df)
    METALS = discover_metal_free_cols(df)
    si_cols = df.filter(like="si_").columns
    m_cols = df.filter(like="m_").columns
    d_cols = df.filter(like="d_").columns

    C_stock = float(df[NaOH_stock].iloc[0])

    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    # Drop failed / missing runs
    df = df.dropna(subset=["NaOH_mole", "mu"])

    plt.plot(
        df["NaOH_mole"],
        df["mu"],
        marker=marker_styles[i],
        #linestyle=line_styles[i],
        c=line_colors[i],
        markersize=3,
        label=lbs[i],
    )

# Convert tick labels from mol → mL
C_stock = float(df[NaOH_stock].iloc[0])  # already defined in the loop
#print(C_stock)
#pause

# Top axis: mL
def mol_to_mL(x):
    return naoh_moles_to_mL(x, C_stock)

def mL_to_mol(x):
    return x * 1e-3 * C_stock

# Bottom axis
ax = plt.gca()
ax.set_xscale("log")
ax.set_xlabel("NaOH added (mole NaOH per L leachate)")
# Top axis
secax = ax.secondary_xaxis("top", functions=(mol_to_mL, mL_to_mol))
secax.set_xscale("log")
secax.set_xlabel("NaOH added (mL NaOH per L leachate)")

mol_ticks = ax.get_xticks()
ml_ticks = naoh_moles_to_mL(mol_ticks, C_stock)

#plt.yscale("log")
#plt.xscale("log")

#plt.xlabel("NaOH volume added (mole)");
plt.ylabel("Ionic Strength (mol.Kg$^{-1}$)")
plt.legend(loc="upper left")
plt.grid(False)
plt.tight_layout()
plt.savefig(f"{fldr}/NaOH_vs_mu_all_experiments.png", dpi=300)
plt.show()

# ==============================================================
# ==============================================================
# ==============================================================

plt.figure(figsize=(7, 5))
for i, fl in enumerate(fls):
    out_dir = os.path.join(fl)
    fname = os.path.join(out_dir, "DOE_merged_mix2_eq_react.csv")
    #print(fname)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    BattLeachate_volume = find_col(df, ["BattLeachate_volume"])
    ph_col   = find_col(df, ["pH","ph"])
    pe_col   = find_col(df, ["pe"])
    t_col    = find_col(df, ["T_C","temp","temperature"], contains="temp")
    mol_cols = discover_phases(df, "_mol")
    PHASES = discover_phases(df)
    METALS = discover_metal_free_cols(df)
    si_cols = df.filter(like="si_").columns
    m_cols = df.filter(like="m_").columns
    d_cols = df.filter(like="d_").columns

    #print(d_cols)

    C_stock = float(df[NaOH_stock].iloc[0])

    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    # Drop failed / missing runs
    df = df.dropna(subset=["NaOH_mole", "mu"])

    plt.plot(
        df["NaOH_mole"],
        df["Alk"],
        marker=marker_styles[i],
        linestyle=line_styles[i],
        c=line_colors[i],
        markersize=3,
        label=lbs[i],
    )

plt.yscale("log")
plt.xscale("log")
plt.xlabel("NaOH volume added (mole)");
plt.ylabel("Alkalinity (mol.Kg$^{-1}$)")
plt.legend(loc="upper left")
plt.grid(False)
plt.tight_layout()
plt.savefig(f"{fldr}/NaOH_vs_Alk_all_experiments.png", dpi=300)
plt.show()


# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================

plt.figure(figsize=(7, 5))
for i, fl in enumerate(fls):
    out_dir = os.path.join(fl)
    fname = os.path.join(out_dir, "DOE_merged_mix2_eq_react.csv")
    #print(fname)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    BattLeachate_volume = find_col(df, ["BattLeachate_volume"])
    ph_col   = find_col(df, ["pH","ph"])
    pe_col   = find_col(df, ["pe"])
    t_col    = find_col(df, ["T_C","temp","temperature"], contains="temp")
    mol_cols = discover_phases(df, "_mol")
    PHASES = discover_phases(df)
    METALS = discover_metal_free_cols(df)
    si_cols = df.filter(like="si_").columns
    m_cols = df.filter(like="m_").columns
    d_cols = df.filter(like="d_").columns

    #print(d_cols)

    C_stock = float(df[NaOH_stock].iloc[0])

    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    # Drop failed / missing runs
    df = df.dropna(subset=["NaOH_mole", "mu"])

    plt.plot(
        df["NaOH_mole"],
        df["pe"],
        marker=marker_styles[i],
        linestyle=line_styles[i],
        c=line_colors[i],
        markersize=3,
        label=lbs[i],
    )

plt.yscale("log")
plt.xscale("log")
plt.xlabel("NaOH volume added (mole)");
plt.ylabel("pe")
plt.legend(loc="lower left")
plt.grid(False)
plt.tight_layout()
plt.savefig(f"{fldr}/NaOH_vs_pe_all_experiments.png", dpi=300)
plt.show()

# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================

plt.figure(figsize=(7, 5))
for i, fl in enumerate(fls):
    out_dir = os.path.join(fl)
    fname = os.path.join(out_dir, "DOE_merged_mix2_eq_react.csv")
    #print(fname)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    BattLeachate_volume = find_col(df, ["BattLeachate_volume"])
    ph_col   = find_col(df, ["pH","ph"])
    pe_col   = find_col(df, ["pe"])
    t_col    = find_col(df, ["T_C","temp","temperature"], contains="temp")
    mol_cols = discover_phases(df, "_mol")
    PHASES = discover_phases(df)
    METALS = discover_metal_free_cols(df)
    si_cols = df.filter(like="si_").columns
    m_cols = df.filter(like="m_").columns
    d_cols = df.filter(like="d_").columns

    #print(d_cols)

    C_stock = float(df[NaOH_stock].iloc[0])

    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    # Drop failed / missing runs
    df = df.dropna(subset=["NaOH_mole", "mu"])

    plt.plot(
        df["NaOH_mole"],
        df["pct_err"],
        marker=marker_styles[i],
        linestyle=line_styles[i],
        c=line_colors[i],
        markersize=3,
        label=lbs[i],
    )

plt.yscale("log")
plt.xscale("log")
plt.xlabel("NaOH volume added (mole)");
plt.ylabel("pct_err")
plt.legend(loc="upper left")
plt.grid(False)
plt.tight_layout()
plt.savefig(f"{fldr}/NaOH_vs_pct_err_all_experiments.png", dpi=300)
plt.show()

# ==============================================================
# Key dissolved metal concentrations (main outcomes) - metal removal, selectivity, or treatment performance
# ==============================================================

plt.figure(figsize=(7, 5))
for i, fl in enumerate(fls):
    out_dir = os.path.join(fl)
    fname = os.path.join(out_dir, "DOE_merged_mix2_eq_react.csv")
    #print(fname)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    BattLeachate_volume = find_col(df, ["BattLeachate_volume"])
    ph_col   = find_col(df, ["pH","ph"])
    pe_col   = find_col(df, ["pe"])
    t_col    = find_col(df, ["T_C","temp","temperature"], contains="temp")
    mol_cols = discover_phases(df, "_mol")
    PHASES = discover_phases(df)
    METALS = discover_metal_free_cols(df)
    si_cols = df.filter(like="si_").columns
    m_cols = df.filter(like="m_").columns
    d_cols = df.filter(like="d_").columns

    #m_cols_main = ['m_Ni2+', 'm_Co2+', 'm_Mn2+', 'm_Mn3+', 'm_Fe2+', 'm_Fe3+']
    #print(m_cols)
    m_cols_main = ['Na']

    C_stock = float(df[NaOH_stock].iloc[0])

    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    # Drop failed / missing runs
    df = df.dropna(subset=["NaOH_mole", "mu"])

    for j, metal in enumerate(m_cols_main):
        plt.plot(
            df["NaOH_mole"],
            df[metal],
            marker=marker_styles[j],
            linestyle=line_styles[i],
            c=line_colors[i],
            markersize=3,
            label=f"{lbs[i]}: {metal}",
        )

    plt.yscale("log")
    plt.xscale("log")
plt.ylim(10e-11,12e1)
plt.xlabel("NaOH volume added (mole)");
plt.ylabel("Total Na dissolved concentration (mol.Kg$^{-1}$)")
plt.legend(loc="lower left")
plt.grid(False)
#plt.title(f"Experiment: {lbs[i]}")
plt.tight_layout()
plt.savefig(f"{fldr}/NaOH_vs_Total_Na_Conc_experiment.png", dpi=300)
plt.show()



for i, fl in enumerate(fls):
    print(fl)
    plt.figure(figsize=(7, 5))
    out_dir = os.path.join(fl)
    fname = os.path.join(out_dir, "DOE_merged_mix2_eq_react.csv")
    #print(fname)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    BattLeachate_volume = find_col(df, ["BattLeachate_volume"])
    ph_col   = find_col(df, ["pH","ph"])
    pe_col   = find_col(df, ["pe"])
    t_col    = find_col(df, ["T_C","temp","temperature"], contains="temp")
    mol_cols = discover_phases(df, "_mol")
    PHASES = discover_phases(df)
    METALS = discover_metal_free_cols(df)
    si_cols = df.filter(like="si_").columns
    m_cols = df.filter(like="m_").columns
    d_cols = df.filter(like="d_").columns

    #m_cols_main = ['m_Ni2+', 'm_Co2+', 'm_Mn2+', 'm_Mn3+', 'm_Fe2+', 'm_Fe3+']
    #print(m_cols)
    m_cols_main = [
        'Al',
        'Co',
        'Fe',
        'Mn',
        'Cu','Ni','Mn'
        ,'Na'
    ]

    C_stock = float(df[NaOH_stock].iloc[0])

    df["NaOH_volume (mL)"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    df["NaOH_Equilavent"] = df["NaOH_volume (mL)"]/1000

    # Drop failed / missing runs
    df = df.dropna(subset=["NaOH_mole", "mu"])

    for j, metal in enumerate(m_cols_main):
        plt.plot(
            df["NaOH_Equilavent"],
            df[metal],
            marker=marker_styles[j],
            linestyle=line_styles[i],
            c=line_colors[j],
            markersize=3,
            label=f"{metal}",
        )

    #total_mg = mg_per_L * 0.001  # L
    sheet_names = ["CC-NRCan3-046 HTE 40%"]
    for name in sheet_names:
        if name not in ExpData_dict:
            print(f"⚠️ Sheet '{name}' not found.")
            continue

        exp_df = ExpData_dict[name].dropna(axis=1, how="all").copy()
        exp_df.columns = exp_df.columns.astype(str).str.strip()

        if "NaOH Equivalents" not in exp_df or "Vol of NaOH (mL)" not in exp_df:
            print(f"❌ Required columns missing in '{name}'.")
            continue

        y_cols = [
            #'Al 396.152 nm ppm',
            #'Co 238.892 nm ppm', 
            ##'Cu 327.395 nm ppm', 
            #'Fe 238.204 nm ppm',
            #'Mn 257.610 nm ppm', 
            ##'Ni 231.604 nm ppm',
            ##'Li 610.365 nm ppm', 
        ]

        NaOH_C_stock = 2.5  # M
        #exp_mgL = exp_df[y_cols]
        exp_mgL = exp_df[y_cols].clip(lower=0)
        print(exp_mgL)
        #pause
        
        total_mol = total_metal_moles_from_icp(exp_mgL, volume_mL=1)
        print(total_mol.values())
        print('--- Vol of NaOH (mL) converted ---')
        print(exp_df["Vol of NaOH (mL)"]/1000*NaOH_C_stock)
        
        for c in y_cols:
            plt.plot(exp_df["Vol of NaOH (mL)"]/1000*NaOH_C_stock, total_mol[c], marker="s", linestyle="None", label=c)

    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(10e-11,12e1)
    plt.xlabel("NaOH volume added (mole)");
    plt.ylabel("Total metal dissolved concentration (mol.Kg$^{-1}$)")
    plt.legend(loc="upper left")
    plt.grid(False)
    plt.title(f"Experiment: {lbs[i]}")
    plt.tight_layout()
    plt.savefig(f"{fldr}/NaOH_vs_TotalMetalConc_experiment_{fl}.png", dpi=300)
    plt.show()

    
# ==============================================================
# Free metal ion molality (all aqueous) - speciation, thermodynamics, and precipitation mechanisms
# ==============================================================

for i, fl in enumerate(fls):
    plt.figure(figsize=(7, 5))
    #fig, ax = plt.subplots(figsize=(7, 5))

    out_dir = os.path.join(fl)
    fname = os.path.join(out_dir, "DOE_merged_mix2_eq_react.csv")
    #print(fname)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    BattLeachate_volume = find_col(df, ["BattLeachate_volume"])
    ph_col   = find_col(df, ["pH","ph"])
    pe_col   = find_col(df, ["pe"])
    t_col    = find_col(df, ["T_C","temp","temperature"], contains="temp")
    mol_cols = discover_phases(df, "_mol")
    PHASES = discover_phases(df)
    METALS = discover_metal_free_cols(df)
    si_cols = df.filter(like="si_").columns
    m_cols = df.filter(like="m_").columns
    d_cols = df.filter(like="d_").columns

    m_cols_main = ['m_Ni2+', 'm_Co2+', 'm_Mn2+', 'm_Mn3+', 'm_Fe2+', 'm_Fe3+']

    C_stock = float(df[NaOH_stock].iloc[0])

    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    # Drop failed / missing runs
    df = df.dropna(subset=["NaOH_mole", "mu"])

    for j, metal in enumerate(m_cols_main):
        plt.plot(
            df["NaOH_mole"],
            df[metal],
            marker=marker_styles[j],
            linestyle=line_styles[i],
            markersize=3,
            label=f"{metal}",
        )

    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(10e-41,10e0)
    plt.xlabel("NaOH volume added (mole)");
    plt.ylabel("Free Metal Ion Molality (mol.Kg$^{-1}$)")
    plt.legend(loc="lower left")
    plt.title(f"Experiment: {lbs[i]}")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{fldr}/NaOH_vs_FreeMetalIonConc_experiment_{fl}.png", dpi=300)
    plt.show()


    
# ==============================================================
# Free metal ion molality (all aqueous) - speciation, thermodynamics, and precipitation mechanisms
# ==============================================================

for i, fl in enumerate(fls):
    plt.figure(figsize=(7, 5))
    out_dir = os.path.join(fl)
    fname = os.path.join(out_dir, "DOE_merged_mix2_eq_react.csv")
    #print(fname)

    if not os.path.exists(fname):
        print(f"⚠️ Missing file: {fname}")
        continue

    df = pd.read_csv(fname)

    NaOH_mole = find_col(df, ["NaOH_mole","NaOH_moles","NaOH_mol"])
    NaOH_stock = find_col(df, ["NaOH_stock"])
    BattLeachate_volume = find_col(df, ["BattLeachate_volume"])
    ph_col   = find_col(df, ["pH","ph"])
    pe_col   = find_col(df, ["pe"])
    t_col    = find_col(df, ["T_C","temp","temperature"], contains="temp")
    mol_cols = discover_phases(df, "_mol")
    PHASES = discover_phases(df)
    METALS = discover_metal_free_cols(df)
    si_cols = df.filter(like="si_").columns
    m_cols = df.filter(like="m_").columns
    d_cols = df.filter(like="d_").columns

    #m_cols_main = ['m_Ni2+', 'm_Co2+', 'm_Mn2+', 'm_Mn3+', 'm_Fe2+', 'm_Fe3+']

    C_stock = float(df[NaOH_stock].iloc[0])

    df["NaOH_volume"] = naoh_moles_to_volume(
        df[NaOH_mole].astype(float),
        C_stock,
        unit="mL"
    )

    # Drop failed / missing runs
    df = df.dropna(subset=["NaOH_mole", "mu"])

    for j, metal in enumerate(d_cols):
        plt.plot(
            df["NaOH_mole"],
            df[metal],
            marker=marker_styles[j],
            linestyle="None",
            markersize=3,
            label=f"{metal}",
        )

    plt.yscale("log")
    plt.xscale("log")
    #plt.ylim(10e-41,10e0)
    plt.xlabel("NaOH volume added (mole)");
    plt.ylabel("Phase (mol)")
    plt.legend(loc="lower left")
    plt.title(f"Experiment: {lbs[i]}")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{fldr}/NaOH_vs_PhaseConc_experiment_{fl}.png", dpi=300)
    plt.show()