import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import re
import os
pd.set_option('future.no_silent_downcasting', True)


plt.rcParams['figure.dpi'] = 150          # for inline plots
plt.rcParams['savefig.dpi'] = 300         # for saved figures
#plt.rcParams['font.family'] = 'Arial'          # or 'Times New Roman', 'Calibri', etc.

MW = {
    "Al": 26.98, # g/mol
    "Co": 58.93,
    "Cu": 63.55,
    "Fe": 55.85,
    "Li": 6.94,
    "Mn": 54.94,
    "Ni": 58.69,
}

## mol/L 
def mgL_to_moles(mgL, MW):
    mg = mgL                        # mg per liter
    g = mg / 1000                   # convert mg → g
    mol = g / MW                    # moles = g/MW
    return mol


# %%

def read_multisheet_excel(file_path):
    """
    Reads all sheets from an Excel file, cleans, and converts to numeric.
    Returns a dictionary of cleaned DataFrames.
    """
    all_data = {}

    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        print(f"📘 Found sheets: {sheet_names}\n")

        for sheet in sheet_names:
            try:
                # Read sheet (no assumption about header position)
                df = pd.read_excel(file_path, sheet_name=sheet, header=None)

                # Identify start of the actual data table
                start_row = df[df.apply(lambda x: x.astype(str)
                                        .str.contains("NaOH Equivalents", case=False)
                                        .any(), axis=1)].index
                if len(start_row) > 0:
                    start_row = start_row[0]
                    df.columns = df.iloc[start_row]
                    df = df.drop(range(start_row + 1))
                    df = df.reset_index(drop=True)

                # Clean column names
                df.columns = [str(c).strip() for c in df.columns]
                df = df.dropna(axis=1, how='all')

                # Convert all numeric-looking columns to numbers
                for col in df.columns:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace(" ", "", regex=False)
                        .replace(["", "NA", "nan"], np.nan)
                    )
                    df[col] = pd.to_numeric(df[col])

                # Second pass: coerce any remaining objects to numeric
                for col in df.columns:
                    if df[col].dtype == "object":
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                all_data[sheet] = df
                print(f"✅ Loaded '{sheet}' → {df.shape[0]} rows, {df.shape[1]} cols")

            except Exception as e:
                print(f"❌ Error reading sheet '{sheet}': {e}")

    except FileNotFoundError:
        print(f"🚫 File not found: {file_path}")
    except Exception as e:
        print(f"⚠️ Unexpected error while opening file: {e}")

    return all_data

# %%


def read_multisheet_with_metadata(file_path, save_dir="parsed_data"):
    """
    Reads all sheets from an Excel file, cleans numeric data,
    extracts metadata (experimental overview), and saves both.
    """

    # Create output directory if missing
    os.makedirs(save_dir, exist_ok=True)

    meta_df = None
    all_data = {}
    metadata_records = []

    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        print(f"📘 Found sheets: {sheet_names}\n")

        for sheet in sheet_names:
            try:
                # Read full sheet (no header)
                df_raw = pd.read_excel(file_path, sheet_name=sheet, header=None)
                df_raw = df_raw.fillna("")

                # ---- 1️⃣ Extract Metadata Block (above the data table)
                # Find where the data table begins (by keyword)
                #start_idx = df_raw[df_raw.apply(
                #    lambda x: x.astype(str).str.contains("NaOH Equivalents", case=False).any(), axis=1
                #)].index
                
                mask = df_raw.apply(lambda x: x.astype(str).apply(lambda v: "NaOH Equivalents" in v if isinstance(v, str) else False)).any(axis=1)
                start_idx = mask[mask].index

                data_start = start_idx[0] if len(start_idx) > 0 else None

                # Extract metadata section (rows before table)
                if data_start:
                    meta_df = df_raw.iloc[:data_start].dropna(how='all').copy()
                    meta_df = meta_df.loc[:, meta_df.apply(lambda x: x.astype(str).str.strip().ne('').any())]

                    # Flatten metadata into key:value pairs
                    metadata_dict = {}
                    for _, row in meta_df.iterrows():
                        key = str(row[0]).strip().replace("\n", " ")
                        if len(row) > 1:
                            value = str(row[1]).strip().replace("\n", " ")
                            metadata_dict[key] = value

                    metadata_dict["Sheet Name"] = sheet
                    metadata_records.append(metadata_dict)

                # ---- 2️⃣ Extract Experimental Data Table
                if data_start is not None:
                    df_data = df_raw.iloc[data_start:].copy()
                    df_data.columns = df_data.iloc[0]
                    df_data = df_data.drop(df_data.index[0]).reset_index(drop=True)
                    df_data.columns = [str(c).strip() for c in df_data.columns]
                    df_data = df_data.replace(["", "NA", "nan"], np.nan)

                    # Convert numeric-looking columns
                    for col in df_data.columns:
                        df_data[col] = (
                            df_data[col].astype(str)
                            .str.replace(",", "", regex=False)
                            .str.replace(" ", "", regex=False)
                        )
                        df_data[col] = pd.to_numeric(df_data[col], errors="ignore")

                    all_data[sheet] = df_data

                    # ---- Save each data table
                    df_data.to_csv(f"{save_dir}/{sheet}_data.csv", index=False)
                    print(f"✅ Saved data table for '{sheet}'")

            except Exception as e:
                print(f"❌ Error reading sheet '{sheet}': {e}")

        # ---- 3️⃣ Save all metadata
        if metadata_records:
            meta_df = pd.DataFrame(metadata_records)
            meta_df.to_csv(f"{save_dir}/_metadata_summary.csv", index=False)
            print(f"\n🧾 Saved metadata summary: {save_dir}/_metadata_summary.csv")

    except FileNotFoundError:
        print(f"🚫 File not found: {file_path}")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")

    return all_data, meta_df

# %%

import pandas as pd
import numpy as np

def clean_metadata_table(meta_df, save_path="cleaned_metadata.csv"):
    """
    Cleans and standardizes metadata summary table.
    Handles missing columns gracefully and avoids deprecated behavior.
    """

    df = meta_df.copy()

    # --- Normalize and clean column names ---
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ")
        .str.replace("  ", " ", regex=False)
    )

    # --- Rename columns (only those that exist) ---
    rename_map = {
        "Volume of  battery solution in vial": "V_battery_vial_%",
        "Total vial volume": "V_vial_mL",
        "NaOH Stock solution  concentration": "NaOH_M",
        "Stirring duration": "t_stir_h",
        "Temperature": "T_condition",
        "Total reactor volume": "V_reactor_mL",
        "Volume of  battery solution in reactor": "V_battery_reactor_mL",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # --- Ensure expected columns exist (add NaNs if missing) ---
    expected_cols = [
        "V_vial_mL", "V_battery_vial_%", "NaOH_M", "t_stir_h",
        "V_reactor_mL", "V_battery_reactor_mL", "Experimental method",
        "Sheet Name", "T_condition"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # --- Convert numeric columns safely ---
    numeric_cols = ["V_vial_mL", "V_battery_vial_%", "NaOH_M",
                    "t_stir_h", "V_reactor_mL", "V_battery_reactor_mL"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Fill missing reactor values ---
    df["V_reactor_mL"] = df["V_reactor_mL"].fillna(df["V_vial_mL"])
    df["V_battery_reactor_mL"] = df["V_battery_reactor_mL"].fillna(
        (df["V_battery_vial_%"] / 100) * df["V_reactor_mL"]
    )

    # --- Clean categorical/text fields ---
    for col in ["T_condition", "Experimental method", "Sheet Name"]:
        df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

    # --- Compute derived feature: mmol NaOH per reactor ---
    df["NaOH_mmol_total"] = df["NaOH_M"] * df["V_reactor_mL"]

    # --- Save cleaned file ---
    df.to_csv(save_path, index=False)
    print(f"✅ Cleaned metadata saved to: {save_path}")

    return df


# %%

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_sheets(
    data_dict,
    sheet_names,
    x_col="NaOH Equivalents",
    y_columns=None,
    convert_units=False,
    mw_dict=None,
    unit_converter=None,
):
    """
    Plot selected sheets with optional unit conversion and aligned secondary x-axis
    based on measured NaOH volume.
    """

    for name in sheet_names:
        if name not in data_dict:
            print(f"⚠️ Sheet '{name}' not found.")
            continue

        df = data_dict[name].dropna(axis=1, how="all").copy()
        df.columns = df.columns.astype(str).str.strip()

        if x_col not in df or "Vol of NaOH (mL)" not in df:
            print(f"❌ Required columns missing in '{name}'.")
            continue

        y_cols = (
            [c for c in df.columns if c != x_col and pd.api.types.is_numeric_dtype(df[c])]
            if y_columns is None else y_columns
        )

        if not y_cols:
            print(f"⚠️ No numeric columns to plot in '{name}'.")
            continue

        # Optional unit conversion
        if convert_units:
            if unit_converter is None or mw_dict is None:
                raise ValueError("unit_converter and mw_dict required")

            for c in y_cols:
                elem = c.split()[0]
                if elem in mw_dict:
                    df[c] = unit_converter(df[c], mw_dict[elem])

        # Ensure monotonic x for interpolation
        df = df.sort_values(x_col)

        fig, ax = plt.subplots(figsize=(8, 5))

        for c in y_cols:
            ax.plot(df[x_col], df[c], marker="o", label=c)

        ax.set_xlabel(x_col)
        ax.set_ylabel("Concentration (mol/L)" if convert_units else "Concentration (mg/L)")

        # Force identical limits
        xmin, xmax = df[x_col].iloc[[0, -1]]
        ax.set_xlim(xmin, xmax)

        # Top axis (measured data)
        ax_top = ax.twiny()
        ax_top.set_xlim(xmin, xmax)
        ax_top.set_xlabel("NaOH added (mL in 1 mL Vial)")

        ticks = ax.get_xticks()
        ticks = ticks[(ticks >= xmin) & (ticks <= xmax)]

        vol = np.interp(
            ticks,
            df[x_col].values,
            df["Vol of NaOH (mL)"].values,
        )

        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels([f"{v:.2f}" for v in vol])

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.title(f"Experimental Results — {name}")
        plt.tight_layout()
        plt.savefig(f"process_plot_{name.replace(' ', '_')}.png", dpi=300)
        plt.show()



def build_process_map(data_dict, sheet_pattern="46", target_element="Ni 231.604 nm ppm"):
    """
    Build process maps (heatmaps) showing metal concentration vs NaOH equivalents and condition.
    Assumes sheet names encode process variable (e.g., '5%', '10%', ...).
    """
    records = []

    # Loop through selected sheets
    for name, df in data_dict.items():
        if sheet_pattern not in name:
            continue
        
        # Extract numeric condition value (e.g., '10%' -> 10)
        import re
        cond_match = re.search(r'(\d+)%', name)
        if not cond_match:
            continue
        condition = float(cond_match.group(1))

        # Ensure numeric conversion
        df = df.apply(pd.to_numeric, errors="coerce")

        if "NaOH Equivalents" not in df.columns or target_element not in df.columns:
            continue

        for _, row in df.iterrows():
            records.append({
                "Condition (%)": condition,
                "NaOH Equivalents": row["NaOH Equivalents"],
                "Concentration": row[target_element]
            })

    # Combine into one DataFrame
    df_all = pd.DataFrame(records)
    df_pivot = df_all.pivot_table(
        index="Condition (%)", columns="NaOH Equivalents", values="Concentration"
    )

    # Plot heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(df_pivot, cmap="viridis", annot=False, cbar_kws={'label': 'Concentration (mg/L)'})
    plt.title(f"Process Map — {target_element}")
    plt.xlabel("NaOH Equivalents")
    plt.ylabel("Condition (%)")
    plt.tight_layout()
    plt.show()

    return df_pivot

def plot_process_map_matrix(data_dict, sheet_pattern="46"):
    """
    Build a grid of process maps (heatmaps) for all elements across selected sheets.

    Parameters
    ----------
    data_dict : dict
        Dictionary of DataFrames from read_multisheet_excel()
    sheet_pattern : str
        Pattern to filter relevant sheets (e.g., '46')
    """

    records = []

    # ---- Collect all numeric data across sheets ----
    for name, df in data_dict.items():
        if sheet_pattern not in name:
            continue

        # Extract numeric condition value (e.g., 10 from '10%')
        cond_match = re.search(r'(\d+)%', name)
        if not cond_match:
            continue
        condition = float(cond_match.group(1))

        df = df.apply(pd.to_numeric, errors="coerce")

        if "NaOH Equivalents" not in df.columns:
            continue

        # Record all element columns
        for _, row in df.iterrows():
            for col in df.columns:
                if col not in ["NaOH Equivalents", "Vol of NaOH (mL)"] and pd.api.types.is_numeric_dtype(df[col]):
                    records.append({
                        "Condition (%)": condition,
                        "NaOH Equivalents": row["NaOH Equivalents"],
                        "Element": col,
                        "Concentration": row[col]
                    })

    df_all = pd.DataFrame(records)
    if df_all.empty:
        print("⚠️ No valid numeric data found for plotting.")
        return

    # ---- Determine unique elements ----
    elements = sorted(df_all["Element"].unique())
    n = len(elements)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    # ---- Create subplot grid ----
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), constrained_layout=True)
    axes = axes.flatten()

    for idx, element in enumerate(elements):
        ax = axes[idx]
        df_elem = df_all[df_all["Element"] == element]

        pivot = df_elem.pivot_table(
            index="Condition (%)",
            columns="NaOH Equivalents",
            values="Concentration"
        )

        sns.heatmap(pivot, cmap="viridis", ax=ax, cbar=False)
        ax.set_title(element, fontsize=10)
        ax.set_xlabel("NaOH Equivalents")
        ax.set_ylabel("Condition (%)")

    # Remove any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # ---- Add a single shared colorbar ----
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(df_all["Concentration"].values)
    fig.colorbar(sm, cax=cbar_ax, label="Concentration (mg/L)")

    plt.suptitle("Process-Map Matrix (All Elements)", fontsize=14, y=1.02)
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib import gridspec

def plot_custom_process_maps(
    data_dict,
    sheet_pattern="46",
    convert_units=False,
    mw_dict=None,
    unit_converter=None
):
    """
    Plot a fixed 3×3 matrix of process maps with optional unit conversion.
    """

    layout = [
        ["Al 396.152 nm ppm", "Fe 238.204 nm ppm", "Cu 327.395 nm ppm"],
        ["Ni 231.604 nm ppm", "Co 238.892 nm ppm", "Mn 257.610 nm ppm"],
        #["Li 610.365 nm ppm", None, None]
    ]

    records = []

    for name, df in data_dict.items():
        if sheet_pattern not in name:
            continue

        cond_match = re.search(r'(\d+)%', name)
        if not cond_match:
            continue
        condition = float(cond_match.group(1))

        df = df.apply(pd.to_numeric, errors="coerce")
        if "NaOH Equivalents" not in df.columns:
            continue

        for _, row in df.iterrows():
            for col in df.columns:
                if col in ["NaOH Equivalents", "Vol of NaOH (mL)"]:
                    continue
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue

                value = row[col]

                # ---- Optional unit conversion ----
                if convert_units:
                    if unit_converter is None or mw_dict is None:
                        raise ValueError("unit_converter and mw_dict must be provided when convert_units=True")

                    element = col.split()[0]  # e.g. "Al" from "Al 396.152 nm ppm"
                    value = unit_converter(value, mw_dict[element])

                records.append({
                    "Condition (%)": condition,
                    "NaOH Equivalents": row["NaOH Equivalents"],
                    "Vol of NaOH (mL)": row["Vol of NaOH (mL)"],
                    "Element": col,
                    "Concentration": value
                })

    df_all = pd.DataFrame(records)
    if df_all.empty:
        print("⚠️ No valid numeric data found for plotting.")
        return

    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3, hspace=0.2)
    cmap = "viridis"

    for i in range(2): # 3
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            element = layout[i][j]

            if element is None:
                ax.axis("off")
                continue

            df_elem = df_all[df_all["Element"] == element]
            if df_elem.empty:
                ax.text(0.4, 0.4, "No data", ha="center", va="center")
                ax.axis("off")
                continue

            pivot = df_elem.pivot_table(
                index="Condition (%)",
                columns="NaOH Equivalents",
                values="Concentration"
            )

            sns.heatmap(
                pivot,
                cmap="cividis",
                ax=ax,
                cbar=False,
                square=True,
                linewidths=0.4,
                linecolor="white"
            )

            ax.set_title(element, fontsize=11, fontweight="bold")
            ax.set_xlabel("NaOH Equivalents")
            ax.set_ylabel("Battery Leachate Volume in Vial (%)")

    cbar_ax = fig.add_subplot(gs[:, 3])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(df_all["Concentration"].values)

    unit_label = "mol/L" if convert_units else "mg/L"
    fig.colorbar(sm, cax=cbar_ax, label=f"Concentration ({unit_label})")

    plt.savefig("process_map_matrix.png", dpi=300)
    plt.show()

    return df_all

if __name__ == "__main__": 

    # Example usage:
    file_path = "2025_HTE_sample_data_updated4.xlsx"
    #data_dict = read_multisheet_excel(file_path)
    data_dict, meta_df = read_multisheet_with_metadata(file_path)
    meta_df = pd.read_csv("parsed_data/_metadata_summary.csv")
    meta_df_clean = clean_metadata_table(meta_df, save_path="cleaned_metadata.csv")

    print(data_dict)

    # %%
    #data_dict.pop("CC-NRCan3-046 MN 30%", None)

    Al_mol = mgL_to_moles(data_dict["CC-NRCan3-046 HTE 5%"]["Mn 257.610 nm ppm"], MW["Al"])

    # %%

    plot_sheets(data_dict, sheet_names=["CC-NRCan3-046 HTE 5%", "CC-NRCan3-046 HTE 10%", "CC-NRCan3-046 HTE 20%", "CC-NRCan3-046 HTE 30%", "CC-NRCan3-046 HTE 40%"],
                    x_col="NaOH Equivalents",
                    convert_units=True,
                    mw_dict=MW,
                    unit_converter=mgL_to_moles
                )
    build_process_map(data_dict, sheet_pattern="46 HTE", target_element="Ni 231.604 nm ppm")
    build_process_map(data_dict, sheet_pattern="46 HTE", target_element="Co 238.892 nm ppm")
    #plot_process_map_matrix(data_dict, sheet_pattern="46")
    df_all = plot_custom_process_maps(data_dict, sheet_pattern="46 HTE", 
                                    convert_units=True,
                                    mw_dict=MW,
                                    unit_converter=mgL_to_moles
                                    )


