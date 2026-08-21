# -*- coding: utf-8 -*-
"""
Parse HTE ICP-OES Excel data for NaOH neutralization experiments.

Each sheet is assumed to have a metadata block above a data table that starts
with a row containing "NaOH Equivalents" as a column header.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec

pd.set_option('future.no_silent_downcasting', True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

MW = {
    "Al": 26.98,
    "Co": 58.93,
    "Cu": 63.55,
    "Fe": 55.85,
    "Li": 6.94,
    "Mn": 54.94,
    "Ni": 58.69,
}

ELEMENTS = ["Al 396.152 nm ppm", "Fe 238.204 nm ppm", "Cu 327.395 nm ppm",
            "Ni 231.604 nm ppm", "Co 238.892 nm ppm", "Mn 257.610 nm ppm"]


def mgL_to_molL(mgL, mw):
    return mgL / 1000.0 / mw


def _find_header_row(df_raw):
    """Return the index of the row containing 'NaOH Equivalents', or None."""
    mask = df_raw.apply(
        lambda col: col.astype(str).str.contains("NaOH Equivalents", case=False)
    ).any(axis=1)
    idx = mask[mask].index
    return idx[0] if len(idx) else None


def _to_numeric_df(df):
    """Strip whitespace/commas and coerce all columns to numeric."""
    for col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace(["", "NA", "nan"], np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _extract_condition(sheet_name):
    """Parse the numeric leachate fraction from a sheet name like '10%'."""
    m = re.search(r'(\d+)%', sheet_name)
    return float(m.group(1)) if m else None


def _collect_records(data_dict, sheet_pattern, convert_units, mw_dict, unit_converter):
    """Gather per-row records from sheets matching sheet_pattern."""
    records = []
    for name, df in data_dict.items():
        if sheet_pattern not in name:
            continue
        condition = _extract_condition(name)
        if condition is None:
            continue
        skip = {"NaOH Equivalents", "Vol of NaOH (mL)"}
        for _, row in df.iterrows():
            for col in df.columns:
                if col in skip or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                value = row[col]
                if convert_units and unit_converter is not None:
                    elem = col.split()[0]
                    if elem in mw_dict:
                        value = unit_converter(value, mw_dict[elem])
                records.append({
                    "Condition (%)": condition,
                    "NaOH Equivalents": row["NaOH Equivalents"],
                    "Vol of NaOH (mL)": row.get("Vol of NaOH (mL)", np.nan),
                    "Element": col,
                    "Concentration": value,
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_sheets(file_path):
    """Read all sheets, detect the data table by header keyword, return dict of DataFrames."""
    excel = pd.ExcelFile(file_path)
    data = {}
    for sheet in excel.sheet_names:
        df_raw = pd.read_excel(file_path, sheet_name=sheet, header=None)
        hr = _find_header_row(df_raw)
        if hr is None:
            print(f"WARNING: 'NaOH Equivalents' header not found in '{sheet}', skipping.")
            continue
        df = df_raw.iloc[hr:].copy()
        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.iloc[1:].reset_index(drop=True)
        df = df.dropna(axis=1, how="all")
        df = _to_numeric_df(df)
        data[sheet] = df
        print(f"  {sheet}: {df.shape[0]} rows x {df.shape[1]} cols")
    return data


def read_sheets_with_metadata(file_path, save_dir="parsed_data"):
    """Read sheets and extract the metadata block (rows above the data table)."""
    os.makedirs(save_dir, exist_ok=True)
    excel = pd.ExcelFile(file_path)
    data = {}
    meta_records = []

    for sheet in excel.sheet_names:
        df_raw = pd.read_excel(file_path, sheet_name=sheet, header=None).fillna("")
        hr = _find_header_row(df_raw)
        if hr is None:
            continue

        # metadata block
        meta_block = df_raw.iloc[:hr].dropna(how="all")
        meta_block = meta_block.loc[:, meta_block.apply(
            lambda c: c.astype(str).str.strip().ne("").any()
        )]
        meta = {"Sheet Name": sheet}
        for _, row in meta_block.iterrows():
            key = str(row.iloc[0]).strip().replace("\n", " ")
            val = str(row.iloc[1]).strip().replace("\n", " ") if len(row) > 1 else ""
            if key:
                meta[key] = val
        meta_records.append(meta)

        # data table
        df = df_raw.iloc[hr:].copy()
        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.iloc[1:].reset_index(drop=True).replace(["", "NA", "nan"], np.nan)
        df = _to_numeric_df(df)
        df.to_csv(f"{save_dir}/{sheet}_data.csv", index=False)
        data[sheet] = df

    meta_df = pd.DataFrame(meta_records)
    meta_df.to_csv(f"{save_dir}/_metadata_summary.csv", index=False)
    print(f"Saved {len(data)} sheets and metadata to '{save_dir}/'")
    return data, meta_df


def clean_metadata(meta_df, save_path="cleaned_metadata.csv"):
    """Rename, cast, and derive columns from the raw metadata summary."""
    df = meta_df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    rename = {
        "Volume of  battery solution in vial": "V_battery_vial_%",
        "Total vial volume": "V_vial_mL",
        "NaOH Stock solution  concentration": "NaOH_M",
        "Stirring duration": "t_stir_h",
        "Temperature": "T_condition",
        "Total reactor volume": "V_reactor_mL",
        "Volume of  battery solution in reactor": "V_battery_reactor_mL",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    numeric = ["V_vial_mL", "V_battery_vial_%", "NaOH_M",
               "t_stir_h", "V_reactor_mL", "V_battery_reactor_mL"]
    for col in numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["V_reactor_mL"] = df["V_reactor_mL"].fillna(df["V_vial_mL"])
    df["V_battery_reactor_mL"] = df["V_battery_reactor_mL"].fillna(
        (df["V_battery_vial_%"] / 100) * df["V_reactor_mL"]
    )
    df["NaOH_mmol_total"] = df["NaOH_M"] * df["V_reactor_mL"]

    for col in ["T_condition", "Experimental method", "Sheet Name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

    df.to_csv(save_path, index=False)
    return df


def plot_sheets(data_dict, sheet_names, x_col="NaOH Equivalents",
                y_columns=None, convert_units=False, mw_dict=None, unit_converter=None):
    for name in sheet_names:
        if name not in data_dict:
            print(f"WARNING: sheet '{name}' not found.")
            continue

        df = data_dict[name].dropna(axis=1, how="all").copy()
        if x_col not in df or "Vol of NaOH (mL)" not in df:
            print(f"WARNING: required columns missing in '{name}'.")
            continue

        y_cols = y_columns or [c for c in df.columns
                               if c != x_col and pd.api.types.is_numeric_dtype(df[c])]
        if not y_cols:
            continue

        if convert_units:
            for c in y_cols:
                elem = c.split()[0]
                if elem in (mw_dict or {}):
                    df[c] = unit_converter(df[c], mw_dict[elem])

        df = df.sort_values(x_col)
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in y_cols:
            ax.plot(df[x_col], df[c], marker="o", label=c)

        xmin, xmax = df[x_col].iloc[0], df[x_col].iloc[-1]
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(x_col)
        ax.set_ylabel("Concentration (mol/L)" if convert_units else "Concentration (mg/L)")

        ax_top = ax.twiny()
        ax_top.set_xlim(xmin, xmax)
        ax_top.set_xlabel("NaOH added (mL)")
        ticks = ax.get_xticks()
        ticks = ticks[(ticks >= xmin) & (ticks <= xmax)]
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels([
            f"{v:.2f}" for v in np.interp(ticks, df[x_col].values, df["Vol of NaOH (mL)"].values)
        ])

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(f"process_plot_{name.replace(' ', '_')}.png", dpi=300)
        plt.show()


def plot_process_map(data_dict, sheet_pattern="46 HTE", target_element="Ni 231.604 nm ppm"):
    """Single-element process map (heatmap) over leachate fraction vs NaOH equivalents."""
    df_all = _collect_records(data_dict, sheet_pattern,
                              convert_units=False, mw_dict=None, unit_converter=None)
    df_all = df_all[df_all["Element"] == target_element]
    if df_all.empty:
        print(f"WARNING: no data for '{target_element}'")
        return

    pivot = df_all.pivot_table(index="Condition (%)", columns="NaOH Equivalents",
                               values="Concentration")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, cmap="viridis", annot=False,
                cbar_kws={"label": "Concentration (mg/L)"})
    plt.title(target_element)
    plt.tight_layout()
    plt.show()
    return pivot


def plot_process_map_grid(data_dict, sheet_pattern="46 HTE",
                          convert_units=False, mw_dict=None, unit_converter=None):
    """3x2 grid of process maps for the six main ICP elements."""
    df_all = _collect_records(data_dict, sheet_pattern,
                              convert_units, mw_dict, unit_converter)
    if df_all.empty:
        print("WARNING: no data found.")
        return

    layout = [row for row in [
        ELEMENTS[:3],
        ELEMENTS[3:],
    ]]
    nrows, ncols = len(layout), 3
    unit_label = "mol/L" if convert_units else "mg/L"

    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[1, 1, 1, 0.05],
                           wspace=0.3, hspace=0.25)

    for i, row_elements in enumerate(layout):
        for j, element in enumerate(row_elements):
            ax = fig.add_subplot(gs[i, j])
            subset = df_all[df_all["Element"] == element]
            if subset.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue
            pivot = subset.pivot_table(index="Condition (%)", columns="NaOH Equivalents",
                                       values="Concentration")
            sns.heatmap(pivot, cmap="cividis", ax=ax, cbar=False,
                        square=True, linewidths=0.4, linecolor="white")
            ax.set_title(element.split()[0], fontsize=11, fontweight="bold")
            ax.set_xlabel("NaOH Equivalents")
            ax.set_ylabel("Leachate vol. (%)")

    cbar_ax = fig.add_subplot(gs[:, ncols])
    sm = plt.cm.ScalarMappable(cmap="cividis")
    sm.set_array(df_all["Concentration"].dropna().values)
    fig.colorbar(sm, cax=cbar_ax, label=f"Concentration ({unit_label})")

    plt.savefig("process_map_grid.png", dpi=300, bbox_inches="tight")
    plt.show()
    return df_all


if __name__ == "__main__":
    FILE = "2025_HTE_sample_data_updated4.xlsx"
    PATTERN = "46 HTE"

    data_dict, meta_df = read_sheets_with_metadata(FILE)
    meta_clean = clean_metadata(meta_df, save_path="cleaned_metadata.csv")

    sheets = [s for s in data_dict if PATTERN in s]
    plot_sheets(data_dict, sheet_names=sheets, x_col="NaOH Equivalents",
                convert_units=True, mw_dict=MW, unit_converter=mgL_to_molL)

    plot_process_map(data_dict, sheet_pattern=PATTERN, target_element="Ni 231.604 nm ppm")
    plot_process_map(data_dict, sheet_pattern=PATTERN, target_element="Co 238.892 nm ppm")

    plot_process_map_grid(data_dict, sheet_pattern=PATTERN,
                          convert_units=True, mw_dict=MW, unit_converter=mgL_to_molL)
