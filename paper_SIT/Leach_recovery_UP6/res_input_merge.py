import pandas as pd
import numpy as np
import glob
import os


# === Paths ===
lhs_path = "lhs_input/hydrolysis_then_NaOHmix_samples.csv"
results_path = "speciation_results"
os.makedirs("speciation_results_merged", exist_ok=True)

# === Load LHS input ===
print("Loading LHS input...")
lhs_df = pd.read_csv(lhs_path)
lhs_df["SampleID"] = lhs_df.index + 1
print(lhs_df.head())
print("Loaded LHS input...")


def load_results(pattern):
    all_files = glob.glob(os.path.join(results_path, pattern))
    frames = []

    for file in all_files:
        try:
            df = pd.read_csv(file, sep=r"\s+", engine="python", encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError, ValueError) as e:
            print(f"ERROR: Could not read {file}: {e}")
            continue

        if df.empty:
            print(f"WARNING: Empty file: {file}")
            continue

        df.columns = (
            df.columns.str.strip()
            .str.replace('"', '', regex=False)
            .str.replace("'", '', regex=False)
        )

        # Fix known header merge issue (PHREEQC spacing bug)
        df.columns = df.columns.str.replace(
            "si_Co\\(OH\\)2\\(s\\)si_Ni\\(OH\\)2\\(s\\)",
            "si_Co(OH)2(s) si_Ni(OH)2(s)",
            regex=True,
        )

        if "state" not in df.columns:
            print(f"WARNING: 'state' column not found in: {os.path.basename(file)}")
            df = df.tail(1)

        df = df.dropna(axis=1, how="all")
        df.columns = df.columns.str.strip()
        df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

        frames.append(df)

    if not frames:
        print(f"NOTE: No valid files found for {pattern}")
        return pd.DataFrame()

    print(f"OK: Loaded {len(frames)} files for pattern {pattern}")
    return pd.concat(frames, ignore_index=True)


print("Loading solution results files containing everything...")
df_hydro = load_results("hydrolyzed_solution_1.csv")
print(df_hydro.head())
print(np.shape(df_hydro))
print("Loaded solution results...")

# Extracting simulation results — Mix 1
df_mix1_react = df_hydro[(df_hydro["state"] == "react") & (df_hydro["Description"] == "Hydrolyzed_metal_sulfate_solution")]
df_mix1_i_soln = df_hydro[(df_hydro["state"] == "i_soln") & (df_hydro["Description"] == "Hydrolyzed_metal_sulfate_solution")]

# Extracting simulation results — Mix 2
df_mix2_react = df_hydro[(df_hydro["state"] == "react") & (df_hydro["Description"] == "NaOH_mixed_solution")]
df_mix2_eq_react = df_hydro[(df_hydro["state"] == "react") & (df_hydro["Description"] == "NaOH_mixed_solution_EQ")]

print(np.shape(df_mix1_react))
print(np.shape(df_mix1_i_soln))
print(np.shape(df_mix2_react))
print(np.shape(df_mix2_eq_react))

print("Merging results with LHS input...")

df_mix1_react = df_mix1_react.reset_index(drop=True)
df_mix1_react["SampleID"] = df_mix1_react.index + 1

df_mix1_i_soln = df_mix1_i_soln.reset_index(drop=True)
df_mix1_i_soln["SampleID"] = df_mix1_i_soln.index + 1

df_mix2_react = df_mix2_react.reset_index(drop=True)
df_mix2_react["SampleID"] = df_mix2_react.index + 1

df_mix2_eq_react = df_mix2_eq_react.reset_index(drop=True)
df_mix2_eq_react["SampleID"] = df_mix2_eq_react.index + 1

merged_mix1_react = lhs_df.merge(df_mix1_react, on="SampleID", how="left")
merged_mix1_i_soln = lhs_df.merge(df_mix1_i_soln, on="SampleID", how="left")
merged_mix2_react = lhs_df.merge(df_mix2_react, on="SampleID", how="left")
merged_mix2_eq_react = lhs_df.merge(df_mix2_eq_react, on="SampleID", how="left")

# Save final datasets
out_path = "speciation_results_merged/DOE_merged_mix1_react.csv"
merged_mix1_react.to_csv(out_path, index=False)
print(f"\nOK: Merged dataset saved to {out_path}")
print(f"Total merged samples: {len(merged_mix1_react)}")

out_path = "speciation_results_merged/DOE_merged_mix1_i_soln.csv"
merged_mix1_i_soln.to_csv(out_path, index=False)
print(f"\nOK: Merged dataset saved to {out_path}")
print(f"Total merged samples: {len(merged_mix1_i_soln)}")

out_path = "speciation_results_merged/DOE_merged_mix2_react.csv"
merged_mix2_react.to_csv(out_path, index=False)
print(f"\nOK: Merged dataset saved to {out_path}")
print(f"Total merged samples: {len(merged_mix2_react)}")

out_path = "speciation_results_merged/DOE_merged_mix2_eq_react.csv"
merged_mix2_eq_react.to_csv(out_path, index=False)
print(f"\nOK: Merged dataset saved to {out_path}")
print(f"Total merged samples: {len(merged_mix2_eq_react)}")
