import pandas as pd
import numpy as np
import glob
import os


# === Paths ===
lhs_path = "lhs_input/hydrolysis_then_NaOHmix_samples.csv"
results_path = "speciation_results"
os.makedirs("speciation_results_merged", exist_ok=True)  # ensure results directory exists

# === Load LHS input ===
print("Loading LHS input...")
lhs_df = pd.read_csv(lhs_path)
lhs_df["SampleID"] = lhs_df.index + 1  # ensure consistent IDs
print(lhs_df.head())
print("Loaded LHS input...")

# ==============================================================
# Function to robustly load simulation result CSVs
# ==============================================================
def load_results(pattern, prefix):
    all_files = glob.glob(os.path.join(results_path, pattern))
    frames = []

    for file in all_files:
        sample_id = int(os.path.basename(file).split("_")[-1].split(".")[0])
        try:
            # 🔹 Read space-delimited files with flexible spacing
            df = pd.read_csv(file, sep=r"\s+", engine="python", encoding="utf-8-sig")

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
# Load both hydrolyzed and NaOH-mixed result sets
# ==============================================================
print("Loading solution results files containg everything...")
df_hydro = load_results("hydrolyzed_solution_1.csv", "hydro")
print(df_hydro.head())
print(np.shape(df_hydro))
print("Loaded solution results...")

# --- Extracting simulation results --- Mix 1
#df_mix1_i_soln = df[(df["state"] == "i_soln") & (df["soln"].astype(str).str.startswith("100"))]
df_mix1_react = df_hydro[(df_hydro["state"] == "react") & (df_hydro["Description"] == "Hydrolyzed_metal_sulfate_solution") ]
df_mix1_i_soln = df_hydro[(df_hydro["state"] == "i_soln") & (df_hydro["Description"] == "Hydrolyzed_metal_sulfate_solution")]

# --- Extracting simulation results --- Mix 2
df_mix2_react = df_hydro[(df_hydro["state"] == "react") &  (df_hydro["Description"] == "NaOH_mixed_solution")]
df_mix2_eq_react = df_hydro[(df_hydro["state"] == "react") & (df_hydro["Description"] == "NaOH_mixed_solution_EQ")]
#df_mix2_react = df_hydro[(df_hydro["state"] == "react") & (df_hydro["Description"] == "NaOH_mixed_solution")]
#print(df_mix2_i_soln)
#print(df_mix2_react)

print(np.shape(df_mix1_react))
print(np.shape(df_mix1_i_soln))
print(np.shape(df_mix2_react))
print(np.shape(df_mix2_eq_react))

print("Merging results with LHS input...")

#          16	      i_soln	           6	         -99	         -99	         -99	     7.00808	           4	         -99	      25.000	-1.98523e-22	  9.8193e-08	           1	 1.98523e-22	 1.01088e-13	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	 -7.0081e+00	 -7.0081e+00	 -1.0000e+03	 -1.0000e+03	 -1.0000e+03	 -1.0000e+03	 -1.0000e+03	 -1.0000e+03	 -1.0000e+03	 -1.0000e+03	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	  6.0000e+00	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  7.6452e-26	  0.0000e+00	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  5.5525e+01	  9.8193e-08	  9.8193e-08	Hydrolyzed_metal_sulfate_solution	
#          16	       react	           6	         -99	           0	           1	     5.37485	     13.1781	  1.0000e+00	      25.000	 -2.3707e-06	 0.000403797	           1	 2.43945e-19	 6.00404e-14	  1.4882e-06	  1.1854e-06	  7.3296e-06	  2.4914e-05	  4.7823e-05	  1.8605e-05	  0.0000e+00	  1.0268e-04	 -5.3749e+00	 -8.6413e+00	 -4.0334e+00	 -1.0000e+03	 -6.6249e+00	 -1.1066e+01	 -5.1876e+00	 -4.6431e+00	 -4.3649e+00	 -4.7789e+00	   -999.9990	      1.7594	      0.4996	      5.9996	      1.5587	      3.8387	      2.2787	     -2.4379	     -8.7934	     -4.4152	     -9.2292	  6.0000e+00	  4.3162e-05	  4.7274e-05	  2.2744e-05	  2.4911e-05	  1.6639e-05	  1.8225e-05	  1.0000e-99	  1.0000e-99	  6.9053e-07	  3.5480e-07	  2.3001e-08	  9.9399e-09	  1.6534e-10	  4.5617e-16	  8.3942e-22	  8.1063e-27	  6.9750e-12	  7.8787e-14	  8.1232e-08	  1.3773e-07	  1.0000e-99	  2.3316e-09	  1.3062e-29	  1.0000e-99	  2.4809e-14	  5.4864e-07	  6.0876e-20	  1.0375e-10	  5.8998e-17	  1.3424e-23	  2.8878e-32	  9.2968e-11	  9.8059e-13	  4.2245e-13	  6.7088e-15	  1.7740e-18	  3.2972e-24	  4.7513e-18	  6.5133e-12	  1.0506e-11	  2.0778e-24	  3.3547e-28	  8.2199e-18	  0.0000e+00	  1.6477e-12	  1.0155e-04	  4.0380e-08	  0.0000e+00	  0.0000e+00	  0.0000e+00	  1.0000e-99	  3.8020e-07	  5.4864e-07	  2.3316e-09	  1.3773e-07	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  5.5525e+01	  4.3135e-06	  2.3373e-09	Hydrolyzed_metal_sulfate_solution	
#          17	      i_soln	       10006	         -99	         -99	         -99	     14.2358	           4	         -99	      25.000	     2.75148	     2.33337	           1	-8.88623e-13	-1.90416e-11	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	  2.7515e+00	  0.0000e+00	 -1.4236e+01	  1.8040e-01	 -1.0000e+03	  2.0404e-01	 -1.0000e+03	 -1.0000e+03	 -1.0000e+03	 -1.0000e+03	 -1.0000e+03	 -1.0000e+03	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	   -999.9990	  6.0000e+00	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.6061e-40	  4.4417e-14	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  5.5525e+01	  6.4569e-15	  2.3334e+00	NaOH_mixed_solution	
#          17	       react	       20006	         -99	           0	           1	      13.889	     1.49467	         -99	      25.000	     1.22288	     1.13066	         0.9	-3.56123e-13	-1.74989e-11	  8.2680e-07	  6.5853e-07	  4.0720e-06	  1.3841e-05	  2.6568e-05	  1.0336e-05	  1.2229e+00	  5.7045e-05	 -1.3889e+01	 -1.4493e-01	 -5.4118e+00	 -1.2705e-01	 -3.8935e+01	 -4.0247e+01	 -2.2321e+01	 -2.5428e+01	 -1.5385e+01	 -1.3157e+01	   -999.9990	     -5.0618	     -6.3216	     -0.8216	     -2.1330	      0.1470	     -1.4130	     -2.5789	    -12.5852	      1.5576	     -0.6144	  6.0000e+00	  4.1222e-16	  1.8661e-15	  3.7358e-26	  1.6912e-25	  6.9684e-14	  3.1546e-13	  1.0000e-99	  1.0000e-99	  6.0845e-30	  2.5279e-22	  0.0000e+00	  7.5576e-07	  0.0000e+00	  6.7371e-25	  2.6263e-22	  1.1774e-18	  6.5851e-07	  0.0000e+00	  2.7713e-16	  4.2347e-26	  1.0000e-99	  1.6023e-31	  1.3725e-15	  1.0000e-99	  2.3302e-08	  2.1922e-19	  2.6545e-05	  2.0171e-10	  2.4299e-08	  2.5668e-06	  7.7371e-06	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	  1.5683e-31	  3.8047e-09	  8.5643e-29	  7.7122e-40	  5.6539e-23	  1.8889e-14	  3.1507e-36	  1.0514e-34	  2.4746e-25	  2.8103e-05	  7.6560e-18	  0.0000e+00	  0.0000e+00	  0.0000e+00	  2.8943e-05	  6.6617e-17	  2.1922e-19	  1.6023e-31	  4.2347e-26	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  5.5525e+01	  1.5697e-14	  1.1305e+00	NaOH_mixed_solution	
#          18	       react	       20006	         -99	           0	           1	      13.889	     1.49455	         -99	      25.000	     1.22283	     1.13066	         0.9	-3.57042e-13	-1.75441e-11	  8.2680e-07	  6.5853e-07	  4.0720e-06	  1.3841e-05	  7.3584e-07	  1.0336e-05	  1.2229e+00	  5.7045e-05	 -1.3889e+01	 -1.4492e-01	 -5.4118e+00	 -1.2705e-01	 -3.8935e+01	 -4.0247e+01	 -2.2321e+01	 -2.5428e+01	 -1.6942e+01	 -1.3157e+01	   -999.9990	     -5.0618	     -6.3216	     -0.8216	     -2.1330	      0.1470	     -1.4130	     -2.5790	    -12.5852	     -0.0000	     -0.6144	  6.0000e+00	  1.1416e-17	  5.1681e-17	  3.7355e-26	  1.6911e-25	  6.9679e-14	  3.1544e-13	  1.0000e-99	  1.0000e-99	  6.0841e-30	  2.5278e-22	  0.0000e+00	  7.5576e-07	  0.0000e+00	  6.7386e-25	  2.6269e-22	  1.1777e-18	  6.5851e-07	  0.0000e+00	  2.7712e-16	  4.2343e-26	  1.0000e-99	  1.6022e-31	  1.3726e-15	  1.0000e-99	  6.4536e-10	  6.0713e-21	  7.3520e-07	  2.0170e-10	  2.4299e-08	  2.5668e-06	  7.7372e-06	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	  0.0000e+00	  9.2267e-38	  3.8044e-09	  8.5661e-29	  7.7115e-40	  5.6462e-23	  1.8869e-14	  3.1495e-36	  1.0520e-34	  2.4720e-25	  2.8103e-05	  7.6558e-18	  0.0000e+00	  0.0000e+00	  0.0000e+00	  2.8943e-05	  6.6612e-17	  6.0713e-21	  1.6022e-31	  4.2343e-26	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  1.0000e-99	  5.5525e+01	  1.5697e-14	  1.1305e+00	NaOH_mixed_solution_EQ	


df_mix1_react = df_mix1_react.reset_index()
df_mix1_react["SampleID"] = df_mix1_react.index + 1  # ensure consistent IDs

# --- Extracting simulation results --- Mix 2
#df_react = df[(df["state"] == "react") & (df["soln"].astype(str).str.startswith("200"))]
#df_i_soln = df[(df["state"] == "i_soln") & (df["soln"].astype(str).str.startswith("100"))]

#df_mix2_i_soln = df_mix2_i_soln[df_mix2_i_soln["Description"] == "Hydrolized_mixed_solution"]
df_mix1_i_soln = df_mix1_i_soln.reset_index()
df_mix1_i_soln["SampleID"] = df_mix1_i_soln.index + 1  # ensure consistent IDs
#print(df_mix1_i_soln)
# =============================================================
#df_mix2_react = df_mix2_react[df_mix2_react["Description"] == "NaOH_mixed_solution"]
df_mix2_react = df_mix2_react.reset_index()
df_mix2_react["SampleID"] = df_mix2_react.index + 1  # ensure consistent IDs
#print(df_mix2_react)
# =============================================================
#df_mix2_react = df_mix2_react[df_mix2_react["Description"] == "NaOH_mixed_solution"]
df_mix2_eq_react = df_mix2_eq_react.reset_index()
df_mix2_eq_react["SampleID"] = df_mix2_eq_react.index + 1  # ensure consistent IDs
print(df_mix2_react)
#pause
# ==============================================================
# Merge with LHS input
# ==============================================================

df_mix1_react
merged_mix1_react = lhs_df.merge(df_mix1_react, on="SampleID", how="left")
#print(merged_mix2_i_soln)

merged_mix1_i_soln = lhs_df.merge(df_mix1_i_soln, on="SampleID", how="left")
# #print(merged_mix2_i_soln)

merged_mix2_react = lhs_df.merge(df_mix2_react, on="SampleID", how="left")
# print(merged_mix2_react.head())

merged_mix2_eq_react = lhs_df.merge(df_mix2_eq_react, on="SampleID", how="left")
# print(merged_mix2_eq_react.head())

# ==============================================================
# Save final dataset
# ==============================================================
out_path = "speciation_results_merged/DOE_merged_mix1_react.csv"
merged_mix1_react.to_csv(out_path, index=False)
print(f"\n✅ Merged dataset saved to {out_path}")
print(f"Total merged samples: {len(merged_mix1_react)}")

out_path = "speciation_results_merged/DOE_merged_mix1_i_soln.csv"
merged_mix1_i_soln.to_csv(out_path, index=False)
print(f"\n✅ Merged dataset saved to {out_path}")
print(f"Total merged samples: {len(merged_mix1_i_soln)}")

out_path = "speciation_results_merged/DOE_merged_mix2_react.csv"
merged_mix2_react.to_csv(out_path, index=False)
print(f"\n✅ Merged dataset saved to {out_path}")
print(f"Total merged samples: {len(merged_mix2_react)}")

out_path = "speciation_results_merged/DOE_merged_mix2_eq_react.csv"
merged_mix2_eq_react.to_csv(out_path, index=False)
print(f"\n✅ Merged dataset saved to {out_path}")
print(f"Total merged samples: {len(merged_mix2_eq_react)}")
