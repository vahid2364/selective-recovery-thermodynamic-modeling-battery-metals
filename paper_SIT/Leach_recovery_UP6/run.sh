#!/bin/bash
# run.sh — end-to-end pipeline for thermodynamic modeling of battery leachate neutralization
#
# Usage:
#   bash run.sh
#
# Steps:
#   1. DOE_phreeqc_init.py  — generate 12,001 LHS leachate compositions + PHREEQC input
#   2. DOE_phreeqc_run.py   — run PHREEQC for all samples
#   3. res_input_merge.py   — merge simulation outputs
#   4. Plotting scripts     — generate all manuscript figures

set -euo pipefail

echo "============================================================"
echo " Battery Leachate Thermodynamic Modeling Pipeline"
echo "============================================================"

# Clean previous outputs (warn the user first)
echo "Removing previous output directories..."
rm -rf figs figs_mix1* figs_mix2* speciation_results speciation_results_merged results
rm -f error.inp phreeqc.log job_run.out job_run.err
rm -f lhs_input/*.png lhs_input/*.csv 2>/dev/null || true

echo ""
echo "[1/4] Generating LHS compositions and PHREEQC input..."
python3 DOE_phreeqc_init.py || { echo "ERROR: DOE_phreeqc_init.py failed"; exit 1; }

echo ""
echo "[2/4] Running PHREEQC simulations..."
python3 DOE_phreeqc_run.py || { echo "ERROR: DOE_phreeqc_run.py failed"; exit 1; }

echo ""
echo "[3/4] Merging simulation outputs..."
python3 res_input_merge.py || { echo "ERROR: res_input_merge.py failed"; exit 1; }

echo ""
echo "[4/4] Generating figures..."
python3 plot_all_mix1.py     || { echo "ERROR: plot_all_mix1.py failed"; exit 1; }
python3 plot_all_mix2_b.py   || { echo "ERROR: plot_all_mix2_b.py failed"; exit 1; }
python3 plot_all_mix2_eq.py  || { echo "ERROR: plot_all_mix2_eq.py failed"; exit 1; }
python3 pH_plot_all_HH.py    || { echo "ERROR: pH_plot_all_HH.py failed"; exit 1; }
python3 src/plot_figures_C_to_new.py || { echo "ERROR: plot_figures_C_to_new.py failed"; exit 1; }

echo ""
echo "============================================================"
echo " Pipeline completed successfully."
echo "============================================================"
