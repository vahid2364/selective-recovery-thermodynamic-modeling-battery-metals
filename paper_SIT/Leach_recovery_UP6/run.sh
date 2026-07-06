#!/bin/bash
#set -e            # stop immediately if any command fails
#set -euo pipefail # catch errors in pipes
set -uo pipefail

rm -r figs || true
rm -r figs_mix1* || true
rm -r figs_mix2* || true
rm -r speciation_results || true
rm -r speciation_results_merged || true
rm -r HH_results_backup || true
rm -r error.inp || true
rm -r phreeqc.log || true
rm lhs_input/*.png 2>/dev/null || true
rm lhs_input/*.csv 2>/dev/null || true
rm -r results || true
rm -f job_run.out job_run.err || true

echo "Open files before run:"
lsof -u $USER | wc -l

python3 DOE_phreeqc_init_backup.py
python3 DOE_phreeqc_run.py || { echo "DOE run failed"; exit 1; }
echo "Merging"
python3 res_input_merge_backup.py || exit 1
python3 plot_all_mix1.py
python3 plot_all_mix2_b.py
python3 plot_all_mix2_eq.py
python3 pH_plot_all_HH_backup.py
python3 src/plot_figures_C_to_new.py
#python3 process_window_classification.py
