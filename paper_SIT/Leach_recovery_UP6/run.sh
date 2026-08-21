#!/bin/bash
# run.sh — end-to-end pipeline for thermodynamic modeling of battery leachate neutralization
#
# Usage:
#   bash run.sh [--dry-run]
#
# Steps:
#   1. DOE_phreeqc_init.py  — generate 12,001 LHS leachate compositions + PHREEQC input
#   2. DOE_phreeqc_run.py   — run PHREEQC for all samples
#   3. res_input_merge.py   — merge simulation outputs
#   4. Plotting scripts     — generate all manuscript figures

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; CYAN="\033[0;36m"; BOLD="\033[1m"; RESET="\033[0m"

print_header() { echo -e "\n${BOLD}${CYAN}============================================================${RESET}"; echo -e "${BOLD}${CYAN}  $1${RESET}"; echo -e "${BOLD}${CYAN}============================================================${RESET}"; }
print_step()   { echo -e "\n${BOLD}[${1}]${RESET} ${2}"; }
print_ok()     { echo -e "  ${GREEN}✔${RESET}  ${1}"; }
print_warn()   { echo -e "  ${YELLOW}⚠${RESET}  ${1}"; }
print_err()    { echo -e "  ${RED}✖${RESET}  ${1}"; }

# ── Timing helpers ──────────────────────────────────────────────────────────
PIPELINE_START=$(date +%s)
step_start() { _STEP_T=$(date +%s); }
step_done()  { echo -e "  ${GREEN}Done${RESET} ($(( $(date +%s) - _STEP_T ))s)"; }

# ── Dry-run flag ────────────────────────────────────────────────────────────
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

run() {
    if $DRY_RUN; then
        echo -e "  ${YELLOW}[dry-run]${RESET} would run: $*"
    else
        "$@"
    fi
}

# ── Pre-flight checks ───────────────────────────────────────────────────────
print_header "Pre-flight checks"

PREFLIGHT_OK=true

# phreeqc on PATH
if command -v phreeqc &>/dev/null; then
    print_ok "phreeqc found: $(command -v phreeqc)"
else
    print_err "phreeqc not found on PATH. Install PHREEQC 3.8.6 and ensure it is on PATH."
    PREFLIGHT_OK=false
fi

# sit.dat database
if [[ -n "${PHREEQC_DB:-}" && -f "$PHREEQC_DB" ]]; then
    print_ok "PHREEQC_DB: $PHREEQC_DB"
else
    _DB_CANDIDATES=(
        "/usr/local/share/doc/phreeqc/database/sit.dat"
        "/usr/share/doc/phreeqc/database/sit.dat"
        "$HOME/.local/share/doc/phreeqc/database/sit.dat"
        "/opt/homebrew/share/phreeqc/database/sit.dat"
    )
    _DB_FOUND=""
    for _p in "${_DB_CANDIDATES[@]}"; do [[ -f "$_p" ]] && { _DB_FOUND="$_p"; break; }; done
    if [[ -n "$_DB_FOUND" ]]; then
        export PHREEQC_DB="$_DB_FOUND"
        print_ok "sit.dat found at: $PHREEQC_DB (set PHREEQC_DB to override)"
    else
        print_err "sit.dat not found. Set: export PHREEQC_DB=/path/to/database/sit.dat"
        PREFLIGHT_OK=false
    fi
fi

# Python packages
for pkg in numpy pandas matplotlib scipy seaborn tqdm pyDOE2; do
    if python3 -c "import $pkg" &>/dev/null; then
        print_ok "Python: $pkg"
    else
        print_warn "Python package missing: $pkg  →  pip install $pkg"
        PREFLIGHT_OK=false
    fi
done

# Required scripts
for script in DOE_phreeqc_init.py DOE_phreeqc_run.py res_input_merge.py \
              plot_all_mix1.py plot_all_mix2_b.py plot_all_mix2_eq.py \
              pH_plot_all_HH.py src/plot_figures_C_to_new.py; do
    if [[ -f "$script" ]]; then
        print_ok "Script: $script"
    else
        print_err "Script not found: $script"
        PREFLIGHT_OK=false
    fi
done

if ! $PREFLIGHT_OK; then
    echo -e "\n${RED}Pre-flight checks failed. Fix the issues above before running.${RESET}"
    exit 1
fi
print_ok "All checks passed."
$DRY_RUN && echo -e "\n${YELLOW}Dry-run mode — no files will be modified.${RESET}\n"

# ── Clean previous outputs ──────────────────────────────────────────────────
print_step "0/4" "Cleaning previous outputs..."
run rm -rf figs figs_mix1* figs_mix2* speciation_results speciation_results_merged results
run rm -f  error.inp phreeqc.log job_run.out job_run.err
run rm -f  lhs_input/*.png lhs_input/*.csv 2>/dev/null || true
print_ok "Output directories cleared."

# ── Step 1 ──────────────────────────────────────────────────────────────────
print_step "1/4" "Generating LHS compositions and PHREEQC input..."
step_start
run python3 DOE_phreeqc_init.py || { print_err "DOE_phreeqc_init.py failed"; exit 1; }
step_done

# ── Step 2 ──────────────────────────────────────────────────────────────────
print_step "2/4" "Running PHREEQC simulations (12,001 samples — this may take several minutes)..."
step_start
run python3 DOE_phreeqc_run.py  || { print_err "DOE_phreeqc_run.py failed"; exit 1; }
step_done

# ── Step 3 ──────────────────────────────────────────────────────────────────
print_step "3/4" "Merging simulation outputs..."
step_start
run python3 res_input_merge.py  || { print_err "res_input_merge.py failed"; exit 1; }
step_done

# ── Step 4 ──────────────────────────────────────────────────────────────────
print_step "4/4" "Generating manuscript figures..."
step_start
for script in plot_all_mix1.py plot_all_mix2_b.py plot_all_mix2_eq.py \
              pH_plot_all_HH.py src/plot_figures_C_to_new.py; do
    echo -e "  → $script"
    run python3 "$script" || { print_err "$script failed"; exit 1; }
done
step_done

# ── Summary ─────────────────────────────────────────────────────────────────
TOTAL=$(( $(date +%s) - PIPELINE_START ))
MINS=$(( TOTAL / 60 )); SECS=$(( TOTAL % 60 ))

N_CSV=$(find speciation_results_merged -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')
N_FIG=$(find figs_mix1* figs_mix2* -name "*.png" 2>/dev/null | wc -l | tr -d ' ')

print_header "Pipeline summary"
echo -e "  ${GREEN}✔${RESET}  Status          : completed successfully"
echo -e "  ${GREEN}✔${RESET}  Total wall time  : ${MINS}m ${SECS}s"
echo -e "  ${GREEN}✔${RESET}  Merged CSVs      : ${N_CSV}"
echo -e "  ${GREEN}✔${RESET}  Figures generated: ${N_FIG}"
echo -e "\n  Results → speciation_results_merged/   Figures → figs_mix*/"
echo ""
