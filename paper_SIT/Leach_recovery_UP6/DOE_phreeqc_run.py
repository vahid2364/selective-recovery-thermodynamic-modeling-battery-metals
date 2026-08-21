# -*- coding: utf-8 -*-
"""
Project   : Battery Recycling - Hydrometallurgical Process
Author    : Vahid Attari
Email     : vahid.attari@nrcan-rncan.gc.ca
Affiliation: Natural Resources Canada (NRCan)

Description:
    Run PHREEQC on the LHS-generated input file produced by DOE_phreeqc_init.py.

    The path to the sit.dat database is resolved in the following order:
      1. Environment variable PHREEQC_DB  (e.g. export PHREEQC_DB=/usr/share/doc/phreeqc/database/sit.dat)
      2. Common system locations (Linux/macOS)
      3. Falls back to the PHREEQC executable's own bundled database if found

    Set PHREEQC_DB before running if your installation is non-standard:
        export PHREEQC_DB=/path/to/sit.dat
        python3 DOE_phreeqc_run.py

License:
    Creative Commons Attribution 4.0 International (CC BY 4.0)
    https://creativecommons.org/licenses/by/4.0/
"""

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------
# Resolve sit.dat database path
# ---------------------------------------------------------------
_CANDIDATE_DB_PATHS = [
    os.environ.get("PHREEQC_DB", ""),                          # user-set env var
    "/usr/local/share/doc/phreeqc/database/sit.dat",           # Linux (apt)
    "/usr/share/doc/phreeqc/database/sit.dat",
    str(Path.home() / ".local/share/doc/phreeqc/database/sit.dat"),
    "/opt/homebrew/share/phreeqc/database/sit.dat",            # macOS Homebrew
    "/usr/local/opt/phreeqc/database/sit.dat",
]

db_path = next((p for p in _CANDIDATE_DB_PATHS if p and Path(p).is_file()), None)

if db_path is None:
    print(
        "ERROR: sit.dat not found. Set the PHREEQC_DB environment variable:\n"
        "    export PHREEQC_DB=/path/to/database/sit.dat\n"
        "then re-run this script."
    )
    sys.exit(1)

print(f"Using database: {db_path}")

# ---------------------------------------------------------------
# Run PHREEQC
# ---------------------------------------------------------------
input_file  = "auto_generated_hydrolysis_then_NaOHmix_LHS.phr"
output_file = "auto_generated_hydrolysis_then_NaOHmix_LHS.out"
cmd = ["phreeqc", input_file, output_file, db_path]

print("Starting PHREEQC simulation...")
result = subprocess.run(cmd, capture_output=True, text=True)
print(f"Return code: {result.returncode}")

if result.returncode != 0:
    print("PHREEQC execution failed:")
    print(result.stderr)
    sys.exit(1)

if result.returncode < 0:
    print(f"Process killed by signal: {-result.returncode}")
    sys.exit(1)

print("PHREEQC completed successfully.")
if result.stdout:
    print(result.stdout)
