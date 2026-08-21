# -*- coding: utf-8 -*-
"""
Project   : Battery Recycling - Hydrometallurgical Process
Author    : Vahid Attari
Email     : vahid.attari@nrcan-rncan.gc.ca
Affiliation: Natural Resources Canada (NRCan)

Description:
    Run PHREEQC on the LHS-generated input file produced by DOE_phreeqc_init.py.
    Streams PHREEQC output in real time and shows a tqdm progress bar based on
    the number of simulation blocks completed.

    The path to the sit.dat database is resolved in the following order:
      1. Environment variable PHREEQC_DB
             export PHREEQC_DB=/path/to/database/sit.dat
      2. Common system locations (Linux / macOS)
      3. Fails with a clear message if not found

License:
    Creative Commons Attribution 4.0 International (CC BY 4.0)
    https://creativecommons.org/licenses/by/4.0/
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
INPUT_FILE   = "auto_generated_hydrolysis_then_NaOHmix_LHS.phr"
OUTPUT_FILE  = "auto_generated_hydrolysis_then_NaOHmix_LHS.out"
N_SAMPLES    = 12001   # expected number of PHREEQC simulation blocks

# ---------------------------------------------------------------
# Resolve sit.dat database path
# ---------------------------------------------------------------
_CANDIDATE_DB_PATHS = [
    os.environ.get("PHREEQC_DB", ""),
    "/usr/local/share/doc/phreeqc/database/sit.dat",
    "/usr/share/doc/phreeqc/database/sit.dat",
    str(Path.home() / ".local/share/doc/phreeqc/database/sit.dat"),
    "/opt/homebrew/share/phreeqc/database/sit.dat",
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

print(f"Database : {db_path}")
print(f"Input    : {INPUT_FILE}")
print(f"Output   : {OUTPUT_FILE}")
print(f"Samples  : {N_SAMPLES:,}")

# ---------------------------------------------------------------
# Run PHREEQC with live progress bar
# ---------------------------------------------------------------
cmd = ["phreeqc", INPUT_FILE, OUTPUT_FILE, db_path]
t0  = time.time()

print("\nStarting PHREEQC simulation...\n")

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

# PHREEQC writes "Reading input data for simulation X" for each block
_sim_re = re.compile(r"Reading input data for simulation\s+(\d+)", re.IGNORECASE)

if tqdm is not None:
    bar = tqdm(total=N_SAMPLES, unit="sim", desc="PHREEQC", ncols=80,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
else:
    bar = None
    print("(install tqdm for a progress bar: pip install tqdm)")

errors = []
for line in proc.stdout:
    m = _sim_re.search(line)
    if m:
        if bar is not None:
            bar.update(1)
        else:
            n = int(m.group(1))
            if n % 500 == 0 or n == N_SAMPLES:
                pct = n / N_SAMPLES * 100
                elapsed = time.time() - t0
                print(f"  {n:>6,}/{N_SAMPLES:,}  ({pct:.1f}%)  [{elapsed:.0f}s elapsed]")
    if "ERROR" in line.upper():
        errors.append(line.rstrip())

if bar is not None:
    bar.close()

proc.wait()
elapsed = time.time() - t0

print(f"\nReturn code : {proc.returncode}")
print(f"Wall time   : {elapsed/60:.1f} min ({elapsed:.0f}s)")

if errors:
    print(f"\nWARNING: {len(errors)} error line(s) detected in PHREEQC output:")
    for e in errors[:10]:
        print(f"  {e}")

if proc.returncode != 0:
    print("\nPHREEQC execution failed. Check the output file for details.")
    sys.exit(1)

print(f"\nPHREEQC completed successfully — output written to {OUTPUT_FILE}")
