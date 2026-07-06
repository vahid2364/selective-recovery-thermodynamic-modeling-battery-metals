# -*- coding: utf-8 -*-
"""
Created on %(date)s

Project   : Battery Recycling - Hydrometallurgical Process

@author: Vahid Attari
@email : vahid.attari@nrcan-rncan.gc.ca
Affiliation: Natural Resources Canada (NRCan)
Description:
    Auto-generate PHREEQC solution blocks using Latin Hypercube Sampling (LHS)
    while keeping all other sections unchanged.

License:
    This script is released under ....    
"""

import subprocess
import sys
from pathlib import Path

# ===============================================================
# Run PHREEQC after file generation
# ===============================================================

input_file = "auto_generated_hydrolysis_then_NaOHmix_LHS.phr"
output_file = "auto_generated_hydrolysis_then_NaOHmix_LHS.out"
db_path = str(Path.home() / "Applications/phreeqc-3.8.6-17100/database/sit.dat")
cmd = ["phreeqc", input_file, output_file, db_path]

print("PHREEQC run is being started...")

# Execute PHREEQC
result = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", result.returncode)

if result.returncode != 0:
    print("PHREEQC execution failed:")
    print(result.stderr)
    sys.exit(1)   # 🔴 force failure to PBS

if result.returncode < 0:
    print(f"Process killed by signal: {-result.returncode}")
    
print("PHREEQC completed successfully!")
print(result.stdout)

