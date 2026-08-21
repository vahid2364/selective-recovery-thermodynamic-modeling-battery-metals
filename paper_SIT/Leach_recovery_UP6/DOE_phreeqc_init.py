# -*- coding: utf-8 -*-
"""
Created: 2025
Project   : Battery Recycling - Hydrometallurgical Process
Author    : Vahid Attari
Email     : vahid.attari@nrcan-rncan.gc.ca
Affiliation: Natural Resources Canada (NRCan)

Description:
    Auto-generate PHREEQC input files where each LHS sample
    performs:
        (1) Metal sulfate hydrolysis in DI water
        (2) Save equilibrated solution
        (3) Neutralization with sampled NaOH dose
"""

import os
import numpy as np
import pandas as pd
from pyDOE2 import lhs
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

# ===============================================================
os.makedirs('speciation_results', exist_ok=True)
# ===============================================================

def is_constant(series, tol=1e-12):
    """Return True if the column has essentially zero variation."""
    return (series.max() - series.min()) < tol

# ===============================================================
# CONFIGURATION
# ===============================================================
n_per_dim = 200
n_samples = 12000
output_file = "auto_generated_hydrolysis_then_NaOHmix_LHS.phr"

# mg/L concentrations from your experiment at NaOH = 0
exp_mgL = {
    "Al": 254.6, #mg/L
    "Co": 3861.4,
    "Cu": 1169.8,
    "Fe": 237,
    "Li": 482.4,
    "Mn": 3144.6,
    "Ni": 7400.6,
}

molar_mass = {
    "Al": 26.98, # g/mol
    "Co": 58.93,
    "Cu": 63.55,
    "Fe": 55.85,
    "Li": 6.94,
    "Mn": 54.94,
    "Ni": 58.69,
}

charge = {
    "Al": 3,
    "Co": 2,
    "Cu": 2,
    "Fe": 3,   # Fe likely Fe3+ in your sulfate feed
    "Li": 1,
    "Mn": 2,
    "Ni": 2,
}

V_L = 0.001 # Volume of leachate in liters (1 mL = 0.001 L)
moles_total = {
    m: (exp_mgL[m] / 1000 / molar_mass[m]) * V_L
    for m in exp_mgL
}

moles = {}
for metal, mgL in exp_mgL.items():
    mm = molar_mass[metal]
    moles[metal] = mgL / 1000 / mm  # mg/L -> g/L -> mol/L

print(moles_total)

print("------------------------------------------------------")
eq_total = sum(moles_total[m] * charge[m] for m in moles)
print("Total equivalents needed:", eq_total)

NaOH_conc = 2.5   # mol/L

def equivalents_added(volume_L, NaOH_conc=1.0, eq_total=eq_total):
    moles_NaOH = NaOH_conc * volume_L
    return moles_NaOH / eq_total

print("------------------------------------------------------")
eq = equivalents_added(0.294/1000, NaOH_conc)  # 0.294 mL NaOH
print("Equivalents added:", eq)


# %%

## moles in a small aliquot
#def mgL_to_moles(mgL, MW):
#    mg = mgL * 0.001         # mg in 1 mL vial
#    mmol = mg / MW           # mmol
#    return mmol / 1000       # convert to mol

## mol/L 
def mgL_to_moles(mgL, MW):
    mg = mgL                        # mg per liter
    g = mg / 1000                   # convert mg → g
    mol = g / MW                    # moles = g/MW
    return mol

MW = {
    "Al2(SO4)3": 342.15, # g/mol
    "Fe2(SO4)3": 399.88,
    "CuSO4": 159.6,
    "CoSO4": 154.99,
    "NiSO4": 154.75,
    "MnSO4": 169.02,
    "Li2SO4": 109.94,
}

Al_mol = mgL_to_moles(exp_mgL["Al"],   MW["Al2(SO4)3"])
Fe_mol = mgL_to_moles(exp_mgL["Fe"],   MW["Fe2(SO4)3"])
Cu_mol = mgL_to_moles(exp_mgL["Cu"],   MW["CuSO4"])
Co_mol = mgL_to_moles(exp_mgL["Co"],   MW["CoSO4"])
Ni_mol = mgL_to_moles(exp_mgL["Ni"],   MW["NiSO4"])
Mn_mol = mgL_to_moles(exp_mgL["Mn"],   MW["MnSO4"])
Li_mol = mgL_to_moles(exp_mgL["Li"],   MW["Li2SO4"])

print(Al_mol,Fe_mol,Cu_mol,Co_mol,Ni_mol,Mn_mol,Li_mol)
print("------------------------------------------------------")

# --- 2. Define ±20% sampling window ---
#def pm20(x):    # plus/minus 20%
#    return (0.5 * x, 1.5 * x)

import numpy as np
import pandas as pd
import itertools

def metal_to_sulfate_equivalent(exp_mgL, fe_state="Fe3+"):
    """
    Convert elemental metal concentrations (mg/L from ICP)
    to stoichiometrically equivalent metal sulfate concentrations (mg/L).

    Parameters
    ----------
    exp_mgL : dict
        {metal_symbol: mg/L elemental concentration}
    fe_state : str
        "Fe3+" → Fe2(SO4)3
        "Fe2+" → FeSO4

    Returns
    -------
    dict
        {metal_symbol: sulfate_mgL}
    """

    # Atomic weights (g/mol)
    atomic_wt = {
        "Al": 26.9815,
        "Co": 58.933,
        "Cu": 63.546,
        "Fe": 55.845,
        "Li": 6.94,
        "Mn": 54.938,
        "Ni": 58.693,
        "S": 32.065,
        "O": 15.999,
    }

    # Sulfate group molar mass
    SO4 = atomic_wt["S"] + 4 * atomic_wt["O"]

    # Sulfate definitions: (formula weight, metals per unit)
    sulfate_defs = {
        "Al": (2 * atomic_wt["Al"] + 3 * SO4, 2),   # Al2(SO4)3
        "Co": (atomic_wt["Co"] + SO4, 1),           # CoSO4
        "Cu": (atomic_wt["Cu"] + SO4, 1),           # CuSO4
        "Li": (2 * atomic_wt["Li"] + SO4, 2),       # Li2SO4
        "Mn": (atomic_wt["Mn"] + SO4, 1),           # MnSO4
        "Ni": (atomic_wt["Ni"] + SO4, 1),           # NiSO4
    }

    # Iron handling
    if fe_state == "Fe3+":
        sulfate_defs["Fe"] = (2 * atomic_wt["Fe"] + 3 * SO4, 2)  # Fe2(SO4)3
    elif fe_state == "Fe2+":
        sulfate_defs["Fe"] = (atomic_wt["Fe"] + SO4, 1)          # FeSO4
    else:
        raise ValueError("fe_state must be 'Fe3+' or 'Fe2+'")

    print('sulfate_defs:')
    print(sulfate_defs)

    sulfate_mgL = {}

    for metal, mgL in exp_mgL.items():
        if metal not in sulfate_defs:
            raise KeyError(f"Sulfate definition missing for {metal}")

        fw_sulfate, stoich = sulfate_defs[metal]

        # mol/L metal
        mol_metal = mgL / (atomic_wt[metal] * 1000)

        # mol/L sulfate salt
        mol_sulfate = mol_metal / stoich

        # mg/L sulfate
        sulfate_mgL[metal] = mol_sulfate * fw_sulfate * 1000

    return sulfate_mgL



def total_metal_moles_from_icp(exp_mgL, volume_mL):
    """
    Convert ICP metal concentrations (mg/L) to total moles
    in a vial of known volume.

    Parameters
    ----------
    exp_mgL : dict
        {metal_symbol: mg/L}
    volume_mL : float
        Sample volume in mL

    Returns
    -------
    dict
        {metal_symbol: total mol}
    """

    atomic_wt = {
        "Al": 26.9815, # MW:342.15, #
        "Co": 58.933,
        "Cu": 63.546,
        "Fe": 55.845,
        "Li": 6.94,
        "Mn": 54.938,
        "Ni": 58.693,
    }

    atomic_wt_salt = {
        "Al": 342.15, #
        "Co": 154.99,
        "Cu": 159.61,
        "Fe": 399.88,
        "Li": 109.94,
        "Mn": 151.00,
        "Ni": 154.76,
    }

    volume_L = volume_mL / 1000.0

    total_mol = {}

    # m = C * V * (1/MW) * 1000
    for metal, mgL in exp_mgL.items():
        mol = (mgL * volume_L) / (atomic_wt_salt[metal] * 1000)
        total_mol[metal] = mol

    return total_mol


######
######  Sampling functions
######

def uniform_grid_sampling(param_ranges, n_per_dim, log_params=None):
    log_params = set(log_params or [])

    keys = list(param_ranges.keys())
    variable_params = []
    constant_params = {}

    for k in keys:
        low, high = param_ranges[k]
        if abs(high - low) < 1e-15:
            constant_params[k] = low
        else:
            variable_params.append((k, low, high))

    grids = []
    grid_keys = []

    for k, low, high in variable_params:
        N = n_per_dim[k] if isinstance(n_per_dim, dict) else n_per_dim

        if k in log_params:
            grids.append(np.logspace(np.log10(low), np.log10(high), N))
        else:
            grids.append(np.linspace(low, high, N))

        grid_keys.append(k)

    samples_matrix = (
        np.array(list(itertools.product(*grids)))
        if grids else np.zeros((1, 0))
    )

    df_samples = pd.DataFrame(samples_matrix, columns=grid_keys)

    for k, v in constant_params.items():
        df_samples[k] = v

    df_samples = df_samples[keys]
    return df_samples, samples_matrix

def biased_grid_sampling(param_ranges, n_per_dim, power=2.0):
    """
    Non-uniform grid sampling biased toward lower bounds.

    param_ranges: dict {name: (low, high)}
    n_per_dim:    int or dict {name: N}
    power:        float > 1  (larger → more density near lower bound)

    Returns:
        df_samples : DataFrame with all parameters
        samples_matrix : numpy array of varying-parameter samples
    """

    import numpy as np
    import pandas as pd
    import itertools

    keys = list(param_ranges.keys())

    variable_params = []
    constant_params = {}

    for k in keys:
        low, high = param_ranges[k]
        if abs(high - low) < 1e-15:
            constant_params[k] = low
        else:
            variable_params.append((k, low, high))

    grids = []
    grid_keys = []

    for k, low, high in variable_params:
        if isinstance(n_per_dim, dict):
            N = n_per_dim[k]
        else:
            N = n_per_dim

        t = np.linspace(0, 1, N)
        t_biased = t**power          # bias toward lower bound
        grid = low + (high - low) * t_biased

        grids.append(grid)
        grid_keys.append(k)

    if grids:
        samples_matrix = np.array(list(itertools.product(*grids)))
    else:
        samples_matrix = np.zeros((1, 0))

    df_samples = pd.DataFrame(samples_matrix, columns=grid_keys)

    for k, v in constant_params.items():
        df_samples[k] = v

    df_samples = df_samples[keys]

    return df_samples, samples_matrix


######
########  Battery composition sampling
###### Battery composition sampling (black mass consistent)

import numpy as np
import pandas as pd

battery_classes = {
    "NMC111": {"Ni": 1, "Mn": 1, "Co": 1},
    "NMC532": {"Ni": 5, "Mn": 3, "Co": 2},
    "NMC523": {"Ni": 5, "Mn": 2, "Co": 3},
    "NMC622": {"Ni": 6, "Mn": 2, "Co": 2},
    "NMC811": {"Ni": 8, "Mn": 1, "Co": 1},
    #"NCA":    {"Ni": 8, "Co": 1, "Al": 1},
    #"LCO":    {"Co": 1},
}

metals = ["Al", "Fe", "Cu", "Co", "Ni", "Mn"]
independent_metals = ["Al", "Fe", "Cu"]


def sample_battery(param_ranges):
    
    # 1. choose chemistry
    chem = np.random.choice(list(battery_classes.keys()))
    ratios = battery_classes[chem]
    
    comp = {}

    # 2. --- cathode metals (preserve ratios) ---
    noisy = {m: ratios[m] * np.random.normal(1, 0.05) for m in ratios}
    total_ratio = sum(noisy.values())
    fractions = {m: noisy[m] / total_ratio for m in noisy}

    # choose total cathode loading
    ref_metal = "Ni" if "Ni" in ratios else list(ratios.keys())[0]
    low, high = param_ranges[f"{ref_metal}_sulfate_mol"]
    total_cathode = np.random.uniform(low, high)

    for m in fractions:
        comp[m] = fractions[m] * total_cathode

    # 3. --- independent metals (impurities) ---
    for m in independent_metals:
        low, high = param_ranges[f"{m}_sulfate_mol"]
        impurity = np.random.uniform(low, high)

        if m in comp:
            comp[m] += 0.1 * impurity  # small addition
        else:
            comp[m] = impurity

    # 4. --- enforce chemistry constraints ---
    if chem == "LCO":
        comp["Ni"] = 0.0
        comp["Mn"] = 0.0

    if chem == "NCA":
        comp["Mn"] = 0.0  # NCA should not contain Mn

    # 5. --- safe fill (ONLY allowed metals) ---
    allowed_metals = set(ratios.keys()) | set(independent_metals)

    for m in metals:
        if m not in comp:
            if m in allowed_metals:
                low, high = param_ranges[f"{m}_sulfate_mol"]
                comp[m] = np.random.uniform(low, high)
            else:
                comp[m] = 0.0  # enforce absence

    return {f"{m}_sulfate_mol": comp[m] for m in metals}, chem


def sample_battery_df(param_ranges, N):
    rows = []
    
    for _ in range(N):
        sample, chem = sample_battery(param_ranges)
        sample["chem"] = chem
        rows.append(sample)
    
    return pd.DataFrame(rows)
#
#


def lhs_sampling(param_ranges, n_samples, log_params=None):
    log_params = log_params or []
    keys = list(param_ranges.keys())

    # LHS in [0,1]
    lhs_unit = lhs(len(keys), samples=n_samples)

    data = {}
    for i, k in enumerate(keys):
        low, high = param_ranges[k]

        if k in log_params:
            low, high = np.log10(low), np.log10(high)
            vals = 10 ** (low + (high - low) * lhs_unit[:, i])
        else:
            vals = low + (high - low) * lhs_unit[:, i]

        data[k] = vals

    return pd.DataFrame(data)


######
######  Unit Conversion functions
######

def convert_to_mix_fractions(V_leach_mL, V_NaOH_mL):
    """
    Convert real mL volumes into PHREEQC MIX fractions (dimensionless).

    Parameters:
        V_leach_mL : float  (mL of metal leachate)
        V_NaOH_mL  : float  (mL of NaOH solution)

    Returns:
        (f_leach, f_NaOH) dimensionless fractions
    """

    # Avoid division by zero
    V_total = V_leach_mL + V_NaOH_mL
    if V_total <= 0:
        return (0.0, 0.0)

    f_leach = V_leach_mL / V_total
    f_NaOH  = V_NaOH_mL  / V_total

    return f_leach, f_NaOH


def pm20_log(x):
    """
    ±20% *in log10 space*.
    Handles concentrations from 1e-3 to 1e-12 smoothly.
    """
    if x <= 0:
        raise ValueError("Value must be positive for log sampling.")

    logx = np.log10(x)
    return (10**(logx - 0.2), 10**(logx + 0.2))

def log_span(x, orders=0):
    """
    Sample ±(orders) orders of magnitude around x in log10 space.
    
    Example:
        orders = 2 → x*(10^-2) to x*(10^+2)
        orders = 3 → x*(10^-3) to x*(10^+3)
    """
    if x <= 0:
        raise ValueError("Value must be positive for log sampling.")

    logx = np.log10(x)
    return (10**(logx - orders), 10**(logx + orders))


def convert_composition_to_moles(salt_data, volume_L=1.0):

    salt_moles = {}
    ion_moles = {}

    for salt in salt_data:
        name = salt["name"]
        g_per_L = salt["g_per_L"]
        MW = salt["MW"]
        stoich = salt["stoich"]

        # grams in system volume
        grams = g_per_L * volume_L

        # moles of salt
        moles = grams / MW
        salt_moles[name] = moles

        # ion-level moles
        for species, coeff in stoich.items():
            ion_moles[species] = ion_moles.get(species, 0) + coeff * moles

    return {"salt_moles": salt_moles, "ion_moles": ion_moles}

try: 

    print("------------------------------------------------------") 
    # salt_data = [
    #     {"name": "Al2(SO4)3·18H2O", "g_per_L": 14.0,  "MW": 666.42, "stoich": {"Al": 2, "SO4": 3}},
    #     {"name": "FeSO4·7H2O", "g_per_L": 5.6,  "MW": 278.01, "stoich": {"Fe": 1, "SO4": 1}},
    #     {"name": "CoSO4·7H2O", "g_per_L": 79.7, "MW": 281.10, "stoich": {"Co": 1, "SO4": 1}},
    #     {"name": "CuSO4·5H2O", "g_per_L": 13.2, "MW": 249.69, "stoich": {"Cu": 1, "SO4": 1}},
    #     {"name": "Li2SO4·H2O", "g_per_L": 18.5, "MW": 127.96, "stoich": {"Li": 2, "SO4": 1}},
    #     {"name": "MnSO4·H2O", "g_per_L": 41.9,  "MW": 169.02, "stoich": {"Mn": 1, "SO4": 1}},
    #     {"name": "NiSO4·6H2O", "g_per_L": 146.3,"MW": 262.84, "stoich": {"Ni": 1, "SO4": 1}},
    # ]

    salt_data = [
        {"name": "Al2(SO4)3", "g_per_L": 9.8,  "MW": 342.15, "stoich": {"Al": 2, "SO4": 3}},
        {"name": "FeSO4·7H2O", "g_per_L": 3.9,  "MW": 278.01, "stoich": {"Fe": 1, "SO4": 1}},
        {"name": "CoSO4·7H2O", "g_per_L": 55.8, "MW": 281.10, "stoich": {"Co": 1, "SO4": 1}},
        {"name": "CuSO4", "g_per_L": 9.2, "MW": 159.61, "stoich": {"Cu": 1, "SO4": 1}},
        {"name": "Li2SO4·H2O", "g_per_L": 12.9, "MW": 127.96, "stoich": {"Li": 2, "SO4": 1}},
        {"name": "MnSO4·H2O", "g_per_L": 29.4,  "MW": 169.02, "stoich": {"Mn": 1, "SO4": 1}},
        {"name": "NiSO4·6H2O", "g_per_L": 102.4,"MW": 262.84, "stoich": {"Ni": 1, "SO4": 1}},
    ]

    # --- call function ---
    result = convert_composition_to_moles(salt_data, volume_L=1)  # 1 L or 100 mL system

    # --- access outputs ---
    salt_moles = result["salt_moles"]
    ion_moles  = result["ion_moles"]

    # --- print nicely ---
    print("Salt moles:")
    for k, v in salt_moles.items():
        print(f"{k}: {v:.4f} mol")

    print("\nIon moles:")
    for k, v in ion_moles.items():
        print(f"{k}: {v:.4f} mol")

    #pause

    print("------------------------------------------------------") 
    sulfates = metal_to_sulfate_equivalent(exp_mgL, fe_state="Fe3+")
    print("Sulfate equivalent mg/L:") 
    print(sulfates) 
    print("------------------------------------------------------") 

    total_metal_moles = total_metal_moles_from_icp(exp_mgL, volume_mL=1*0.3)  # 1 mL vial
    print("Total metal from ICP (mole):") 
    print(total_metal_moles) 
    #print(total_metal_moles['Al'])

    total_sulfate_moles = total_metal_moles_from_icp(sulfates, volume_mL=1*0.3)  # 1 mL vial
    print("Total sulfates from ICP (mole):") 
    print(total_sulfate_moles) 
    print("------------------------------------------------------") 
    ####
    ####
    total_metal_moles = salt_moles
    ####
    ####
    # ===============================================================

    
    param_ranges = {
        "temp": (25, 25),           # °C
        ##
        "NaOH_stock": (1, 1),   # M NaOH
        #"NaOH_mole": (0.05, 1.0), # NaOH mole - 0.5 L of 1 M NaOH = 0.5 mol
        "NaOH_mole": (0.02, 1.5), # NaOH mole - 0.5 L of 2.5 M NaOH = 1.25 mol
        ##
        "BattLeachate_volume": (1, 1),   # leachate fraction (0.1–1.0)
        ##
        #"V_batt_pct": (5, 40),     # % leachate volume at vial
    }

    od = 0.9
    comp_ranges = {
        # metal sulfate molar amounts (±20% around experiment)
        # "Al_sulfate_mol": log_span(total_metal_moles['Al'], orders=1), #log_span(Al_mol),
        # "Fe_sulfate_mol": log_span(total_metal_moles['Fe'], orders=1), #log_span(Fe_mol),
        # "Cu_sulfate_mol": log_span(total_metal_moles['Cu'], orders=1), #log_span(Cu_mol),
        # "Co_sulfate_mol": log_span(total_metal_moles['Co'], orders=1), #log_span(Co_mol),
        # "Ni_sulfate_mol": log_span(total_metal_moles['Ni'], orders=1), #log_span(Ni_mol),
        # "Mn_sulfate_mol": log_span(total_metal_moles['Mn'], orders=1), #log_span(Mn_mol),
        #
        "Al_sulfate_mol": log_span(total_metal_moles['Al2(SO4)3'], orders=od), 
        "Fe_sulfate_mol": log_span(total_metal_moles['FeSO4·7H2O'], orders=od),
        "Cu_sulfate_mol": log_span(total_metal_moles['CuSO4'], orders=od), 
        "Co_sulfate_mol": log_span(total_metal_moles['CoSO4·7H2O'], orders=od), 
        "Ni_sulfate_mol": log_span(total_metal_moles['NiSO4·6H2O'], orders=od),
        "Mn_sulfate_mol": log_span(total_metal_moles['MnSO4·H2O'], orders=od), 
    }

    print("param_ranges")
    print(param_ranges)
    print("comp_ranges")
    print(comp_ranges)

    # ===============================================================
    # Battery Type SAMPLING
    # ===============================================================

    # 1. Generate NaOH and other series once
    df_other = lhs_sampling(param_ranges, n_samples, log_params=["NaOH_mole"])
    print(df_other)

    # NaOH_df, _ = uniform_grid_sampling(
    #     {"NaOH_mole": param_ranges["NaOH_mole"]},
    #     n_per_dim=t_samples,
    #     log_params=["NaOH_mole"]
    # )
    # NaOH_values = NaOH_df["NaOH_mole"].values
    # print(NaOH_values)

    # 2. Generate sulfate samples once
    # sulfate_ranges = {k: v for k, v in param_ranges.items() if "sulfate" in k}
    # df_sulfates = lhs_sampling(
    #     sulfate_ranges,
    #     n_samples=n_samples,
    #     log_params=list(sulfate_ranges.keys())
    # )
    # print(df_sulfates)

    # 2. Generate sulfate samples once -- Battery Type SAMPLING
    df_sulfates = sample_battery_df(comp_ranges, N=n_samples)
    print(df_sulfates)

    df_final = pd.concat([df_sulfates, df_other], axis=1)
    samples = df_final.copy()




    # ===============================================================
    # ADDING SDL composition as a fixed point
    # ===============================================================


    values_star = [
        {"Al_sulfate_mol": 0.041},
        {"Co_sulfate_mol": 0.284},
        {"Cu_sulfate_mol": 0.083},
        {"Fe_sulfate_mol": 0.020},
        #{"Li_sulfate_mol": 0.145},
        {"Mn_sulfate_mol": 0.248},
        {"Ni_sulfate_mol": 0.557},
        {"Ni_sulfate_mol": 0.557},
        {"chem": "NMC523 - SDL"},
        {"temp": 25,},
        {"NaOH_mole": 0.75}, 
        {"BattLeachate_volume": 1}, 
        {"NaOH_stock": 1}
    ]

    # flatten list of dicts → one dict
    row = {}
    for d in values_star:
        row.update(d)

    df_new = pd.DataFrame([row])

    # merge (append as new row)
    df_final = pd.concat([df_final, df_new], ignore_index=True)
    print(df_final)

    samples = df_final.copy()

    # ===============================================================
    # UNIFORM SAMPLING
    # ===============================================================

    #keys = list(param_ranges.keys())
    #df_grid, none = uniform_grid_sampling(param_ranges, 
    #                                      n_per_dim=n_per_dim, 
    #                                      log_params=["Al_sulfate_mol","Fe_sulfate_mol"]
    #                                      )
    
    #df_grid, none = biased_grid_sampling(
    #    param_ranges=param_ranges,
    #    n_per_dim=n_per_dim,
    #    power=2.0
    #)
    
    #samples = df_grid.copy()
    #print(samples)

    # ===============================================================
    # LHS SAMPLING
    # ===============================================================
    
    # # keys → list of parameter names
    # # Latin Hypercube sampling matrix
    # keys = list(param_ranges.keys())
    # lhs_matrix = lhs(len(keys), samples=n_samples)

    # scaled = {}
    # for i, k in enumerate(keys):

    #     low, high = param_ranges[k]

    #     # --------------------------------------------------------
    #     # 1️⃣ Log-sampling for sulfate mol parameters
    #     #    Detect sulfate species by "_mol" suffix
    #     # --------------------------------------------------------
    #     if k.endswith("_mol"):  
    #         # low and high are already 10**(logx ± orders)
    #         # so sample directly in log10-space
    #         log_low = np.log10(low)
    #         log_high = np.log10(high)

    #         # LHS → log-space
    #         scaled_vals = 10 ** (log_low + lhs_matrix[:, i] * (log_high - log_low))
    #         scaled[k] = scaled_vals

    #     # --------------------------------------------------------
    #     # 2️⃣ Linear sampling for all other parameters
    #     # --------------------------------------------------------
    #     else:
    #         scaled_vals = low + lhs_matrix[:, i] * (high - low)
    #         scaled[k] = scaled_vals

    # # Convert to DataFrame
    # samples = pd.DataFrame(scaled)

    # Save
    os.makedirs("lhs_input", exist_ok=True)
    samples.to_csv(
        "lhs_input/hydrolysis_then_NaOHmix_samples.csv",
        index=False,
        float_format="%.4g",
    )


    # Loop through each parameter
    for col in samples.select_dtypes(include="number").columns:
        if is_constant(samples[col]):
            print(f"Skipping KDE for '{col}' (constant column)")
            continue

        data = samples[col].dropna()
        data = data[data > 0]
        if (data <= 0).any():
            print(f"Skipping '{col}' (non-positive values for log scale)")
            continue

        plt.figure()

        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 50)
        plt.hist(data, bins=bins, alpha=0.6, color='blue', edgecolor='black')
        print("col:", col)

        # --- overlay only relevant salts ---
        for salt in salt_data:
            if col[:2] in salt["stoich"]:
                mol_per_L = salt["g_per_L"] / salt["MW"]

                # scale by stoichiometric coefficient
                value = mol_per_L * salt["stoich"][col[:2]]

                plt.axvline(value, linestyle='--', linewidth=1)

                plt.text(value, plt.ylim()[1]*0.72, salt["name"],
                        rotation=90, va='bottom', ha='right', fontsize=12)
                plt.xlabel(salt["name"], fontsize=17)

        plt.xscale("log")
        #plt.xlabel(col, fontsize=14)
        plt.ylabel("Count", fontsize=17)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(f"lhs_input/hist_{col}.png")
        plt.close()


    import itertools

    plt.figure(figsize=(7, 5))

    # --- define styles ONCE ---
    colors = [
        "#000000",  # black
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
    ]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
    #linewidths = [2] * len(colors)
    linewidths = [3, 3, 3, 3, 3, 3, 4]

    style_cycle = itertools.cycle(zip(colors, linestyles, linewidths))

    label_map = {
        "Al_sulfate_mol": "Al$_2$(SO$_4$)$_3",
        "Fe_sulfate_mol": "FeSO$_4$·7H$_2$O",
        "Co_sulfate_mol": "CoSO$_4$·7H$_2$O",
        "Cu_sulfate_mol": "CuSO$_4$",
        "Ni_sulfate_mol": "NiSO$_4$·6H$_2$O",
        "Mn_sulfate_mol": "MnSO$_4$·H$_2$O",
        "NaOH_mole": "NaOH dose",   
    }

    for col in samples.select_dtypes(include="number").columns:
        data = samples[col].dropna()
        data = data[data > 0]
        if len(data) == 0:
            continue

        log_data = np.log10(data)

        color, ls, lw = next(style_cycle)

        sns.kdeplot(
            log_data,
            label=label_map.get(col, col),
            color=color,
            linestyle=ls,
            linewidth=lw
        )

    # --- labels OUTSIDE loop ---
    plt.xlabel("log$_{10}$(Concentration)", fontsize=16)
    plt.ylabel("Density", fontsize=18)
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=14.5, frameon=False, handlelength=2.5)

    plt.tight_layout()
    plt.savefig("lhs_input/kdePlot_all.png", dpi=300)
    plt.close()

    #pause
    
        
except Exception as e:
    print("ERROR: Error while defining param_ranges:")
    print("   →", str(e))
    raise

# ===============================================================
# HEADER
# ===============================================================
header = """TITLE Hydrolysis + NaOH neutralization workflow

PHASES
Al(OH)3(am)
    Al(OH)3 = Al+3 - 3H+ + 3H2O
    log_k 6.5 # 6-7 from literature
Al(OH)3_metastable
    Al(OH)3 = Al+3 - 3H+ + 3H2O
    log_k 3.5     # or from literature
Fe(OH)3(a)
    Fe(OH)3 = Fe+3 - 3H+ + 3H2O
    log_k 3.5
Fe(OH)3(cr)
    Fe(OH)3 = Fe+3 - 3H+ + 3H2O
    log_k 1.22
Fe(OH)3(s)
    Fe(OH)3 = Fe+3 - 3H+ + 3H2O
    log_k 2.78
    -analytic 27.8E-1 00E+0 00E+0 00E+0 00E+0
Cu(OH)2(s)
    Cu(OH)2 = Cu+2 - 2H+ + 2H2O
    log_k 8.34
#    delta_h -62.764 #kJ/mol
#   Enthalpy of formation: -443.996  kJ/mol
#    -analytic -23.55781E-1 00E+0 32.78392E+2 00E+0 00E+0
Co(OH)2(s)
    Co(OH)2 = Co+2 - 2H+ + 2H2O
    log_k 13.11
Ni(OH)2(s)
    Ni(OH)2 = Ni+2 - 2H+ + 2H2O
    log_k 13.30
    delta_h -84.389 #kJ/mol
Mn(OH)2(s)
    Mn(OH)2 = Mn+2 - 2H+ + 2H2O
    log_k 15.28
    -analytic 15.3E+0 00E+0 00E+0 00E+0 00E+0

SOLUTION_SPECIES

# ============================
# Neutral sulfate ion pairs (M2+ + SO4-2 = MSO4^0)
# ============================

Al+3 + SO4-2 = AlSO4+
    log_k 3.20   # NOTE: NOT neutral (charge +1), included for completeness
Fe+3 + SO4-2 = FeSO4+
    log_k 4.00   # NOTE: NOT neutral (charge +1), included for completeness
Fe+2 + SO4-2 = FeSO4
    log_k 2.30
Ni+2 + SO4-2 = NiSO4
    log_k 2.20
Mn+2 + SO4-2 = MnSO4
    log_k 2.00

KNOBS
    -iterations 6000
    -convergence_tolerance 1e-7
    -tolerance             1e-11
    -logfile true
"""

# ===============================================================
# GENERATE INPUT BLOCKS
# ===============================================================
with open(output_file, "w") as f:
    f.write(header)

    for idx, row in samples.iloc[0:].iterrows():
        sol_id = idx + 1
        #scale = 0.01 * (row.V_batt_pct / 5.0)  # metal sulfate scaling

        f.write(f"\n# ===============================================================\n")
        f.write(f"TITLE PART A. -- Reacting DI Water with metal sulfates \n")
        f.write(f"# === Sample {sol_id}: T={row.temp:.1f}°C, NaOH_mole={row.NaOH_mole:.2g}, "
                f"Leachate={row.BattLeachate_volume:.5f}%, NaOH_stock={row.NaOH_stock:.2f} M ===\n\n")

        # ---- Stage 1: Hydrolysis of metal sulfates ----
        f.write("# ==== 1. Mix 1: Start with DI water ====\n")
        f.write(f"SOLUTION {sol_id} DI_water\n")
        f.write(f"    temp {row.temp:.1f}\n")
        f.write(f"    pH 7 charge\n")
        f.write(f"    water 1.0\n")
        #f.write(f"    units mol/L\n\n")
        f.write(f"    units mol/kgw\n\n")

        f.write("# ==== 2. Dissolve solid metal sulfates ====\n")
        f.write("# ==== phase name [ saturation index [( alternative formula or phase )] [ amount [( dis or pre ) ]]] ====\n")
        f.write(f"REACTION {sol_id}\n")
        f.write(f"    Al2(SO4)3       {row.Al_sulfate_mol:.7e}\n") # The numbers (e.g., 0.07185) are in moles of solid reacted - Moles of the phase in the phase assemblage or moles of the alternative reaction.
        f.write(f"    FeSO4:7H2O      {row.Fe_sulfate_mol:.7e}\n")
        f.write(f"    CuSO4           {row.Cu_sulfate_mol:.7e}\n")
        f.write(f"    CoSO4:7H2O      {row.Co_sulfate_mol:.7e}\n")
        f.write(f"    NiSO4:6H2O      {row.Ni_sulfate_mol:.7e}\n")
        f.write(f"    MnSO4:H2O       {row.Mn_sulfate_mol:.7e}\n")
        f.write("    1.0\n\n")

        f.write("# ==== 3. Output the hydrolyzed solution ====\n")
        f.write(f"SELECTED_OUTPUT {sol_id}\n")
        f.write(f"   -file speciation_results/hydrolyzed_solution_{sol_id}.csv\n")
        f.write("    -reset true\n")
        f.write("    -temperature true\n")
        f.write("    -pH true\n")
        f.write("    -pe true\n")
        f.write("    -ionic_strength true\n")
        f.write("    -charge_balance true\n")
        f.write("    -alkalinity true\n")
        f.write("    -molalities true\n")
        f.write("    -totals Al Fe Cu Co Ni Mn Na S\n")
        f.write("    -activities H+ OH- SO4-2 Na+ Al+3 Fe+3 Cu+2 Co+2 Ni+2 Mn+2\n")
        f.write("    -equilibrium_phases FeAl2O4(s) Gibbsite Fe(OH)3(s) Cu(OH)2(s) Co(OH)2(s) Ni(OH)2(s) Mn(OH)2(s)\n")
        f.write("    -si FeAl2O4(s) Gibbsite Fe(OH)3(s) Cu(OH)2(s) Co(OH)2(s) Ni(OH)2(s) Mn(OH)2(s)\n")
        f.write("USER_PUNCH\n")
        f.write(
            '    -headings '
            '"Solution_ID" '
            'a_Al3+ m_Al+3 a_Fe2+ m_Fe+2 a_Fe3+ m_Fe+3 a_Cu+ m_Cu+ a_Cu2+ m_Cu+2 '            
            'a_Ni2+ m_Ni+2 a_Co2+ m_Co+2 a_Co3+ m_Co+3 a_Mn2+ m_Mn+2 a_Mn3+ m_Mn+3 a_Li+ m_Li+ a_Na+ m_Na+ '
            'm_AlOH2+ m_Al(OH)2+ m_AlSO4+ m_AlO2- m_Al(SO4)2- m_AlSO4+ '
            'm_FeOH+ m_Fe(OH)2 m_Fe(OH)3- m_Fe(OH)4- m_FeSO4+ m_FeSO4 '
            'm_CuOH+ m_CuSO4 '
            'm_CoOH+ m_CoSO4 m_Co(OH)4-2 '
            'm_NiOH+ m_Ni(OH)2 m_NiSO4 m_Ni(OH)3- '
            'm_MnOH+ m_Mn(OH)2 m_Mn(OH)3- m_Mn(OH)4-2 '
            'm_Al2(OH)2+4 m_Al3(OH)4+5 m_Al13O4(OH)24+7 '
            'm_Fe2(OH)2+4 m_Fe3(OH)4+5 '
            'm_Ni4(OH)4+4 '
            'm_Mn2(OH)3+ '
            'm_MnO4- m_MnO4-2 '
            'm_H2 m_O2 '
            'm_SO4-2 m_HSO4- m_S2O3-2 m_S2O4-2 m_S2O6-2 '
            'm_NaSO4- m_MnSO4 m_NiSO4 m_CoSO4 m_CuSO4 '
            'm_H2O m_H+ m_OH- '
            '"Description"\n'
        ) 
        f.write(
        f"    10 PUNCH {sol_id}, "
        "ACT('Al+3'), MOL('Al+3'), ACT('Fe+2'), MOL('Fe+2'), ACT('Fe+3'), MOL('Fe+3'), ACT('Cu+'), MOL('Cu+'), ACT('Cu+2'), MOL('Cu+2'), "
        "ACT('Ni+2'), MOL('Ni+2'), ACT('Co+2'), MOL('Co+2'), ACT('Co+3'), MOL('Co+3'), ACT('Mn+2'), MOL('Mn+2'), ACT('Mn+3'), MOL('Mn+3'), "
        "ACT('Li+'), MOL('Li+'), ACT('Na+'), MOL('Na+'),  "
        "MOL('AlOH+2'), MOL('Al(OH)2+'), MOL('AlSO4+'), MOL('AlO2-'), MOL('Al(SO4)2-'), MOL('AlSO4+'), "
        "MOL('FeOH+'), MOL('Fe(OH)2'), MOL('Fe(OH)3-'), MOL('Fe(OH)4-'), MOL('FeSO4+'), MOL('FeSO4'), "
        "MOL('CuOH+'), MOL('CuSO4')\n"
        )
        f.write(
        "    20 PUNCH "
        "MOL('CoOH+'), MOL('CoSO4'), MOL('Co(OH)4-2'), "
        "MOL('NiOH+'), MOL('Ni(OH)2'), MOL('NiSO4'), MOL('Ni(OH)3-'), "
        "MOL('MnOH+'), MOL('Mn(OH)2'), MOL('Mn(OH)3-'), MOL('Mn(OH)4-2'), "
        "MOL('Al2(OH)2+4'), MOL('Al3(OH)4+5'), MOL('Al13O4(OH)24+7')\n"
        )

        f.write(
        "    30 PUNCH "
        "MOL('Fe2(OH)2+4'), MOL('Fe3(OH)4+5'), MOL('Ni4(OH)4+4'), MOL('Mn2(OH)3+'), "
        "MOL('MnO4-'), MOL('MnO4-2') "
        "MOL('H2'), MOL('O2'), "
        "MOL('SO4-2'), MOL('HSO4-'), MOL('S2O3-2'), MOL('S2O4-2'), MOL('S2O6-2')\n"
        )

        f.write(
        "    40 PUNCH "
        "MOL('NaSO4-'), MOL('MnSO4'), MOL('NiSO4'), MOL('CoSO4'), MOL('CuSO4'), "
        "MOL('H2O'), MOL('H+'), MOL('OH-'), \"Hydrolyzed_metal_sulfate_solution\"\n\n"
        )

        #f.write("\nEND\n\n")
        f.write(f"SAVE solution {sol_id}\nEND\n\n")
        
        # ---- Stage 2: NaOH neutralization ----
        f.write(f"TITLE PART B.1 -- Reacting hydrolized metal sulfates with NaOH solution - No Precipitation\n")
        f.write(f"USE solution {sol_id}\n")
        f.write("INCREMENTAL_REACTIONS false \n")
        f.write(f"REACTION {sol_id+100000} #NaOH_add\n")
        f.write(f"    Na+ {row.NaOH_stock:.2f}\n")
        f.write(f"    OH- {row.NaOH_stock:.2f}\n")
        f.write(f"    {row.NaOH_mole:.5e}\n")
        #f.write("    " + " ".join(f"{x:.4g}" for x in np.linspace(0, np.max(row.NaOH_mole), n_per_dim)) + " moles \n\n")
        #f.write("    " + " ".join(f"{x:.4g}" for x in np.linspace(0, samples["NaOH_mole"].max(), n_per_dim)) + " moles\n\n")
        #f.write(
        #    "    " + " ".join(f"{x:.8g}" for x in samples["NaOH_mole"].to_numpy()) + " moles\n\n"
        #)
        
        #f.write(f"SAVE solution {sol_id+20000}\n") #
        #f.write(f"END\n") #
        f.write("# ==== 7. Output neutralized solution and SI ====\n")
        f.write(f"SELECTED_OUTPUT {sol_id+100000}\n")
        f.write(f"   -file speciation_results/NaOH_mixed_solution_{sol_id}.csv\n")
        f.write("    -reset true\n")
        f.write("    -percent_error true\n")   
        f.write("    -temperature true\n")
        f.write("    -pH true\n")
        f.write("    -pe true\n")
        f.write("    -ionic_strength true\n")
        f.write("    -charge_balance true\n")
        f.write("    -alkalinity true\n")   
        f.write("    -molalities true\n")
        f.write("    -totals Al Fe Cu Co Ni Mn Na S \n")
        f.write("    -activities H+ OH- SO4-2 Na+ Al+3 Fe+3 Cu+2 Co+2 Ni+2 Mn+2\n")
        f.write("    -equilibrium_phases FeAl2O4(s) Gibbsite Fe(OH)3(s) Cu(OH)2(s) Co(OH)2(s) Ni(OH)2(s) Mn(OH)2(s)\n")
        f.write("    -si FeAl2O4(s) Gibbsite Fe(OH)3(s) Cu(OH)2(s) Co(OH)2(s) Ni(OH)2(s) Mn(OH)2(s)\n")
        f.write("USER_PUNCH\n")
        f.write(
            '    -headings '
            '"Solution_ID" '
            'a_Al3+ m_Al+3 a_Fe2+ m_Fe+2 a_Fe3+ m_Fe+3 a_Cu+ m_Cu+ a_Cu2+ m_Cu+2 '            
            'a_Ni2+ m_Ni+2 a_Co2+ m_Co+2 a_Co3+ m_Co+3 a_Mn2+ m_Mn+2 a_Mn3+ m_Mn+3 a_Li+ m_Li+ a_Na+ m_Na+ '
            'm_AlOH2+ m_Al(OH)2+ m_AlSO4+ m_AlO2- m_Al(SO4)2- m_AlSO4+ '
            'm_FeOH+ m_Fe(OH)2 m_Fe(OH)3- m_Fe(OH)4- m_FeSO4+ m_FeSO4 '
            'm_CuOH+ m_CuSO4 '
            'm_CoOH+ m_CoSO4 m_Co(OH)4-2 '
            'm_NiOH+ m_Ni(OH)2 m_NiSO4 m_Ni(OH)3- '
            'm_MnOH+ m_Mn(OH)2 m_Mn(OH)3- m_Mn(OH)4-2 '
            'm_Al2(OH)2+4 m_Al3(OH)4+5 m_Al13O4(OH)24+7 '
            'm_Fe2(OH)2+4 m_Fe3(OH)4+5 '
            'm_Ni4(OH)4+4 '
            'm_Mn2(OH)3+ '
            'm_MnO4- m_MnO4-2 '
            'm_H2 m_O2 '
            'm_SO4-2 m_HSO4- m_S2O3-2 m_S2O4-2 m_S2O6-2 '
            'm_NaSO4- m_MnSO4 m_NiSO4 m_CoSO4 m_CuSO4 '
            'm_H2O m_H+ m_OH- '
            '"Description"\n'
        )

        f.write(
            f"    10 PUNCH "
            f"{sol_id}, "
            "ACT('Al+3'), MOL('Al+3'), ACT('Fe+2'), MOL('Fe+2'), ACT('Fe+3'), MOL('Fe+3'), ACT('Cu+'), MOL('Cu+'), ACT('Cu+2'), MOL('Cu+2'), "
            "ACT('Ni+2'), MOL('Ni+2'), ACT('Co+2'), MOL('Co+2'), ACT('Co+3'), MOL('Co+3'), ACT('Mn+2'), MOL('Mn+2'), ACT('Mn+3'), MOL('Mn+3'), "
            "ACT('Li+'), MOL('Li+'), ACT('Na+'), MOL('Na+'),  "
            "MOL('AlOH+2'), MOL('Al(OH)2+'), MOL('AlSO4+'), MOL('AlO2-'), MOL('Al(SO4)2-'), MOL('AlSO4+'), "
            "MOL('FeOH+'), MOL('Fe(OH)2'), MOL('Fe(OH)3-'), MOL('Fe(OH)4-'), MOL('FeSO4+'), MOL('FeSO4'), "
            "MOL('CuOH+'), MOL('CuSO4'), "
            "MOL('CoOH+'), MOL('CoSO4'), MOL('Co(OH)4-2'), "
            "MOL('NiOH+'), MOL('Ni(OH)2'), MOL('NiSO4'), MOL('Ni(OH)3-'), "
            "MOL('MnOH+'), MOL('Mn(OH)2'), MOL('Mn(OH)3-'), MOL('Mn(OH)4-2'), "
            "MOL('Al2(OH)2+4'), MOL('Al3(OH)4+5'), MOL('Al13O4(OH)24+7'), "
            "MOL('Fe2(OH)2+4'), MOL('Fe3(OH)4+5'), "
            "MOL('Ni4(OH)4+4'), "
            "MOL('Mn2(OH)3+'), "
            "MOL('MnO4-'), MOL('MnO4-2')"
            "MOL('H2'), MOL('O2'), "
            "MOL('SO4-2'), MOL('HSO4-'), MOL('S2O3-2'), MOL('S2O4-2'), MOL('S2O6-2'), "
            "MOL('NaSO4-'), MOL('MnSO4'), MOL('NiSO4'), MOL('CoSO4'), MOL('CuSO4'), "
            "MOL('H2O'), MOL('H+'), MOL('OH-'), \"NaOH_mixed_solution\" \nEND\n\n"
        )

        f.write(f"TITLE PART B.2 -- Reacting hydrolized metal sulfates with NaOH solution - With Precipitation \n")
        f.write(f"USE solution {sol_id}\n")
        f.write("INCREMENTAL_REACTIONS false \n")
        f.write(f"REACTION {sol_id+20000} #NaOH_add\n")
        f.write(f"    Na+ {row.NaOH_stock:.2f}\n")
        f.write(f"    OH- {row.NaOH_stock:.2f}\n")
        f.write(f"    {row.NaOH_mole:.5e}\n\n")
#        f.write("    " + " ".join(f"{x:.8g}" for x in np.linspace(0, samples["NaOH_mole"].max(), n_per_dim)) + " moles\n\n")
        #f.write(
        #    "    " + " ".join(f"{x:.8g}" for x in samples["NaOH_mole"].to_numpy()) + " moles\n\n"
        #)

        f.write("# ==== 6. Check for metal hydroxide formation ====\n")
        f.write("# ==== 7. Output neutralized solution and SI ====\n")
        f.write(f"EQUILIBRIUM_PHASES {sol_id+20000}\n")
        f.write("    Gibbsite  0.0 0\n")
        #f.write("    Al(OH)3(am)  0.0 1e-12\n")
        f.write("    Fe(OH)3(s)   0.0 0\n")
        f.write("    Cu(OH)2(s)   0.0 0\n")
        f.write("    Co(OH)2(s)   0.0 0\n")
        f.write("    Ni(OH)2(s)   0.0 0\n")
        f.write("    Mn(OH)2(s)   0.0 0\n\n")
        #f.write("    FeAl2O4(s)   0.0 0\n")
        #f.write("   -force_equality\n")
        
        #f.write(f"SAVE solution {sol_id+20000}\n") #
        #f.write(f"END\n") #
        f.write("# ==== 7. Output neutralized solution and SI ====\n")
        f.write(f"SELECTED_OUTPUT {sol_id+20000}\n")
        f.write(f"   -file speciation_results/NaOH_mixed_solution_{sol_id}_eq.csv\n")
        f.write("    -reset true\n")
        f.write("    -percent_error true\n")   
        f.write("    -temperature true\n")
        f.write("    -pH true\n")
        f.write("    -pe true\n")
        f.write("    -ionic_strength true\n")
        f.write("    -charge_balance true\n")
        f.write("    -alkalinity true\n")   
        f.write("    -molalities true\n")
        f.write("    -totals Al Fe Cu Co Ni Mn Na S \n")
        f.write("    -activities H+ OH- SO4-2 Na+ Al+3 Fe+3 Cu+2 Co+2 Ni+2 Mn+2\n")
        f.write("    -equilibrium_phases FeAl2O4(s) Gibbsite Fe(OH)3(s) Cu(OH)2(s) \n")
        f.write("    -si FeAl2O4(s) Gibbsite Fe(OH)3(s) Cu(OH)2(s) Co(OH)2(s) Ni(OH)2(s) Mn(OH)2(s)\n")
        f.write("USER_PUNCH\n")
        f.write(
            '    -headings '
            '"Solution_ID" '
            'a_Al3+ m_Al+3 a_Fe2+ m_Fe+2 a_Fe3+ m_Fe+3 a_Cu+ m_Cu+ a_Cu2+ m_Cu+2 '            
            'a_Ni2+ m_Ni+2 a_Co2+ m_Co+2 a_Co3+ m_Co+3 a_Mn2+ m_Mn+2 a_Mn3+ m_Mn+3 a_Li+ m_Li+ a_Na+ m_Na+ '
            'm_AlOH2+ m_Al(OH)2+ m_AlSO4+ m_AlO2- m_Al(SO4)2- m_AlSO4+ '
            'm_FeOH+ m_Fe(OH)2 m_Fe(OH)3- m_Fe(OH)4- m_FeSO4+ m_FeSO4 '
            'm_CuOH+ m_CuSO4 '
            'm_CoOH+ m_CoSO4 m_Co(OH)4-2 '
            'm_NiOH+ m_Ni(OH)2 m_NiSO4 m_Ni(OH)3- '
            'm_MnOH+ m_Mn(OH)2 m_Mn(OH)3- m_Mn(OH)4-2 '
            'm_Al2(OH)2+4 m_Al3(OH)4+5 m_Al13O4(OH)24+7 '
            'm_Fe2(OH)2+4 m_Fe3(OH)4+5 '
            'm_Ni4(OH)4+4 '
            'm_Mn2(OH)3+ '
            'm_MnO4- m_MnO4-2 '
            'm_H2 m_O2 '
            'm_SO4-2 m_HSO4- m_S2O3-2 m_S2O4-2 m_S2O6-2 '
            'm_NaSO4- m_MnSO4 m_NiSO4 m_CoSO4 m_CuSO4 '
            'm_H2O m_H+ m_OH- '
            '"Description"\n'
        )

        f.write(
            f"    10 PUNCH "
            f"{sol_id}, "
            "ACT('Al+3'), MOL('Al+3'), ACT('Fe+2'), MOL('Fe+2'), ACT('Fe+3'), MOL('Fe+3'), ACT('Cu+'), MOL('Cu+'), ACT('Cu+2'), MOL('Cu+2'), "
            "ACT('Ni+2'), MOL('Ni+2'), ACT('Co+2'), MOL('Co+2'), ACT('Co+3'), MOL('Co+3'), ACT('Mn+2'), MOL('Mn+2'), ACT('Mn+3'), MOL('Mn+3'), "
            "ACT('Li+'), MOL('Li+'), ACT('Na+'), MOL('Na+'),  "
            "MOL('AlOH+2'), MOL('Al(OH)2+'), MOL('AlSO4+'), MOL('AlO2-'), MOL('Al(SO4)2-'), MOL('AlSO4+'), "
            "MOL('FeOH+'), MOL('Fe(OH)2'), MOL('Fe(OH)3-'), MOL('Fe(OH)4-'), MOL('FeSO4+'), MOL('FeSO4'), "
            "MOL('CuOH+'), MOL('CuSO4'), "
            "MOL('CoOH+'), MOL('CoSO4'), MOL('Co(OH)4-2'), "
            "MOL('NiOH+'), MOL('Ni(OH)2'), MOL('NiSO4'), MOL('Ni(OH)3-'), "
            "MOL('MnOH+'), MOL('Mn(OH)2'), MOL('Mn(OH)3-'), MOL('Mn(OH)4-2'), "
            "MOL('Al2(OH)2+4'), MOL('Al3(OH)4+5'), MOL('Al13O4(OH)24+7'), "
            "MOL('Fe2(OH)2+4'), MOL('Fe3(OH)4+5'), "
            "MOL('Ni4(OH)4+4'), "
            "MOL('Mn2(OH)3+'), "
            "MOL('MnO4-'), MOL('MnO4-2'), "
            "MOL('H2'), MOL('O2'), "
            "MOL('SO4-2'), MOL('HSO4-'), MOL('S2O3-2'), MOL('S2O4-2'), MOL('S2O6-2'), "
            "MOL('NaSO4-'), MOL('MnSO4'), MOL('NiSO4'), MOL('CoSO4'), MOL('CuSO4'), "
            "MOL('H2O'), MOL('H+'), MOL('OH-'), \"NaOH_mixed_solution_EQ\" \nEND\n\n"
        )

    #f.write("\n# ==== Run all cells ====\n")
    #f.write("RUN_CELLS\n  -cells " + " ".join(str(i + 1) for i in range(n_samples)) + "\n")

plt.close('all')
print(f"OK: Generated {samples.shape} LHS-based hydrolysis + NaOH mixing simulations.")
print(f" PHREEQC file: {output_file}")
print(" Parameter samples saved to: lhs_input/hydrolysis_then_NaOHmix_samples.csv")