"""
Gaussian copula sampling vs. Latin Hypercube Sampling (LHS) comparison.

The Gaussian copula separates marginal distributions from inter-variable
dependence. Here, marginals are fit empirically from the LHS data (log-scale),
and the correlation matrix encodes physically motivated relationships between
metals derived from battery cathode stoichiometry.

Usage (run from repo root):
    python src/plot_copula_vs_lhs.py

Outputs → results/copula_vs_lhs/
    fig_copula_vs_lhs_kde.png        — marginal KDE comparison
    fig_copula_vs_lhs_scatter.png    — pairwise scatter (key pairs)
    fig_copula_vs_lhs_corr.png       — correlation matrices side by side
    copula_samples.csv               — generated copula samples
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
UP_DIR = Path(os.environ.get("UP_DIR", "paper_SIT/Leach_recovery_UP3"))
CSV    = UP_DIR / "lhs_input" / "hydrolysis_then_NaOHmix_samples.csv"
OUTDIR = Path("results") / "copula_vs_lhs"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 5001   # match LHS ensemble size
RNG       = np.random.default_rng(42)

# ── metal columns of interest ──────────────────────────────────────────────────
METALS = ["Fe_sulfate_mol", "Al_sulfate_mol", "Cu_sulfate_mol",
          "Ni_sulfate_mol", "Co_sulfate_mol", "Mn_sulfate_mol", "NaOH_mole"]

LABELS = {
    "Fe_sulfate_mol": "Fe (FeSO₄·7H₂O)",
    "Al_sulfate_mol": "Al (Al₂(SO₄)₃)",
    "Cu_sulfate_mol": "Cu (CuSO₄)",
    "Ni_sulfate_mol": "Ni (NiSO₄·6H₂O)",
    "Co_sulfate_mol": "Co (CoSO₄·7H₂O)",
    "Mn_sulfate_mol": "Mn (MnSO₄·H₂O)",
    "NaOH_mole":      "NaOH dose",
}

# Okabe-Ito colors (same order as recovery plots)
COLORS = {
    "Fe_sulfate_mol": "#E69F00",
    "Al_sulfate_mol": "#56B4E9",
    "Cu_sulfate_mol": "#009E73",
    "Ni_sulfate_mol": "#F0E442",
    "Co_sulfate_mol": "#0072B2",
    "Mn_sulfate_mol": "#D55E00",
    "NaOH_mole":      "#CC79A7",
}

# ── load LHS data ──────────────────────────────────────────────────────────────
lhs = pd.read_csv(CSV)[METALS].dropna()
print(f"LHS samples loaded: {len(lhs)}")

# work in log10 space throughout
lhs_log = np.log10(lhs.values)   # shape (n, 7)

# ── Step 1: fit empirical marginals (kernel density on log scale) ──────────────
# We use the empirical CDF (via rankdata) to convert LHS → uniform → normal.
# This is the non-parametric copula approach: no distributional assumption needed.

def to_uniform(x):
    """Empirical CDF: map sorted ranks to (0,1) avoiding boundaries."""
    n = len(x)
    ranks = stats.rankdata(x)           # 1-based ranks
    return ranks / (n + 1)             # Hazen plotting position

def to_normal(u):
    """Probit transform: uniform → standard normal."""
    return stats.norm.ppf(u)

# transform each column: log-values → uniform → normal scores
normal_scores = np.column_stack([
    to_normal(to_uniform(lhs_log[:, j]))
    for j in range(lhs_log.shape[1])
])

# ── Step 2: estimate empirical correlation matrix from LHS ────────────────────
R_empirical = np.corrcoef(normal_scores, rowvar=False)

# ── Step 3: define physically motivated correlation matrix ────────────────────
# Battery-chemistry rationale:
#   Ni–Co, Ni–Mn, Co–Mn  : stoichiometrically coupled in NMC cathodes (+)
#   Ni–Al                 : NCA contains both Al and Ni              (+, weaker)
#   Fe–Cu                 : both hardware/current-collector impurities(+, weak)
#   NaOH – total metals   : dose scales with hydroxide demand         (+, moderate)
#   All other pairs       : near-independent                          (~0)

cols = METALS
idx  = {c: i for i, c in enumerate(cols)}

R_physical = np.eye(len(cols))

def set_corr(R, a, b, r):
    i, j = idx[a], idx[b]
    R[i, j] = R[j, i] = r

# cathode stoichiometry block (Ni, Co, Mn)
set_corr(R_physical, "Ni_sulfate_mol", "Co_sulfate_mol",  0.75)
set_corr(R_physical, "Ni_sulfate_mol", "Mn_sulfate_mol",  0.70)
set_corr(R_physical, "Co_sulfate_mol", "Mn_sulfate_mol",  0.65)

# NCA: Ni–Al coupling
set_corr(R_physical, "Ni_sulfate_mol", "Al_sulfate_mol",  0.35)
set_corr(R_physical, "Co_sulfate_mol", "Al_sulfate_mol",  0.20)

# hardware impurities
set_corr(R_physical, "Fe_sulfate_mol", "Cu_sulfate_mol",  0.30)

# NaOH scales with total metal load (Ni and Co dominate)
set_corr(R_physical, "NaOH_mole", "Ni_sulfate_mol",  0.55)
set_corr(R_physical, "NaOH_mole", "Co_sulfate_mol",  0.45)
set_corr(R_physical, "NaOH_mole", "Mn_sulfate_mol",  0.40)
set_corr(R_physical, "NaOH_mole", "Al_sulfate_mol",  0.25)
set_corr(R_physical, "NaOH_mole", "Fe_sulfate_mol",  0.15)
set_corr(R_physical, "NaOH_mole", "Cu_sulfate_mol",  0.10)

# ensure positive semi-definiteness (nearest PSD via eigenvalue clipping)
eigvals, eigvecs = np.linalg.eigh(R_physical)
eigvals = np.clip(eigvals, 1e-6, None)
R_physical = eigvecs @ np.diag(eigvals) @ eigvecs.T
# re-normalise diagonal to 1
d = np.sqrt(np.diag(R_physical))
R_physical = R_physical / np.outer(d, d)

# ── Step 4: Gaussian copula sampling ─────────────────────────────────────────
# Draw from multivariate normal with R_physical, then map back through
# the empirical marginals of the LHS data.

Z = RNG.multivariate_normal(mean=np.zeros(len(cols)),
                             cov=R_physical,
                             size=N_SAMPLES)          # (N, 7) normal scores

U = stats.norm.cdf(Z)   # → uniform [0,1] via Gaussian copula

# invert through empirical quantile of each LHS marginal
copula_log = np.column_stack([
    np.quantile(lhs_log[:, j], U[:, j])
    for j in range(len(cols))
])

copula_samples = pd.DataFrame(10**copula_log, columns=cols)
copula_samples.to_csv(OUTDIR / "copula_samples.csv", index=False)
print(f"Copula samples saved: {len(copula_samples)}")

# ── plotting helpers ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      12,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ── Figure 1: marginal KDE comparison ────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=False)
axes = axes.flatten()

for ax, col in zip(axes, METALS):
    lhs_vals  = np.log10(lhs[col].values)
    cop_vals  = np.log10(copula_samples[col].values)
    color = COLORS[col]

    sns.kdeplot(lhs_vals,  ax=ax, color=color,  linestyle="-",  lw=2.0,
                label="LHS", fill=True, alpha=0.15)
    sns.kdeplot(cop_vals,  ax=ax, color="k",    linestyle="--", lw=1.8,
                label="Copula", fill=False)

    ax.set_title(LABELS[col], fontsize=11)
    ax.set_xlabel("log₁₀(mol)", fontsize=10)
    ax.set_ylabel("Density",    fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.minorticks_on()

# hide unused subplot
axes[-1].set_visible(False)

fig.suptitle("Marginal distributions: LHS vs. Gaussian copula sampling",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUTDIR / "fig_copula_vs_lhs_kde.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig_copula_vs_lhs_kde.png")

# ── Figure 2: pairwise scatter for key pairs ──────────────────────────────────
PAIRS = [
    ("Ni_sulfate_mol", "Co_sulfate_mol"),
    ("Ni_sulfate_mol", "Mn_sulfate_mol"),
    ("Co_sulfate_mol", "Mn_sulfate_mol"),
    ("Fe_sulfate_mol", "Cu_sulfate_mol"),
    ("Ni_sulfate_mol", "Al_sulfate_mol"),
    ("NaOH_mole",      "Ni_sulfate_mol"),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()

for ax, (cx, cy) in zip(axes, PAIRS):
    lx_lhs = np.log10(lhs[cx].values)
    ly_lhs = np.log10(lhs[cy].values)
    lx_cop = np.log10(copula_samples[cx].values)
    ly_cop = np.log10(copula_samples[cy].values)

    r_lhs = np.corrcoef(lx_lhs, ly_lhs)[0, 1]
    r_cop = np.corrcoef(lx_cop, ly_cop)[0, 1]

    ax.scatter(lx_lhs, ly_lhs, s=4, alpha=0.25, color="#999999", label=f"LHS  (r={r_lhs:.2f})")
    ax.scatter(lx_cop, ly_cop, s=4, alpha=0.25, color="#0072B2", label=f"Copula (r={r_cop:.2f})")

    ax.set_xlabel(f"log₁₀ {LABELS[cx]}", fontsize=9)
    ax.set_ylabel(f"log₁₀ {LABELS[cy]}", fontsize=9)
    ax.legend(fontsize=8, frameon=True, markerscale=3)
    ax.minorticks_on()

fig.suptitle("Pairwise scatter: LHS (grey) vs. Gaussian copula (blue)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTDIR / "fig_copula_vs_lhs_scatter.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig_copula_vs_lhs_scatter.png")

# ── Figure 3: correlation matrices side by side ───────────────────────────────
R_lhs    = np.corrcoef(np.log10(lhs[METALS].values), rowvar=False)
R_cop    = np.corrcoef(np.log10(copula_samples[METALS].values), rowvar=False)
tick_labels = [LABELS[c].split(" ")[0] for c in METALS]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
titles = ["LHS (empirical)", "Copula (physical prior)", "Difference (Copula − LHS)"]
matrices = [R_lhs, R_cop, R_cop - R_lhs]
cmaps    = ["RdBu_r", "RdBu_r", "PuOr_r"]
vlims    = [(-1, 1), (-1, 1), (-0.5, 0.5)]

for ax, mat, title, cmap, (vmin, vmax) in zip(axes, matrices, titles, cmaps, vlims):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(METALS)))
    ax.set_yticks(range(len(METALS)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # annotate cells
    for i in range(len(METALS)):
        for j in range(len(METALS)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="k" if abs(mat[i,j]) < 0.6 else "w")

fig.suptitle("Pearson correlation in log₁₀ space", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTDIR / "fig_copula_vs_lhs_corr.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig_copula_vs_lhs_corr.png")

print(f"\nAll outputs in {OUTDIR}/")
print("\nPhysically motivated R matrix (Copula prior):")
df_R = pd.DataFrame(R_physical, index=tick_labels, columns=tick_labels)
print(df_R.round(2).to_string())
