#!/usr/bin/env python3
"""
dispersion_alpha_beta_fixed.py

Sweep over α∈[−0.025,0.025], β∈[0,0.25] for D_A/D_B=0.1,
compute max Re[λ] for k>0, and plot a sensible heatmap.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- FIXED PARAMETERS ----
Db = 1.0
Da = 0.01

# grid resolution
alpha_vals = np.linspace(-0.025, 0.025, 1000)
beta_vals  = np.linspace(-0.25, 0.25, 1000)

# wavenumber grid (complex)
k = np.linspace(0, 15, 750).astype(np.complex128)

# ---- FUNCTIONS ----
def homogeneous_solution(a, b):
    A0 = a + b
    B0 = b/(A0*A0)
    return A0, B0

def dispersion_branches(kc, a, b, Da, Db):
    A0, B0 = homogeneous_solution(a, b)
    fa, fb = 2*A0*B0 - 1, A0*A0
    ga, gb = -2*A0*B0,     -A0*A0
    tr, det = fa+gb, fa*gb - fb*ga
    k2 = kc*kc
    bb = k2*(Da+Db) - tr
    cc = det + k2*(k2*Da*Db - gb*Da - fa*Db)
    disc = bb*bb - 4*cc
    sqrt_disc = np.sqrt(disc)
    λp = (-bb + sqrt_disc)/2
    λm = (-bb - sqrt_disc)/2
    return λp, λm

# ---- BUILD THE MAP ----
map_growth = np.full((1000, 1000), np.nan)

for i, a in enumerate(alpha_vals):
    for j, b in enumerate(beta_vals):
        if (abs(a+b) < 1e-5):
            continue
        λp, λm = dispersion_branches(k, a, b, Da, Db)
        k_max_idx = np.argmax(λp.real)
        k_max = k[k_max_idx]
        rp = λp.real[k_max_idx]
        if (rp > 1e-6):
            map_growth[j,i] = k_max.real

# ---- COLOR SCALE CLAMPING ----
# Use the 2nd and 98th percentiles to cut off outliers
valid = map_growth[~np.isnan(map_growth)]
vmin, vmax = np.percentile(valid, [2,98])

# ---- PLOTTING ----
fig, ax = plt.subplots(figsize=(8,6))

im = ax.imshow(
    map_growth,
    origin='lower',
    extent=[alpha_vals[0], alpha_vals[-1],
            beta_vals[0],  beta_vals[-1]],
    aspect='auto',
    cmap='Reds',
    vmin=vmin, vmax=vmax
)

# colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"Critical $k_{max}$", fontsize=12)

# labels & title
ax.set_title(rf"Critical $k_{{max}}$ for $D_A/D_B = {Da/Db:.2f}$")
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
ax.grid(True, ls='--', lw=0.5, alpha=0.6)

plt.tight_layout()
plt.savefig("critical_k.png", dpi=600)
plt.show()
