#!/usr/bin/env python3
"""
plot_std_sweep_1x3.py

Reads CSVs from three sweep regimes and plots σ_A vs time in a 1×3 grid.
Each subplot’s legend is arranged row-first in a 2×3 layout, boxed, with LaTeX titles.
"""

import glob, os, re
import math
import pandas as pd
import matplotlib.pyplot as plt

# --- GLOBAL FIXED VALUES (when that param is NOT swept) ---
alpha_const = 0.05
beta_const  = 0.80
da_db_const = 0.05

# LaTeX legend‐title for each regime
LEGEND_TITLES = {
    "alpha":  r"$\alpha$",
    "beta":   r"$\beta$",
    "diff":   r"$D_A / D_B$",
}

# Regimes in left→right order
REGIMES = [
    dict(name="alpha", title=r"(a) Sweep $\alpha$"),
    dict(name="beta",  title=r"(b) Sweep $\beta$"),
    dict(name="diff",  title=r"(c) Sweep $D_A / D_B$"),
]

fig, axes = plt.subplots(1, 3, figsize=(13,5), sharey=True)

for ax, regime in zip(axes, REGIMES):
    param = regime['name']
    ax.set_title(regime['title'])
    ax.set_xlabel("Time")
    if ax is axes[0]:
        ax.set_ylabel(r"$\sigma_A$ (Activator Std. Dev.)")
    ax.grid(True)

    # gather CSVs and plot
    pattern = f"std_sweep_{param}_*.csv"
    files   = sorted(glob.glob(pattern))
    if len(files) != 6:
        raise RuntimeError(f"Expected 6 CSVs for '{param}', found {len(files)}")
    regex = re.compile(rf"std_sweep_{param}_([\d\.]+)\.csv")

    for fn in files:
        m = regex.search(os.path.basename(fn))
        if not m: continue
        val = float(m.group(1))
        df  = pd.read_csv(fn)
        ax.plot(df['time_step'], df['sigma_A'], label=f"{val:.2f}")

    # grab the automatically created handles+labels
    handles, labels = ax.get_legend_handles_labels()

    # reorder for row-first fill in a 2×3 grid
    ncol = 3
    nrow = math.ceil(len(labels)/ncol)
    h2, l2 = [], []
    for row in range(nrow):
        for col in range(ncol):
            idx = row*ncol + col
            if idx < len(labels):
                h2.append(handles[idx])
                l2.append(labels[idx])

    # draw the boxed legend beneath, row-first
    leg = ax.legend(
        h2, l2,
        ncol=ncol,
        title=LEGEND_TITLES[param],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize="small",
        title_fontsize="medium",
        frameon=True,
        edgecolor="black",
        handlelength=1.5,
        columnspacing=1.0
    )
    leg.get_frame().set_boxstyle("round,pad=0.3")

# leave space for the three legends
plt.subplots_adjust(bottom=0.25)

plt.savefig("std_evolution_activator_rowfirst.png", dpi=600)
plt.show()


# optional: clean up CSVs
for f in glob.glob("std_sweep_*.csv"):
    os.remove(f)
