#!/usr/bin/env python3
"""
speed_compare.py

Estimate wave speed from simulation CSV snapshots and compare it to the
linear-theory prediction  c_theory = Im[ω(k_max)] / k_max.

Requires:
    • numpy, pandas, scipy
    • dispersion_relation.py  (must define k_max(...) and omega(...))

Example:

    python speed_compare.py \
        --csv_glob 'A_*.csv' \
        --dt 0.001 --dx 1.0 \
        --alpha 0.1 --beta 0.5 \
        --DA 100.0 --DB 1.0

Steps:
  1. Load all CSVs matching the glob; assumes filenames are in time order.
  2. Take the mid-row, compute the k=1 Fourier phase φ(t).
  3. Linear-fit  φ ≈ k * c_exp * t   ⇒   experimental speed c_exp.
  4. Get k_max and ω(k_max) from dispersion_relation, compute c_theory.
  5. Print both speeds and the relative error.
"""

import argparse, glob, importlib, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ------------------------------------------------------------------ helpers
def load_stack(csv_glob):
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise FileNotFoundError(f"No files matching '{csv_glob}'")
    arr = [pd.read_csv(f, header=None).values for f in files]
    return np.stack(arr), files


def experimental_speed_fft(stack, dt, dx, k_index=1, row=None):
    """
    Compute experimental speed via phase of the k_index Fourier mode.
    """
    if row is None:
        row = stack.shape[1] // 2                     # middle row
    strip = stack[:, row, :]                          # (Nt, Nx)
    ft = np.fft.rfft(strip, axis=-1)                  # time × k
    phase = np.unwrap(np.angle(ft[:, k_index]))       # (Nt,)
    t = np.arange(len(phase)) * dt
    slope, _, _, _, _ = linregress(t, phase)          # phase ≈ k*c*t
    k = 2 * np.pi * k_index / (strip.shape[1] * dx)
    return slope / k                                 # c_exp


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_glob', required=True,
                   help="Glob for CSV snapshots, e.g. 'output/A_*.csv'")
    p.add_argument('--dt',  type=float, required=True,
                   help='Δt between saved frames')
    p.add_argument('--dx',  type=float, required=True,
                   help='Spatial grid spacing')
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--beta',  type=float, required=True)
    p.add_argument('--DA',    type=float, required=True)
    p.add_argument('--DB',    type=float, required=True)
    p.add_argument('--method', choices=['fft', 'peak'], default='fft',
                   help='How to estimate speed (default=fft phase)')
    p.add_argument('--row', type=int, default=None,
                   help='Row index to analyse (default=middle)')
    args = p.parse_args()

    # -------- load snapshots
    stack, files = load_stack(args.csv_glob)
    Nt = stack.shape[0]
    print(f"Loaded {Nt} frames from t = 0 to t = {(Nt-1)*args.dt:.2f}")

    # -------- experimental speed
    if args.method == 'fft':
        c_exp = experimental_speed_fft(stack, args.dt, args.dx,
                                       row=args.row)
    else:
        # simple peak-tracking fallback
        strip = stack[:, args.row or stack.shape[1]//2, :]
        x_peak = strip.argmax(axis=-1)
        t = np.arange(len(x_peak)) * args.dt
        slope, *_ = linregress(t, x_peak * args.dx)
        c_exp = slope
    print(f"Experimental speed  c_exp    = {c_exp:.5f}  (space units / time)")

    # -------- theoretical speed
    try:
        dr = importlib.import_module('dispersion_relation')
    except ImportError:
        print("ERROR: dispersion_relation.py not found on PYTHONPATH.")
        sys.exit(1)

    k_max, omega = dr.k_max(args.alpha, args.beta, args.DA, args.DB)
    c_theory = omega.imag / k_max
    print(f"Theoretical speed   c_theory = {c_theory:.5f}")
    rel_err = 100 * (c_exp - c_theory) / c_theory
    print(f"Relative error      = {rel_err:+.2f} %")


if __name__ == '__main__':
    main()
