#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import glob
from scipy.stats import binned_statistic

# --------------------
# USER SETTINGS
# --------------------
NBINS   = 60
PRED_K  = 4.953   # <-- hardcode your predicted k here
DX      = 1.0   # grid spacing in simulation units
OUTPUT_VIDEO = 'a0.200_b2.000_diff0.01.mp4'

# --------------------
# FFT + Radial Functions
# --------------------
def compute_fft(field, dx=DX):
    """
    Compute 2D FFT and return:
      power : shifted power spectrum
      kgrid : radial wave‑number |k| in radians/unit length
      kx, ky: 1D arrays of k_x and k_y for axis extents
    """
    N, M = field.shape
    # center the FFT
    F = np.fft.fftshift(np.fft.fft2(field))
    power = np.abs(F)**2

    # frequencies in cycles/unit, then convert to radians/unit
    fx = np.fft.fftfreq(M, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    kx = 4*np.pi * np.fft.fftshift(fx)  # spans [-2pi, 2pi]
    ky = 4*np.pi * np.fft.fftshift(fy)
    KX, KY = np.meshgrid(kx, ky)
    kgrid  = np.sqrt(KX**2 + KY**2)
    return power, kgrid, kx, ky


def radial_average(power, kgrid, nbins=NBINS):
    """
    Compute radial average of power vs |k| using nbins bins.
    Returns bin centers and mean power.
    """
    k_flat = kgrid.ravel()
    p_flat = power.ravel()
    kmax   = k_flat.max()
    bins   = np.linspace(0, kmax, nbins+1)

    rad_mean, edges, _ = binned_statistic(
        k_flat, p_flat, statistic='mean', bins=bins
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, rad_mean

# --------------------
# Load Data Files
# --------------------
files = sorted(glob.glob("A_t*.csv"))
if not files:
    raise RuntimeError("No CSV files matching 'A_t*.csv'")
total_frames = len(files)

# --------------------
# Precompute Extents
# --------------------
fmin, fmax   = np.inf, -np.inf
smax         = 0
pfmax        = 0
rmin, rmax   = np.inf, -np.inf

# sample first frame
sample        = np.loadtxt(files[0], delimiter=',')
p0, kg0, kx, ky = compute_fft(sample)
rb0, _       = radial_average(p0, kg0)

for fname in files:
    fld = np.loadtxt(fname, delimiter=',')
    fmin, fmax = min(fmin, fld.min()), max(fmax, fld.max())
    smax       = max(smax, np.std(fld))

    p, kg, *_ = compute_fft(fld)
    p[kg==0]  = 0         # zero out DC
    pfmax     = max(pfmax, p.max())

    _, rad = radial_average(p, kg)
    rmin, rmax = min(rmin, rad.min()), max(rmax, rad.max())

# --------------------
# Setup Figure
# --------------------
fig, axs = plt.subplots(2, 2, figsize=(12,10))
ax_std, ax_rad, ax_field, ax_fft = axs.flatten()

# Initial field image
field_im = ax_field.imshow(
    sample, vmin=fmin, vmax=fmax, cmap='viridis', origin='lower'
)

# Initial FFT image
fft_power, fft_kgrid, kx, ky = compute_fft(sample)
fft_power[fft_kgrid==0] = 0
fft_im = ax_fft.imshow(
    fft_power,
    norm=LogNorm(vmin=1e-1, vmax=pfmax),
    origin='lower',
    extent=[kx.min(), kx.max(), ky.min(), ky.max()]
)

# Time‑series and radial plot lines
std_line, = ax_std.plot([], [], 'b-', lw=2)
rad_line, = ax_rad.plot([], [], 'r-', lw=2)
# Predicted k marker
dashed = ax_rad.axvline(
    PRED_K, color='k', linestyle='--', label='predicted k'
)
ax_rad.legend()

# Titles & labels
ax_field.set_title('Activator Field')
ax_fft.set_title('2D Power Spectrum')
ax_std.set_title('Std Dev of Field')
ax_rad.set_title('Radially Averaged Power Spectrum')

ax_std.set_xlim(0, total_frames)
ax_std.set_ylim(0, smax*1.1)
ax_rad.set_yscale('log')
ax_rad.set_xlim(rb0[0], rb0[-1])
ax_rad.set_ylim(rmin*0.1, rmax*1.1)
ax_field.set_xlabel('x')
ax_field.set_ylabel('y')
ax_fft.set_xlabel(r'$k_x$')
ax_fft.set_ylabel(r'$k_y$')
ax_rad.set_xlabel(r'$|\mathbf{k}|$')

# Colorbars
fig.colorbar(field_im, ax=ax_field, label='Concentration')
fig.colorbar(fft_im, ax=ax_fft, label='Power Intensity')

std_vals = []

# --------------------
# Animation Functions
# --------------------
def init():
    std_line.set_data([], [])
    rad_line.set_data([], [])
    return field_im, fft_im, std_line, rad_line


def update(frame):
    fld = np.loadtxt(files[frame], delimiter=',')
    field_im.set_data(fld)

    # FFT update
    p, kg, _, _ = compute_fft(fld)
    p[kg==0]    = 0
    fft_im.set_data(p)

    # std dev update
    std_vals.append(np.std(fld))
    xs = np.arange(len(std_vals))
    std_line.set_data(xs, std_vals)
    ax_std.set_title(f'Std Dev (frame {frame+1}/{total_frames})')

    # radial update
    centers, rad = radial_average(p, kg)
    rad_line.set_data(centers, rad)
    return field_im, fft_im, std_line, rad_line

# --------------------
# Create & Save Animation
# --------------------
ani = animation.FuncAnimation(
    fig, update, init_func=init,
    frames=total_frames, interval=100, blit=False
)
ani.save(OUTPUT_VIDEO, writer='ffmpeg', fps=90)
plt.show()