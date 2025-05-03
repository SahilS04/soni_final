import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormaps

# Define parameter space
# Using full resolution as requested
alpha_vals = np.linspace(0, 0.2, 400)
beta_vals = np.linspace(0, 2, 400)
d_ratios = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]  # Da / Db values

Da_base = 1.0

# Dispersion relation k range and resolution
# Using full resolution as requested
k = np.linspace(0, 30, 750)
k_complex = k.astype(np.complex128)
k_threshold = 1e-2 # Wavenumber threshold to distinguish from k=0

# --- Helper Functions ---
def homogeneous_solution(alpha, beta):
    """
    Calculates the homogeneous steady state (A0, B0).
    Returns (NaN, NaN) if the steady state is non-physical (A0<=0 or B0<=0)
    or mathematically undefined (A0=0).
    """
    A0 = alpha + beta
    # Handle potential division by zero or non-physical states
    if A0 == 0:
        return np.nan, np.nan # Mathematically undefined
    B0 = beta / (A0*A0)
    if A0 <= 0 or B0 <= 0: # Check for physical validity (concentrations must be positive)
        return np.nan, np.nan
    return A0, B0

def dispersion_relation_w(k_complex, alpha, beta, Da, Db):
    """
    Calculates both branches of the dispersion relation omega(k).
    Returns w_plus, w_minus. Handles potential complex sqrt issues
    and invalid steady states.
    """
    A, B = homogeneous_solution(alpha, beta)
    # If steady state is invalid, return NaNs
    if np.isnan(A):
        nan_array = np.full_like(k_complex, np.nan, dtype=np.complex128)
        return nan_array, nan_array

    # Calculate Jacobian elements evaluated at the steady state
    fa = 2*A*B - 1
    fb = A*A
    ga = -2*A*B
    gb = -A*A

    # Calculate trace and determinant of the reaction Jacobian
    tr = fa + gb
    det = fa*gb - fb*ga

    k2 = k_complex * k_complex # Wavenumber squared

    # Coefficients of the quadratic equation for omega: omega^2 + b*omega + c = 0
    # Note the sign convention might differ in sources, this matches the quadratic formula: (-b +/- sqrt(b^2-4ac))/2a with a=1
    b_term = k2 * (Da + Db) - tr
    c_term = det + k2 * (k2 * (Da * Db) - gb * Da - fa * Db)

    # Calculate discriminant carefully to handle complex results
    discriminant_complex = b_term**2 - 4*c_term
    # Ensure sqrt handles complex numbers correctly
    sqrt_discriminant = np.sqrt(discriminant_complex)

    # Calculate the two eigenvalues (growth rates) omega
    w_plus = (-b_term + sqrt_discriminant) / 2.0
    w_minus = (-b_term - sqrt_discriminant) / 2.0

    return w_plus, w_minus

# --- Main Loop ---
# Increase figure size for better readability
fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True) # Increased figsize
axs = axs.flatten()

# Store min/max Re(w) across all plots for consistent color scaling
global_min_re_w = np.inf
global_max_re_w = -np.inf
all_re_w_maps = [] # Store the computed maps for each d_ratio

print("Calculating dispersion relations (this may take a while)...")
for i, d_ratio in enumerate(d_ratios):
    Da = d_ratio * Da_base
    Db = Da_base
    print(f"  Processing D_A/D_B = {d_ratio:.2f}...")

    # Initialize map with NaNs (will indicate k_max=0 or invalid parameters)
    re_w_map = np.full((len(beta_vals), len(alpha_vals)), np.nan)

    for ai, alpha in enumerate(alpha_vals):
        for bi, beta in enumerate(beta_vals):
            # Calculate dispersion relations for both branches
            w_plus, w_minus = dispersion_relation_w(k_complex, alpha, beta, Da, Db)

            # Check if calculation was valid (might be NaN if steady state invalid)
            if np.isnan(w_plus).any() or np.isnan(w_minus).any():
                continue # Leave re_w_map[bi, ai] as NaN

            # Find the index of the maximum real part for each branch
            max_real_plus_idx = np.argmax(w_plus.real)
            max_real_minus_idx = np.argmax(w_minus.real)

            # Determine which branch has the overall maximum real part
            if w_plus[max_real_plus_idx].real >= w_minus[max_real_minus_idx].real:
                # Omega_plus branch dominates
                w_max_branch = w_plus
                k_max_idx = max_real_plus_idx
            else:
                # Omega_minus branch dominates
                w_max_branch = w_minus
                k_max_idx = max_real_minus_idx

            # Get the wavenumber and the corresponding omega at this maximum
            k_max = k[k_max_idx]
            w_max = w_max_branch[k_max_idx]

            # Store the real part of omega_max only if k_max is significantly greater than 0
            if k_max > k_threshold:
                re_w_map[bi, ai] = w_max.real

    all_re_w_maps.append(re_w_map)
    # Update global min/max, ignoring NaNs encountered in the map
    current_min = np.nanmin(re_w_map)
    current_max = np.nanmax(re_w_map)
    if not np.isnan(current_min): # Check if current_min is a valid number
        global_min_re_w = min(global_min_re_w, current_min)
    if not np.isnan(current_max): # Check if current_max is a valid number
        global_max_re_w = max(global_max_re_w, current_max)

# --- Plotting ---
print("Plotting heatmaps...")
# Determine colormap and range based on global min/max
if np.isinf(global_min_re_w) or np.isinf(global_max_re_w):
     print("Warning: No valid Re(omega) values found. Using default range.")
     global_min_re_w = -0.1
     global_max_re_w = 0.1

if global_min_re_w < 0 < global_max_re_w:
    cmap = cm.coolwarm
    limit = max(abs(global_min_re_w), abs(global_max_re_w))
    limit *= 1.05 # Add buffer
    vmin = -limit
    vmax = limit
elif global_max_re_w <= 0:
    cmap = cm.Blues_r
    vmin = global_min_re_w * 1.05 if global_min_re_w != 0 else -0.01
    vmax = 0
elif global_min_re_w >= 0:
    cmap = cm.Reds
    vmin = 0
    vmax = global_max_re_w * 1.05 if global_max_re_w != 0 else 0.01
else: # Fallback if only NaNs
    cmap = cm.coolwarm
    vmin = -0.1
    vmax = 0.1

cmap.set_bad('black') # Color for NaN values

im = None # Initialize im to be used for colorbar later
for i, d_ratio in enumerate(d_ratios):
    ax = axs[i]
    re_w_map = all_re_w_maps[i] # Get the map for this d_ratio

    # Display the heatmap
    im = ax.imshow(re_w_map, origin='lower',
                   extent=[alpha_vals[0], alpha_vals[-1], beta_vals[0], beta_vals[-1]],
                   aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # Set titles and labels
    ax.set_title(fr'$D_A / D_B$ = {d_ratio:.2f}', fontsize=12) # Slightly larger title font
    if i >= 3: # Only add x-label to bottom row plots
        ax.set_xlabel(r'$\alpha$', fontsize=12)
    if i % 3 == 0: # Only add y-label to left column plots
        ax.set_ylabel(r'$\beta$', fontsize=12)

    # Add grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.6)


# Add overall title
fig.suptitle(r'Maximum Growth Rate Re[$\omega(k_{max})$] (where $k_{max}>' + f'{k_threshold:.1e}$)', fontsize=18, y=0.98) # Increased font size and adjusted y

# --- Adjust layout and add colorbar ---
# Adjust subplot parameters for spacing and colorbar room
# left, bottom, right, top define the bounding box of the subplots
# wspace, hspace define the spacing between subplots
fig.subplots_adjust(left=0.08, bottom=0.08, right=0.85, top=0.92, wspace=0.3, hspace=0.3) # Adjusted right, wspace, hspace

# Add an axes to the right side of the figure for the colorbar
# [left, bottom, width, height] in figure coordinates (0 to 1)
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) # Positioned to the right

# Add the colorbar to the dedicated axes
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'Max Re[$\omega(k)$] for $k>k_{thr}$', fontsize=12) # Added fontsize

# Save the figure
plt.savefig('re_omega_heatmap_k_gt_0_adjusted.png', dpi=300, bbox_inches='tight')
print("Plot saved as re_omega_heatmap_k_gt_0_adjusted.png")
# Display the plot
plt.show()