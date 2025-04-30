import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Define parameter space
alpha_vals = np.linspace(-0.2, 0.2, 400)
beta_vals = np.linspace(-2, 2, 400)
d_ratios = [0.01, 0.1, 1, 10, 100, 1000]  # Da / Db values

Da_base = 1.0

# Define regimes
REGIMES = {
    'pos_real_pos_imag': 0,
    'pos_real_zero_imag': 1,
    'neg_real_pos_imag': 2,
    'neg_real_zero_imag': 3,
    'invalid': 4
}

# Color map mapping regimes to integers
regime_colors = {
    0: 'red',       # pos_real_pos_imag
    1: 'orange',    # pos_real_zero_imag
    2: 'blue',      # neg_real_pos_imag
    3: 'gray',       # neg_real_zero_imag
    4: 'black'      # invalid
}

# Create a custom ListedColormap from regime_colors
from matplotlib.colors import ListedColormap
cmap = ListedColormap([regime_colors[i] for i in range(len(REGIMES))])

# Dispersion relation
k = np.linspace(0, 30, 750)
k_complex = k.astype(np.complex128)

def homogeneous_solution(alpha, beta):
    A0 = alpha + beta
    B0 = beta / (A0*A0)
    return A0, B0

def dispersion_relation_pos(k, alpha, beta, Da, Db):
    A, B = homogeneous_solution(alpha, beta)
    fa = 2*A*B - 1
    fb = A*A
    ga = -2*A*B
    gb = -A*A
    tr = fa + gb
    det = fa*gb - fb*ga

    b = k*k*(Da + Db) - tr
    c = det + k*k*(k*k*(Da*Db) - gb*Da - fa*Db)
    return (-b + np.sqrt(b*b - 4*c)) / 2

# Create figure with 3 columns and 2 rows
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.flatten()

k_threshold = 1e-6

for i, d_ratio in enumerate(d_ratios):
    Da = d_ratio * Da_base
    Db = Da_base

    regime_map = np.zeros((len(beta_vals), len(alpha_vals)))

    for ai, alpha in enumerate(alpha_vals):
        for bi, beta in enumerate(beta_vals):
            w = dispersion_relation_pos(k_complex, alpha, beta, Da, Db)
            k_max_idx = np.argmax(w.real)
            k_max = k[k_max_idx]
            w_max = w[k_max_idx]
            
            if (k_max > k_threshold):
                if w_max.real > 0 and abs(w_max.imag) > 0:
                    regime = REGIMES['pos_real_pos_imag']
                elif w_max.real > 0 and np.isclose(w_max.imag, 0):
                    regime = REGIMES['pos_real_zero_imag']
                elif w_max.real <= 0 and abs(w_max.imag) > 0:
                    regime = REGIMES['neg_real_pos_imag']
                else:
                    regime = REGIMES['neg_real_zero_imag']
            
            else:
                regime = REGIMES['invalid']

            regime_map[bi, ai] = regime  # Note: beta = row, alpha = col

    ax = axs[i]
    im = ax.imshow(regime_map, origin='lower', extent=[alpha_vals[0], alpha_vals[-1], beta_vals[0], beta_vals[-1]],
                   aspect='auto', cmap=cmap, vmin=0, vmax=len(REGIMES)-1)
    ax.set_title(fr'$D_A / D_B$ = {d_ratio:.2f}', fontsize=10)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')

# Create custom legend
legend_elements = [Patch(facecolor=regime_colors[val], label=key.replace('_', ' ')) for key, val in REGIMES.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize='small')

plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.suptitle('Pattern Regimes Across Parameter Space', fontsize=16)
plt.savefig('hopf_turing_pattern_regimes.png', dpi=300, bbox_inches='tight')
plt.show()
