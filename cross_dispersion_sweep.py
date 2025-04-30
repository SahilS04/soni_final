import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Define parameter space
alpha_vals = np.linspace(-.20, .20, 400)  # alpha values
beta_vals = np.linspace(-2, 2, 400)  # beta values
D_ratio = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11]  # Chi_B ratio

Xa = 1.0
Xb = 10.0
Db_base = 1.0

# Define regimes
REGIMES = {
    'pos_real_pos_imag': 0,
    'pos_real_zero_imag': 1,
    'neg_real_pos_imag': 2,
    'neg_real_zero_imag': 3,
    'invald': 4
}

# Color map mapping regimes to integers
regime_colors = {
    0: 'red',       # pos_real_pos_imag
    1: 'orange',    # pos_real_zero_imag
    2: 'blue',      # neg_real_pos_imag
    3: 'gray',       # neg_real_zero_imag
    4: 'black'
}

# Create a custom ListedColormap from regime_colors
from matplotlib.colors import ListedColormap
cmap = ListedColormap([regime_colors[i] for i in range(len(REGIMES))])

# Dispersion relation
k = np.linspace(0, 15, 750)
k_complex = k.astype(np.complex128)

def homogeneous_solution(alpha, beta):
    A0 = alpha + beta
    B0 = beta / (A0*A0)
    return A0, B0

def dispersion_relation_pos(k, alpha, beta, Xa, Xb, D_ratio):
    A, B = homogeneous_solution(alpha, beta)
    Db = Db_base
    Da = Db * D_ratio
    fa = 2*A*B - 1
    fb = A*A
    ga = -2*A*B
    gb = -A*A
    tr = fa + gb
    det = fa*gb - fb*ga
    k2 = k*k
    b = k2*(Da + Db) - tr
    c = det + k2*(k2*(Da*Db) - gb*Da - fa*Db) - fb*Xb - ga*Xa + Xb*Xa
    return (-b + np.sqrt(b*b - 4*c)) / 2

# Create figure with 3 columns and 2 rows
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.flatten()

for i, D_ratio in enumerate(D_ratio):
    regime_map = np.zeros((len(beta_vals), len(alpha_vals)))

    for ai, alpha in enumerate(alpha_vals):
        for bi, beta in enumerate(beta_vals):
            w = dispersion_relation_pos(k_complex, alpha, beta, Xa, Xb, D_ratio)
            k_max_idx = np.argmax(w.real)
            k_max = k[k_max_idx]
            w_max = w[k_max_idx]

            if (k_max > 1):
                if w_max.real > 0 and abs(w_max.imag) > 0:
                    regime = REGIMES['pos_real_pos_imag']
                elif w_max.real > 0 and np.isclose(w_max.imag, 0):
                    regime = REGIMES['pos_real_zero_imag']
                elif w_max.real <= 0 and abs(w_max.imag) > 0:
                    regime = REGIMES['neg_real_pos_imag']
                else:
                    regime = REGIMES['neg_real_zero_imag']
            
            else:
                regime = REGIMES['invald']
            

            regime_map[bi, ai] = regime  # Note: beta = row, alpha = col

    ax = axs[i]
    im = ax.imshow(regime_map, origin='lower', extent=[alpha_vals[0], alpha_vals[-1], beta_vals[0], beta_vals[-1]],
                   aspect='auto', cmap=cmap, vmin=0, vmax=len(REGIMES)-1)
    ax.set_title(fr'$\frac{{D_A}}{{D_B}} = {D_ratio:.2f}$', fontsize=10)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')

# Create legend
legend_elements = [Patch(facecolor=regime_colors[val], label=key.replace('_', ' ')) for key, val in REGIMES.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize='small')
fig.text(0.5, 0.06, r'$\frac{\chi_A}{\chi_B}$ = 0.1', ha='center', fontsize=9)
plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.suptitle('Pattern Regimes Across Parameter Space', fontsize=16)
plt.savefig('hopf_turing_cross_0.1.png', dpi=300, bbox_inches='tight')
plt.show()
