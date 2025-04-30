import numpy as np
import matplotlib.pyplot as plt

alpha = 0.15
beta = 0.30
Da = 1.0
Db = 60.0

def homogeneous_solution(alpha, beta):
    A0 = alpha + beta
    B0 = beta / (A0*A0)
    return A0, B0

#Define Dispersion Relation
def dispersion_relation_pos(k, alpha, beta, Da, Db):
    A, B = homogeneous_solution(alpha, beta)
    fa = 2*A*B -1
    fb = A*A
    ga = -2*A*B
    gb = -(A*A)
    tr = fa + gb
    det = fa*gb - fb*ga
    k2 = k*k
    b = k2*(Da + Db) - tr
    c = det + k2*(k2*(Da*Db) - gb*Da - fa*Db)
    return (-b + np.sqrt(b*b - 4*c))/2

def dispersion_relation_neg(k, alpha, beta, Da, Db):
    A, B = homogeneous_solution(alpha, beta)
    fa = 2*A*B -1
    fb = A*A
    ga = -2*A*B
    gb = -(A*A)
    tr = fa + gb
    det = fa*gb - fb*ga
    k2 = k*k
    b = k2*(Da + Db) - tr
    c = det + k2*(k2*(Da*Db) - gb*Da - fa*Db)
    return (-b - np.sqrt(b*b - 4*c))/2

# Set the domain
k = np.linspace(0, 3, 750)
k_complex = k.astype(np.complex128)

# Evaluate the function
w = dispersion_relation_pos(k_complex, alpha, beta, Da, Db)
w_neg = dispersion_relation_neg(k_complex, alpha, beta, Da, Db)

k_max_idx = np.argmax(w.real)
k_max1 = k[k_max_idx]
w_max = w[k_max_idx]
w_imag_max = w.imag[k_max_idx]
print(f"Maximum real part of w: {w_max.real} at k = {k_max1}")
print(f"Maximum imaginary part of w: {w_imag_max} at k = {k_max1}")


#Plot
fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex=True)

#Real W_Positive Real
axs[0, 0].plot(k, w.real, label=r'Re[$\omega_+$]', color='blue')
axs[0, 0].set_ylabel(r'Re[$\omega_+$]')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot W_Positive Imaginary
axs[1, 0].plot(k, w.imag, label=r'Im[$\omega_+$]', color='red')
axs[1, 0].set_ylabel(r'Im[$\omega_+$]')
axs[1, 0].legend()
axs[1, 0].grid(True)

#Real W_Negative Real
axs[0, 1].plot(k, w_neg.real, label=r'Re[$\omega_-$]', color='blue')
axs[0, 1].set_ylabel(r'Re[$\omega_-$]')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot W_Negative Imaginary
axs[1, 1].plot(k, w_neg.imag, label=r'Im[$\omega_-$]', color='red')
axs[1, 1].set_ylabel(r'Re[$\omega_-$]')
axs[1, 1].legend()
axs[1, 1].grid(True)

#Organize Figure
fig.suptitle('Dispersion Relations', fontsize=14)
plt.subplots_adjust(left=0.1, right=0.85, top=0.9)

param_texts = [
    fr'$\alpha$ = {alpha}',
    fr'$\beta$ = {beta}',
    fr'$\frac{{D_A}}{{D_B}}$ = {Da/Db:.2f}',
    fr'$k_{{\max}}$ = {k_max1:.2f}',
    fr'$\omega_{{\max}}$ = {w_max:.2f}',
]

fig.text(0.87, 0.5, '\n'.join(param_texts), fontsize='medium', ha='left', va='center')
plt.savefig(f'a{alpha:.2f}_b{beta:.2f}_diff{(Da/Db):.2f}.png', dpi=300)
plt.show()


# ----------------------------------------------------------------------
# public helpers expected by other scripts
# ----------------------------------------------------------------------

def omega(alpha, beta, DA, DB, k_complex, chi_a=0.0, chi_b=0.0):
    """
    Return the eigenvalue ω(k) with the maximum real part.
    For χ_a = χ_b = 0 this reduces to the classic Schnakenberg result.
    """
    # pick the correct dispersion relation depending on χ
    if abs(chi_a) < 1e-12 and abs(chi_b) < 1e-12:
        # no cross-diffusion: use the existing pos/neg functions
        w_plus  = dispersion_relation_pos(k, alpha, beta, DA, DB)
        w_minus = dispersion_relation_neg(k, alpha, beta, DA, DB)
    else:
        # include χ-terms explicitly (single quadratic; same algebra)
        A, B = homogeneous_solution(alpha, beta)
        fa, fb =  2*A*B - 1,  A**2
        ga, gb = -2*A*B,      -A**2
        k2     = k*k
        b  = k2*(DA+DB) - (fa+gb)
        c  = (fa*gb - fb*ga) \
           + k2*(k2*DA*DB - gb*DA - fa*DB) \
           - fb*chi_b - ga*chi_a + chi_a*chi_b
        disc = b*b - 4.0*c
        # numerical noise can push disc slightly < 0
        disc = disc if disc > 0 else 0.0
        w_plus  = (-b + np.sqrt(disc)) / 2.0
        w_minus = (-b - np.sqrt(disc)) / 2.0

    # return the one with the larger real part
    return w_plus if w_plus.real >= w_minus.real else w_minus


def k_max(alpha, beta, DA, DB, chi_a=0.0, chi_b=0.0,
          k_max=3.0, nk=600):
    """
    Brute-force search for the wavenumber with maximum Re ω.
    """
    k_grid = np.linspace(0.0, k_max, nk)
    growth = [omega(alpha, beta, DA, DB, k,
                    chi_a=chi_a, chi_b=chi_b).real
              for k in k_grid]
    idx = int(np.argmax(growth))
    return k_grid[idx]
