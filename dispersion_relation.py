import numpy as np
import matplotlib.pyplot as plt

alpha = 0.025
beta = 0.5
Da = .05
Db = 1.0

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
plt.savefig(f'a{alpha:.3f}_b{beta:.3f}_diff{(Da/Db):.2f}.png', dpi=300)
plt.show()
