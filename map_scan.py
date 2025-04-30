#!/usr/bin/env python3
"""
map_scan.py  –  walk inside the stationary-Turing region and show
how cross-diffusion widens the pocket.

Output:
    map_noX.csv    # classification without χ
    map_X.csv      # with χ_a, χ_b
    optional PNG heat-maps

usage example
-------------
python map_scan.py --alpha 0.05 0.25 21 \
                   --beta  0.20 0.60 21 \
                   --DA 1   --DB 60 \
                   --chi_a 1  --chi_b 12 \
                   --kmax 3.0 --nk 600
"""
import argparse, csv, itertools, pathlib
import numpy as np
from dispersion_relation import omega   # your existing file

# -------------------- helper --------------------
def classify(alpha, beta, DA, DB, chi_a, chi_b,
             kmax=3.0, nk=600, thresh=1e-6):
    kgrid = np.linspace(0.0, kmax, nk)
    growth = []
    for k in kgrid:
        w = omega(alpha, beta, DA, DB, k,
                  chi_a=chi_a, chi_b=chi_b)
        growth.append((w.real, w.imag, k))
    # fastest-growing mode
    idx = int(np.argmax([g[0] for g in growth]))
    Re, Im, k_star = growth[idx]
    if Re < thresh:
        return "stable", k_star, Re, Im
    if k_star < thresh:
        return "hopf",   k_star, Re, Im
    if abs(Im) < 1e-3:
        return "turing", k_star, Re, Im
    return "turing-hopf", k_star, Re, Im

# -------------------- main ----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--alpha', nargs=3, type=float,
                   metavar=('start','stop','num'), required=True)
    p.add_argument('--beta',  nargs=3, type=float, required=True)
    p.add_argument('--DA', type=float, required=True)
    p.add_argument('--DB', type=float, required=True)
    p.add_argument('--chi_a', type=float, default=0.0)
    p.add_argument('--chi_b', type=float, default=0.0)
    p.add_argument('--kmax', type=float, default=3.0)
    p.add_argument('--nk',   type=int,   default=600)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()

    a_start, a_stop, a_num = args.alpha
    b_start, b_stop, b_num = args.beta
    a_grid = np.linspace(a_start, a_stop, int(a_num))
    b_grid = np.linspace(b_start, b_stop, int(b_num))

    out_noX, out_X = [], []
    for a,b in itertools.product(a_grid, b_grid):
        tag, kstar, Re, Im = classify(a,b,args.DA,args.DB,0,0,
                                      args.kmax,args.nk)
        out_noX.append((a,b,tag,kstar,Re,Im))
        tagX,kx,ReX,ImX = classify(a,b,args.DA,args.DB,
                                   args.chi_a,args.chi_b,
                                   args.kmax,args.nk)
        out_X.append((a,b,tagX,kx,ReX,ImX))

    def dump(name, rows):
        f = pathlib.Path(name).open('w', newline='')
        w = csv.writer(f)
        w.writerow(['alpha','beta','tag','k*','Reω','Imω'])
        w.writerows(rows)
        f.close()
        print("wrote", name)

    dump("map_noX.csv", out_noX)
    dump("map_X.csv",   out_X)

    if args.plot:
        import matplotlib.pyplot as plt
        def to_mat(rows, what):
            Z = np.zeros((len(a_grid),len(b_grid)))
            tag2int = {'stable':0,'hopf':1,'turing':2,'turing-hopf':3}
            for a,b,tag,_,_,_ in rows:
                i = np.where(a_grid==a)[0][0]
                j = np.where(b_grid==b)[0][0]
                Z[i,j] = tag2int[tag]
            return Z
        fig,axs = plt.subplots(1,2,figsize=(9,4),sharey=True)
        for ax,Z,title in zip(axs,[to_mat(out_noX,'tag'),
                                  to_mat(out_X,'tag')],
                              ['no cross-diff', 'with cross-diff']):
            im=ax.imshow(Z,origin='lower',
                         extent=[args.beta[0],args.beta[1],
                                 args.alpha[0],args.alpha[1]],
                         cmap='viridis',vmin=0,vmax=3)
            ax.set_title(title)
            ax.set_xlabel('β'); ax.set_ylabel('α')
        cbar=fig.colorbar(im,ax=axs.ravel().tolist(),
                          ticks=[0,1,2,3])
        cbar.ax.set_yticklabels(['stable','Hopf','Turing','T-Hopf'])
        plt.tight_layout(); plt.savefig('scan_map.png',dpi=250)
        print("saved scan_map.png")

if __name__ == '__main__':
    main()
