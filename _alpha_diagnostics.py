"""Diagnostic: spectral/participation/overlap quantities that govern alpha.

No active learning, no tuning. One PLCFCM fit at 2% labels (seed 0, eta=0),
then replicate the exact R/alpha construction used by p2_method.select and
report every candidate effective-dimension statistic + the actual magnitude
of gain vs redundancy terms in the greedy score.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import numpy as np

sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments')
from src import config as C
from src.data import load_dataset
from src.plcfcm import plcfcm_fit
from src.robust_selection import (compute_redundancy_matrix, compute_alpha,
                                  robust_gain_dv_batch)

ALL = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment',
       'balance', 'aggregation', 'compound',
       'letter', 'shuttle', 'usps', 'fashion']

rows = []
for ds in ALL:
    X, y, info = load_dataset(ds)
    n, c = X.shape[0], info['c']
    rng = np.random.RandomState(0)
    init = max(c, int(0.02 * n))
    li = rng.choice(n, size=init, replace=False)
    lab = y[li].copy()
    U, V, _ = plcfcm_fit(X, li, lab, c=c, seed=0)

    unlabeled = np.setdiff1d(np.arange(n), li)
    Uu = U[unlabeled]
    nu = len(unlabeled)

    # Replicate p2_method R / alpha construction exactly
    if nu <= C.MAX_REDUNDANCY_N:
        R = compute_redundancy_matrix(Uu)
        alpha_used, lam2_used = compute_alpha(R, nu)
        n_eff_note = nu
    else:
        rng2 = np.random.RandomState(0)
        sub = rng2.choice(nu, size=min(C.REDUNDANCY_SUBSAMPLE_N, nu), replace=False)
        R = compute_redundancy_matrix(Uu[sub])
        alpha_used, lam2_used = compute_alpha(R, len(sub))
        n_eff_note = len(sub)

    m_size = R.shape[0]
    d = R.sum(axis=1)
    L = -R.copy()
    np.fill_diagonal(L, d)
    ev = np.linalg.eigvalsh(L)
    ev_pos = ev[1:]  # drop lambda1=0
    lam2 = float(ev[1])
    # Participation ratio of Laplacian (paper Eq. def), positive modes only
    sL = ev_pos[ev_pos > 1e-10]
    PR_L = float((sL.sum() ** 2) / np.sum(sL ** 2))
    # Effective rank of R (Gram), positive eigenvalues
    sv = np.linalg.eigvalsh(R)
    svp = sv[sv > 1e-10]
    PR_R = float((svp.sum() ** 2) / np.sum(svp ** 2))
    maxR = float(R.max())
    # mean off-diagonal cosine
    iu = np.triu_indices(m_size, k=1)
    rbar = float(R[iu].mean())
    # within / cross block cosine
    blocks = Uu[:m_size].argmax(axis=1) if nu <= C.MAX_REDUNDANCY_N else Uu[np.sort(sub)].argmax(axis=1)
    same = blocks[:, None] == blocks[None, :]
    mask = np.zeros_like(R, dtype=bool)
    mask[iu] = True
    r_within = float(R[mask & same].mean()) if np.any(mask & same) else float('nan')
    r_cross = float(R[mask & ~same].mean()) if np.any(mask & ~same) else float('nan')

    # candidate alphas
    alpha_code = min(max(lam2, 1.0) / m_size, 1.0 / (maxR + 1e-12))
    alpha_pr = min(lam2 / PR_L, 1.0 / (maxR + 1e-12))
    alpha_pr_floor = min(max(lam2, 1.0) / PR_L, 1.0 / (maxR + 1e-12))

    # gain scale vs redundancy term scale (one greedy step illustrative)
    rng3 = np.random.RandomState(0)
    gains, _ = robust_gain_dv_batch(Uu, c, kappa=C.ROBUST_KAPPA_BASE, rng=rng3)
    g_pos = gains[gains > 0]
    g_mean = float(g_pos.mean()) if len(g_pos) else 0.0
    # redundancy term magnitude at t=~10 selected: lam * alpha_within * mean r_within
    term_code = C.P2_LAM * (2 * alpha_code) * max(r_within, 0)
    term_pr = C.P2_LAM * (2 * alpha_pr_floor) * max(r_within, 0)

    rows.append(dict(ds=ds, n=n, c=c, nu=nu, m=m_size,
                     lam2=lam2, PR_L=PR_L, PR_R=PR_R,
                     PR_L_over_m=PR_L / m_size, maxR=maxR, rbar=rbar,
                     r_within=r_within, r_cross=r_cross,
                     a_code=alpha_code, a_pr=alpha_pr_floor,
                     gain=g_mean, term_code=term_code, term_pr=term_pr,
                     ratio_code=term_code / (g_mean + 1e-12),
                     ratio_pr=term_pr / (g_mean + 1e-12)))

hdr = f"{'dataset':>11} {'n':>6} {'c':>3} {'m':>5} {'lambda2':>10} {'PR(L)':>8} {'PR(R)':>7} {'PR/m':>6} {'maxR':>6} {'rbar':>6} {'r_w':>6} {'r_x':>6} {'a_code':>9} {'a_pr':>9} {'gain':>6} {'penCode/g':>9} {'penPr/g':>8}"
print(hdr)
for r in rows:
    print(f"{r['ds']:>11} {r['n']:>6} {r['c']:>3} {r['m']:>5} {r['lam2']:>10.3f} {r['PR_L']:>8.1f} {r['PR_R']:>7.2f} "
          f"{r['PR_L_over_m']:>6.3f} {r['maxR']:>6.3f} {r['rbar']:>6.3f} {r['r_within']:>6.3f} {r['r_cross']:>6.3f} "
          f"{r['a_code']:>9.5f} {r['a_pr']:>9.5f} {r['gain']:>6.3f} {r['ratio_code']:>9.4f} {r['ratio_pr']:>8.4f}")

import csv
with open(r'D:\P2OLSAL_FSS\experiments\results\alpha_diagnostics.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print('\nwritten results/alpha_diagnostics.csv')
