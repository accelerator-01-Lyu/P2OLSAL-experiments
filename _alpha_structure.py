"""Corrected structural diagnostic (fixes subsample block-label permutation).

Block labels MUST be computed on the SAME row order as R = redundancy(Um).
Adds gamma, gamma_W, r_W, PC, normalized entropy, p. Structural quantities at
the CURRENT U at each active-learning budget (scheme-neutral: uses PLCFCM with
incrementally grown random-labeled set, eta=0, seed 0), averaged over rounds.
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
from src.p2_method import block_contrast_stats, partition_diagnostics
from src.robust_selection import compute_redundancy_matrix

ALL = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment',
       'balance', 'aggregation', 'compound',
       'letter', 'shuttle', 'usps', 'fashion']
BUDGETS = [0.05, 0.10, 0.15, 0.20]

records = []
for ds in ALL:
    X, y, info = load_dataset(ds)
    n, c = X.shape[0], info['c']
    rng = np.random.RandomState(0)
    init = max(c, int(0.02 * n))
    li = rng.choice(n, size=init, replace=False).tolist()
    lab = y[np.array(li)].copy()
    round_stats = []
    for bf in BUDGETS:
        U, V, _ = plcfcm_fit(X, np.array(li), lab, c=c, seed=0)
        unlabeled = np.setdiff1d(np.arange(n), np.array(li))
        Uu = U[unlabeled]
        nu = len(Uu)
        if nu <= C.MAX_REDUNDANCY_N:
            Um = Uu
        else:
            r2 = np.random.RandomState(0)
            sub_idx = r2.choice(nu, size=min(C.REDUNDANCY_SUBSAMPLE_N, nu), replace=False)
            Um = Uu[sub_idx]                 # SAME order fed to R
        R = compute_redundancy_matrix(Um)
        st = block_contrast_stats(Um, R, c)  # blocks computed inside from Um -> aligned
        pc, hn = partition_diagnostics(U)
        st.update(PC=pc, H_norm=hn, budget=bf)
        round_stats.append(st)
        # grow labels for next round (random, scheme-neutral trajectory)
        k = max(1, int(bf * n) - len(li))
        remain = np.setdiff1d(np.arange(n), np.array(li))
        pick = rng.choice(remain, size=min(k, len(remain)), replace=False)
        li.extend(pick.tolist())
        lab = np.concatenate([lab, y[np.array(pick)]])
    # average over rounds
    def avg(k):
        return float(np.mean([s[k] for s in round_stats]))
    rec = dict(ds=ds, n=n, c=c,
               lambda2=avg('lambda2'), d_eff=avg('d_eff'),
               r_within=avg('r_within'), r_cross=avg('r_cross'),
               gamma=avg('gamma'), r_W=avg('r_W'), p=avg('p'),
               PC=avg('PC'), H_norm=avg('H_norm'))
    rec['gamma_W'] = float(np.clip((rec['r_within'] - rec['r_W']) /
                                   (1 - rec['r_W'] + 1e-12), 0, 1))
    records.append(rec)

hdr = (f"{'dataset':>11}{'n':>6}{'c':>3}{'d_eff':>7}{'lambda2':>10}{'r_w':>7}{'r_x':>7}"
       f"{'gamma':>7}{'gW':>7}{'r_W':>7}{'p':>6}{'PC':>7}{'Hnorm':>7}")
print(hdr)
for r in records:
    print(f"{r['ds']:>11}{r['n']:>6}{r['c']:>3}{r['d_eff']:>7.2f}{r['lambda2']:>10.2f}"
          f"{r['r_within']:>7.3f}{r['r_cross']:>7.3f}{r['gamma']:>7.3f}{r['gamma_W']:>7.3f}"
          f"{r['r_W']:>7.3f}{r['p']:>6.2f}{r['PC']:>7.3f}{r['H_norm']:>7.3f}")

import csv
with open(r'D:\P2OLSAL_FSS\experiments\results\alpha_structure.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader(); w.writerows(records)
print('\nwritten results/alpha_structure.csv')
