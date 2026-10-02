# -*- coding: utf-8 -*-
"""Robust aggregation: ecoli, several seeds x early rounds.

At every round we fit ONE membership U (rho=0 trajectory), score every unlabeled
candidate's fuzzy entropy H(u_i) and expected information gain g_mean
(= w_i^rob on ecoli where the DV floor is identically 0), then run P2OLSAL.select
twice on that SAME U with rho_expl=0 vs 0.01. Selected points are recorded per
group and aggregated. Output: results/selection_pool.csv
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
import numpy as np
import pandas as pd
from src.data import load_dataset
from src.plcfcm import plcfcm_fit, fuzzy_entropy
from src.p2_method import P2OLSAL
from src.robust_selection import robust_gain_dv_batch

SEEDS = [0, 1, 2, 3, 4]
N_ROUNDS = 6
BUDGET = 5


def run():
    X, y_true, info = load_dataset("ecoli")
    n, c = X.shape[0], info["c"]
    rows = []
    for s in SEEDS:
        rng = np.random.RandomState(s)
        init_count = max(c, int(0.02 * n))
        labeled_idx = rng.choice(n, size=init_count, replace=False).tolist()
        labels = y_true[labeled_idx].copy()
        for t in range(N_ROUNDS):
            labeled_mask = np.zeros(n, dtype=bool)
            labeled_mask[labeled_idx] = True
            U, V, _ = plcfcm_fit(X, np.array(labeled_idx), labels, c=c, seed=s + t)
            unlabeled = np.where(~labeled_mask)[0]
            _, g_mean = robust_gain_dv_batch(
                U[unlabeled], c, kappa=1.0, rng=np.random.RandomState(s + t))
            H_all = fuzzy_entropy(U[unlabeled])
            pos = {g: k for k, g in enumerate(unlabeled)}
            for rho in [0.0, 0.01]:
                m = P2OLSAL(robust=True, exploration=True, redundancy=True,
                            fmis=True, block=True, rho_expl=rho)
                sel, _ = m.select(X, U, V, labeled_mask, BUDGET, c,
                                  y_true=y_true, seed=s + t)
                for gi in sel:
                    rows.append({"seed": s, "round": t, "rho": rho,
                                 "H": float(H_all[pos[gi]]),
                                 "g_mean": float(g_mean[pos[gi]])})
            # advance along the rho=0 trajectory
            m0 = P2OLSAL(robust=True, exploration=True, redundancy=True,
                         fmis=True, block=True, rho_expl=0.0)
            sel0, _ = m0.select(X, U, V, labeled_mask, BUDGET, c,
                                y_true=y_true, seed=s + t)
            labeled_idx.extend(sel0)
            labels = np.concatenate([labels, y_true[sel0]])
    df = pd.DataFrame(rows)
    p = os.path.join(EXP, "results", "selection_pool.csv")
    df.to_csv(p, index=False)
    return df


def summarize(df):
    from scipy import stats
    out = {}
    for rho in [0.0, 0.01]:
        d = df[df.rho == rho]
        out[rho] = dict(
            n=len(d), H_m=d.H.mean(), H_se=d.H.sem(),
            g_m=d.g_mean.mean(), g_se=d.g_mean.sem())
    dH = out[0.01]["H_m"] - out[0.0]["H_m"]
    dg = out[0.01]["g_m"] - out[0.0]["g_m"]
    tH, p_H = stats.ttest_ind(df[df.rho == 0.01].H, df[df.rho == 0.0].H,
                              equal_var=False)
    tg, p_g = stats.ttest_ind(df[df.rho == 0.01].g_mean, df[df.rho == 0.0].g_mean,
                              equal_var=False)
    print(out)
    print(f"dH={dH:+.4f} (Welch t={tH:.2f}, p={p_H:.4g})")
    print(f"dg={dg:+.4f} (Welch t={tg:.2f}, p={p_g:.4g})")


if __name__ == "__main__":
    df = run()
    summarize(df)
