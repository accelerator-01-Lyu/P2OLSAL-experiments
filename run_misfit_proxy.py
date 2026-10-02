"""Fig 5: Misfit-proxy validity on ecoli.

Runs P2_Full (rho=0) and records every round t:
  - delta_mis_hat: centroid-perturbation ENSEMBLE VARIANCE.  The library's
    ``misclassification_proxy`` stub computes within-row across-class variance
    (its bootstrap loop adds fresh noise each iter and never accumulates U),
    which on ecoli is flat and uncorrelated.  We therefore implement the
    documented semantics directly: bootstrap-resample the labeled set, refit
    PLCFCM, and measure per-sample variance of membership across the B
    refits, averaged over the unlabeled pool.
  - zeta_t: mean L1 distance between expert one-hot and model membership for
    the samples selected in that round.
Expectation: significant positive correlation.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from src.data import load_dataset
from src.plcfcm import plcfcm_fit
from src.p2_method import P2OLSAL


def ensemble_variance(X, labeled_idx, labels, c, seed, n_boot=8):
    """Per-sample ensemble variance of membership under bootstrap label resample.

    Returns array of shape (n,): mean across classes of var across bootstraps.
    """
    rng = np.random.RandomState(seed + 999)
    n = X.shape[0]
    U_perts = []
    li = np.asarray(labeled_idx)
    for b in range(n_boot):
        idx_boot = rng.choice(li, size=len(li), replace=True)
        lab_boot = labels[[list(li).index(i) for i in idx_boot]]
        U_b, _, _ = plcfcm_fit(X, idx_boot, lab_boot, c=c, seed=seed + b)
        U_perts.append(U_b)
    U_perts = np.stack(U_perts, axis=0)          # (B, n, c)
    per_sample = U_perts.var(axis=0).mean(axis=1)  # (n,)
    return per_sample


def main():
    X, y_true, info = load_dataset("ecoli")
    n, c = X.shape[0], info["c"]
    print(f"[ecoli] n={n}, d={X.shape[1]}, c={c}")

    BUDGET = 5
    N_ROUNDS = 12
    SEED = 0

    rng = np.random.RandomState(SEED)
    init_count = max(c, int(0.02 * n))
    labeled_idx = rng.choice(n, size=init_count, replace=False).tolist()
    labels = y_true[labeled_idx].copy()
    # P2_Full defaults: rho=0, robust=True, redundancy=True
    method = P2OLSAL(robust=True, exploration=True, redundancy=True,
                     fmis=True, block=True, rho_expl=0.0, name="P2_Full")

    rows = []
    for t in range(N_ROUNDS):
        U, V, _ = plcfcm_fit(X, np.array(labeled_idx), labels, c=c,
                             seed=SEED + t)
        labeled_mask = np.zeros(n, dtype=bool)
        labeled_mask[labeled_idx] = True
        sel_idx, diag = method.select(
            X, U, V, labeled_mask, BUDGET, c,
            y_true=y_true, seed=SEED + t)

        unlabeled = np.where(~labeled_mask)[0]
        # delta_mis_hat: bootstrap ensemble variance over unlabeled pool
        ens_var = ensemble_variance(X, labeled_idx, labels, c, seed=SEED + t)
        delta_mis = float(np.mean(ens_var[unlabeled]))

        # zeta_t: expert-model L1 disagreement on selected samples
        zetas = []
        for s in sel_idx:
            s_onehot = np.zeros(c)
            s_onehot[y_true[s]] = 1.0
            zetas.append(float(np.sum(np.abs(s_onehot - U[s]))))
        zeta_t = float(np.mean(zetas))

        rows.append({
            "t": t + 1,
            "n_labeled": len(labeled_idx),
            "delta_mis_hat": delta_mis,
            "zeta_t": zeta_t,
        })
        print(f"round {t+1:2d}: n_labeled={len(labeled_idx):3d}, "
              f"delta_mis={delta_mis:.5f}, zeta_t={zeta_t:.4f}")

        new_labels = y_true[sel_idx]
        labeled_idx.extend(sel_idx)
        labels = np.concatenate([labels, new_labels])

    df = pd.DataFrame(rows)
    res_dir = os.path.join(HERE, "results")
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, "misfit_proxy_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df)

    r, p = pearsonr(df.delta_mis_hat.values, df.zeta_t.values)
    print(f"\nPearson r = {r:.3f}, p = {p:.4f}, n = {len(df)}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(df.delta_mis_hat, df.zeta_t, c="#2C3E50", s=70, zorder=3,
               edgecolors="white", linewidths=0.6)
    for _, row in df.iterrows():
        ax.annotate(f"t={int(row.t)}",
                    (row.delta_mis_hat, row.zeta_t),
                    fontsize=9, xytext=(6, 5),
                    textcoords="offset points")

    slope, intercept = np.polyfit(df.delta_mis_hat.values,
                                 df.zeta_t.values, 1)
    xs = np.linspace(df.delta_mis_hat.min() * 0.95,
                     df.delta_mis_hat.max() * 1.05, 50)
    ax.plot(xs, slope * xs + intercept, "--", color="#E74C3C",
            lw=1.8, label=f"linear fit (Pearson r = {r:.2f}, p = {p:.3f})")

    ax.set_xlabel(r"$\delta_{\mathrm{mis}}^{\hat{}}$ "
                  r"(centroid-perturbation ensemble variance)", fontsize=14)
    ax.set_ylabel(r"$\zeta_t$ (expert–model $L_1$ disagreement)",
                  fontsize=14)
    ax.legend(fontsize=12, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    fig_dir = os.path.join(HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "fig5_misfit_proxy.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
