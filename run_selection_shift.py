"""Fig 2: Selection-shift scatter on ecoli.

Compares rho_expl=0 (optimal) vs rho_expl=0.01 (harmful). For every selected
sample we record its FCM fuzzy entropy H(u_i) and its robust information gain
w_i^rob.
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

from src.data import load_dataset
from src.plcfcm import plcfcm_fit, fuzzy_entropy
from src.p2_method import P2OLSAL
from src.robust_selection import robust_gain_dv_batch


def main():
    X, y_true, info = load_dataset("ecoli")
    n, c = X.shape[0], info["c"]
    print(f"[ecoli] n={n}, d={X.shape[1]}, c={c}")

    BUDGET = 5
    N_ROUNDS = 5
    SEED = 0

    rows = []
    for rho in [0.0, 0.01]:
        rng = np.random.RandomState(SEED)
        init_count = max(c, int(0.02 * n))
        labeled_idx = rng.choice(n, size=init_count, replace=False).tolist()
        labels = y_true[labeled_idx].copy()
        method = P2OLSAL(robust=True, exploration=True, redundancy=True,
                         fmis=True, block=True, rho_expl=rho,
                         name=f"P2_rho{rho}")

        for t in range(N_ROUNDS):
            U, V, _ = plcfcm_fit(X, np.array(labeled_idx), labels, c=c,
                                 seed=SEED + t)
            labeled_mask = np.zeros(n, dtype=bool)
            labeled_mask[labeled_idx] = True
            sel_idx, diag = method.select(
                X, U, V, labeled_mask, BUDGET, c,
                y_true=y_true, seed=SEED + t)

            unlabeled = np.where(~labeled_mask)[0]
            # Re-derive gains for all unlabeled with the same seed the method
            # used internally (robust_gain_dv_batch is the first rng call in
            # select(), so a fresh RandomState reproduces the MC draws).
            # NOTE: on ecoli the DV robust floor w_rob (1st return) is
            # identically 0 (g_mean < kappa/2 everywhere); the codebase's own
            # point_gains / zeta_t record g_mean (2nd return), which is the
            # per-sample expected information gain that actually varies.
            w_rob_all, g_mean_all = robust_gain_dv_batch(
                U[unlabeled], c, kappa=1.0,
                rng=np.random.RandomState(SEED + t))
            pos_of = {g: k for k, g in enumerate(unlabeled)}

            for s in sel_idx:
                H = float(fuzzy_entropy(U[s:s + 1])[0])
                w = float(g_mean_all[pos_of[s]])
                w_floor = float(w_rob_all[pos_of[s]])
                rows.append({
                    "rho": rho, "round": t, "idx": int(s),
                    "H": H, "w_rob": w, "w_rob_floor": w_floor,
                })

            new_labels = y_true[sel_idx]
            labeled_idx.extend(sel_idx)
            labels = np.concatenate([labels, new_labels])
            print(f"  rho={rho} round={t}: selected={sel_idx}, "
                  f"n_labeled={len(labeled_idx)}")

    df = pd.DataFrame(rows)
    res_dir = os.path.join(HERE, "results")
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, "selection_shift_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path} ({len(df)} rows)")
    print(df.groupby("rho")[["H", "w_rob"]].agg(["mean", "min", "max"]))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    d0 = df[df.rho == 0.0]
    d1 = df[df.rho == 0.01]
    ax.scatter(d0.H, d0.w_rob, c="#3498DB", marker="o", s=75, alpha=0.85,
               edgecolors="white", linewidths=0.6,
               label=r"$\rho=0$ (optimal)")
    ax.scatter(d1.H, d1.w_rob, c="#E74C3C", marker="^", s=90, alpha=0.9,
               edgecolors="white", linewidths=0.6,
               label=r"$\rho=0.01$ (harmful)")
    ax.set_xlabel(r"$H(u_i^{\mathrm{FCM}})$ (fuzzy entropy, nats)",
                  fontsize=15)
    ax.set_ylabel(r"$w_i^{\mathrm{rob}}$ (robust information gain)",
                  fontsize=15)
    ax.set_xlim(left=0.0)
    ax.legend(fontsize=13, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    fig_dir = os.path.join(HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "fig2_selection_shift.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
