# -*- coding: utf-8 -*-
"""Sample 1: three-regime regret (synthetic controlled simulation).

SYNTHETIC CONTROLLED SIMULATION, illustrative of Theorems 6.3-6.5, NOT a real
dataset. Stylized scalar DGP for the zero-mean opportunity-cost shock X_t
(200 independent seeds, T=200, base seed 20261001); realized average regret is
the magnitude of the cumulative mean, Regret(T)/T = |(1/T) sum X_t|:
  well-specified : X_t = a eps_t,                eps_t~N(0,1) iid ; E|sum|/T = a sqrt(2/pi) T^-1/2 -> 0
  misspecified   : X_t = rho_* + a eps_t                            ; mean -> rho_* > 0 (Thm 6.4 floor)
  adversarial    : X_t = min{2 C, C(1+0.15 eps_t)}                  ; mean -> C_adv     (Thm 6.5 bound)
a=0.60, rho_*=0.25, C_adv=0.80.
"""
import os, sys
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments\figures')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _journal_style as js


def build():
    res = r'D:\P2OLSAL_FSS\experiments\results'
    a, rho, Cadv, T, S = 0.60, 0.25, 0.80, 200, 200
    rounds = np.arange(1, T + 1)
    master = np.random.default_rng(20261001)
    W = np.zeros((S, T)); M = np.zeros((S, T)); A = np.zeros((S, T))
    for s in range(S):
        rng = np.random.default_rng(master.integers(1, 2**31 - 1))
        e = rng.standard_normal(T)
        W[s] = np.abs(np.cumsum(a * e)) / rounds
        M[s] = np.cumsum(rho + a * e) / rounds
        A[s] = np.cumsum(np.minimum(2 * Cadv, Cadv * (1.0 + 0.15 * e))) / rounds
    sims = {'well-specified': W, 'misspecified': M, 'adversarial': A}
    rows = []
    for k, Z in sims.items():
        m = Z.mean(0); sd = Z.std(0, ddof=1)
        for i, t in enumerate(rounds):
            rows.append(dict(regime=k, round=int(t), mean=m[i], sd=sd[i],
                             lo95=m[i] - 1.96 * sd[i] / np.sqrt(S),
                             hi95=m[i] + 1.96 * sd[i] / np.sqrt(S),
                             a=a, rho_star=rho, C_adv=Cadv, T=T, n_seeds=S,
                             seed_base=20261001))
    pd.DataFrame(rows).to_csv(os.path.join(res, 'three_regime_sim.csv'), index=False)

    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.95))
    # (a) well-specified
    ax = axs[0]; m = W.mean(0); sd = W.std(0, ddof=1)
    band = 1.96 * sd / np.sqrt(S)
    ax.fill_between(rounds, m - band, m + band, color=js.P2, alpha=0.15, zorder=2,
                    linewidth=0)
    ax.plot(rounds, m, color=js.P2, lw=2.6, zorder=4,
            label=r'simulated $\mathrm{Regret}(T)/T$')
    bound = a * np.sqrt(2.0 / np.pi) / np.sqrt(rounds)
    ax.plot(rounds, bound, '--', color='black', lw=1.7, zorder=5,
            label=r'theory: $a\sqrt{2/\pi}\,T^{-1/2}$')
    ax.annotate(r'$O(T^{-1/2})\to 0$', xy=(150, bound[149] + 0.02), fontsize=10,
                color='black', ha='center')
    ax.set_title('Well-specified i.i.d.'); ax.set_xlabel('round $T$')
    ax.set_ylabel(r'average regret $\mathrm{Regret}(T)/T$')
    ax.set_ylim(0, 0.56); ax.legend(loc='upper right')
    # (b) misspecified
    ax = axs[1]; m = M.mean(0); sd = M.std(0, ddof=1); band = 1.96 * sd / np.sqrt(S)
    ax.fill_between(rounds, m - band, m + band, color=js.P2, alpha=0.15, zorder=2,
                    linewidth=0)
    ax.plot(rounds, m, color=js.P2, lw=2.6, zorder=4)
    ax.axhline(rho, ls='--', color='black', lw=1.7, zorder=5,
               label=fr'irreducible floor $\rho_*={rho}$')
    ax.annotate(fr'$\rho_*={rho}$ (Thm. 6.4)', xy=(104, rho + 0.012), fontsize=10,
                ha='center')
    ax.set_title('Model misspecification'); ax.set_xlabel('round $T$')
    ax.set_ylim(0.05, 0.46); ax.legend(loc='upper right')
    # (c) adversarial
    ax = axs[2]; m = A.mean(0); sd = A.std(0, ddof=1); band = 1.96 * sd / np.sqrt(S)
    ax.fill_between(rounds, m - band, m + band, color=js.P2, alpha=0.15, zorder=2,
                    linewidth=0)
    ax.plot(rounds, m, color=js.P2, lw=2.6, zorder=4)
    ax.axhline(Cadv, ls='--', color='black', lw=1.7, zorder=5,
               label=fr'bound $C_{{\rm adv}}={Cadv}$')
    ax.annotate(fr'$C_{{\rm adv}}={Cadv}$ (Thm. 6.5)', xy=(100, Cadv + 0.02),
                fontsize=10, ha='center')
    ax.set_title('Adversarial oracle'); ax.set_xlabel('round $T$')
    ax.set_ylim(0.55, 1.02); ax.legend(loc='lower right')
    for j, ax in enumerate(axs):
        js.despine(ax)
        js.panel_label(ax, f'({chr(97 + j)})', x=-0.03, y=1.06)
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.07, right=0.99,
                        wspace=0.26)
    fig.text(0.5, 0.99,
             'Synthetic controlled simulation — illustrative of Theorems 6.3–6.5',
             ha='center', va='top', fontsize=12.5, fontweight='bold')
    js.subtitle(
        fig,
        r'stylized scalar opportunity-cost shock $X_t$;  $\mathrm{Regret}(T)/T=|\sum_{t}X_t|/T$;  200 seeds, $T=200$',
        y=0.915)
    paths = js.save_both(fig, 'fig_three_regime', tight=False)
    print('three_regime', paths)
    return paths


if __name__ == '__main__':
    build()
