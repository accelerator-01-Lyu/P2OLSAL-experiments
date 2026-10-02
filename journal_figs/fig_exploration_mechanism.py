# -*- coding: utf-8 -*-
"""Mechanism figure 1: exploration double-counting cliff + selection shift.

(a) "Double-counting cliff": ACC@10% vs exploration weight rho_expl for ecoli
    and iris (seed-averaged, budget_frac=0.1). rho=0 plotted as the leftmost
    categorical point (log-scale cannot host 0); the steep 0 -> 0.01 drop is
    the double-counting cliff.
(b) "Selection score shift": selected samples scatter as fuzzy membership
    entropy $H(u_i)$ vs robust gain $w_i^{rob}$. rho=0 reaches high-entropy,
    near-zero-gain points; rho=0.01 drifts to low-entropy, negative-gain outliers.
Data: results/rho_ablation.csv, results/selection_shift_data.csv.
"""
import os, sys
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments\figures')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _journal_style as js

RES = r'D:\P2OLSAL_FSS\experiments\results'
RHO_ORDER = [0.0, 0.01, 0.1, 0.3, 0.5, 1.0]
RHO_TICK = ['0', '0.01', '0.1', '0.3', '0.5', '1.0']


def build():
    # ---------------- (a) double-counting cliff ----------------
    ra = pd.read_csv(os.path.join(RES, 'rho_ablation.csv'))
    ra = ra[ra.budget_frac == 0.1]
    g = ra.groupby(['dataset', 'rho_expl']).ACC.mean().unstack('rho_expl') * 100.0
    xpos = np.arange(len(RHO_ORDER))

    # ---------------- (b) selection score shift ----------------
    pool = pd.read_csv(os.path.join(RES, 'selection_pool.csv'))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(js.TEXTWIDTH_IN, 3.7))

    # --- panel (a) ---
    spec = [('ecoli', js.P2, 'o', 'ecoli'),
            ('iris', js.BASELINE_COLORS['BALD'], 'D', 'iris')]
    for ds, col, mk, lab in spec:
        y = [g.loc[ds, r] for r in RHO_ORDER]
        axA.plot(xpos, y, color=col, lw=2.0, marker=mk, ms=6.5,
                 mfc=col, mec='white', mew=0.9, zorder=6, label=lab)
    # highlight the 0 -> 0.01 cliff band
    axA.axvspan(-0.45, 0.95, color=js.P2, alpha=0.06, zorder=1)
    axA.annotate('double-counting cliff',
                 xy=(0.5, (g.loc['ecoli', 0.0] + g.loc['ecoli', 0.01]) / 2),
                 xytext=(2.0, 64.6), textcoords='data',
                 fontsize=9.0, color=js.P2, style='italic',
                 arrowprops=dict(arrowstyle='-', color='#9A9A9A', lw=0.55,
                                 shrinkA=0, shrinkB=5))
    axA.set_xticks(xpos)
    axA.set_xticklabels(RHO_TICK)
    axA.set_xlim(-0.45, 5.45)
    axA.set_xlabel(r'exploration weight $\rho_{\mathrm{expl}}$')
    axA.set_ylabel('ACC@10% (%)')
    axA.set_title('Double-counting cliff', fontsize=11.5)
    axA.legend(loc='lower left', framealpha=0.92)
    js.despine(axA)

    # --- panel (b) ---
    # aggregated selected points over 5 seeds x 6 rounds on ecoli; light x-jitter
    # so repeated near-identical picks do not stack. y is expected info gain
    # (g_mean = w_i^rob on ecoli, where the DV floor is identically 0).
    rng = np.random.default_rng(0)
    d0 = pool[pool.rho == 0.0]
    d1 = pool[pool.rho == 0.01]
    axB.scatter(d0.H + rng.normal(0, 0.008, len(d0)), d0.g_mean, s=26,
                color=js.BASELINE_COLORS['BADGE'], marker='s', alpha=0.55,
                edgecolor='white', linewidth=0.4, zorder=5,
                label=r'$\rho_{\mathrm{expl}}=0$ selected')
    axB.scatter(d1.H + rng.normal(0, 0.008, len(d1)), d1.g_mean, s=26,
                color=js.P2, marker='o', alpha=0.55,
                edgecolor='white', linewidth=0.4, zorder=6,
                label=r'$\rho_{\mathrm{expl}}=0.01$ selected')
    # group means
    axB.scatter([d0.H.mean()], [d0.g_mean.mean()], s=130,
                color=js.BASELINE_COLORS['BADGE'], marker='s', edgecolor='black',
                linewidth=1.0, zorder=8)
    axB.scatter([d1.H.mean()], [d1.g_mean.mean()], s=130,
                color=js.P2, marker='o', edgecolor='black', linewidth=1.0,
                zorder=9)
    axB.axhline(0, color='#777777', lw=1.0, zorder=1)
    axB.annotate('', xy=(0.10, d1.g_mean.mean()),
                 xytext=(0.24, d0.g_mean.mean()),
                 arrowprops=dict(arrowstyle='-|>', color=js.P2, lw=1.3,
                                 shrinkA=0, shrinkB=0), zorder=10)
    axB.text(1.72, -0.225,
             r'exploration bonus pulls selected points toward'
             '\n' r'low-entropy, low-$w^{\mathrm{rob}}$ membership outliers'
             '\n(double counting; 5 seeds × 6 rounds)',
             fontsize=7.6, color='#444444', style='italic',
             ha='right', va='center')
    axB.set_xlabel(r'fuzzy entropy $H(u_i^{\mathrm{FCM}})$')
    axB.set_ylabel(r'expected robust gain $w_i^{\mathrm{rob}}$')
    axB.set_title('Selection score shift', fontsize=11.5)
    axB.legend(loc='upper right', framealpha=0.92, fontsize=7.8)
    js.despine(axB)

    for j, ax in enumerate((axA, axB)):
        js.panel_label(ax, f'({chr(97 + j)})', x=-0.13, y=1.06)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.16,
                        wspace=0.30)
    # inter-panel mechanism arrow (figure-level, in the top gap)
    from matplotlib.patches import FancyArrowPatch
    arr = FancyArrowPatch((0.455, 0.925), (0.545, 0.925),
                          transform=fig.transFigure, arrowstyle='-|>',
                          color='#777777', lw=1.1, shrinkA=2, shrinkB=2,
                          mutation_scale=12)
    fig.patches.append(arr)
    fig.text(0.50, 0.955, 'double-counting mechanism', ha='center', va='bottom',
             fontsize=9.0, color='#555555', style='italic')
    paths = js.save_both(fig, 'fig_exploration_mechanism')
    print('exploration_mechanism', paths)
    return paths


if __name__ == '__main__':
    build()
