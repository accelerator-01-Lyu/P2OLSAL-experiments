# -*- coding: utf-8 -*-
"""Mechanism figure 4: parameter invariance (a,b,d drawn; c reserved).

(a) alpha scale A vs B vs C on six representative datasets (B == C).
(b) alpha0 sensitivity: ACC vs alpha0 is essentially flat (invariance).
(c) placeholder: per-round gamma trajectory (data pending).
(d) c0 sensitivity: four large datasets, ACC identical at c0 in {.5,1.067,2}.
Data: results/alpha_scheme_acc10_means.csv, results/alpha0_sensitivity.csv,
      results/c0_sensitivity_stage1.csv.
"""
import os, sys
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments\figures')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _journal_style as js

RES = r'D:\P2OLSAL_FSS\experiments\results'
BIG6 = ['fashion', 'letter', 'shuttle', 'usps', 'ecoli', 'compound']


def build():
    a_sch = pd.read_csv(os.path.join(RES, 'alpha_scheme_acc10_means.csv')).set_index('dataset')
    a0 = pd.read_csv(os.path.join(RES, 'alpha0_sensitivity.csv'))
    c0 = pd.read_csv(os.path.join(RES, 'c0_sensitivity_stage1.csv')).set_index('dataset')
    gam = pd.read_csv(os.path.join(RES, 'gamma_trajectory.csv'))

    fig, ((axa, axb), (axc, axd)) = plt.subplots(2, 2, figsize=(js.TEXTWIDTH_IN, 5.6))

    # --- (a) alpha scheme ---
    x = np.arange(len(BIG6)); w = 0.26
    axa.bar(x - w, a_sch.loc[BIG6, 'A'].values * 100, width=w * 0.92,
            color=js.P2, edgecolor='white', linewidth=0.4, label='A')
    axa.bar(x, a_sch.loc[BIG6, 'B'].values * 100, width=w * 0.92,
            color=js.BASELINE_COLORS['BADGE'], edgecolor='white', linewidth=0.4,
            label='B')
    axa.bar(x + w, a_sch.loc[BIG6, 'C'].values * 100, width=w * 0.92,
            color='#6BAED6', edgecolor='white', linewidth=0.4, label='C')
    axa.set_xticks(x); axa.set_xticklabels(BIG6, rotation=30, ha='right', fontsize=8.5)
    axa.set_ylabel('ACC@10% (%)')
    axa.set_title('alpha scale: A vs B vs C (B=C)', fontsize=11)
    axa.legend(loc='lower right', fontsize=8.5, ncol=3, framealpha=0.92)
    js.despine(axa)

    # --- (b) alpha0 sensitivity ---
    a0b = a0[a0.budget_frac == 0.1]
    g = a0b.groupby(['dataset', 'alpha0']).ACC.mean().unstack('alpha0') * 100.0
    avals = sorted(g.columns.tolist())
    for ds in g.index:
        axb.plot(avals, g.loc[ds].values, '-o', color='#9E9E9E', lw=1.1, ms=3.5,
                 alpha=0.8, zorder=3)
    mean = g.mean(axis=0).reindex(avals)
    axb.plot(avals, mean.values, '-o', color=js.P2, lw=2.4, ms=6, zorder=6,
             label='mean (ecoli, iris)')
    axb.set_xscale('log')
    axb.set_xticks(avals); axb.set_xticklabels([str(v) for v in avals])
    axb.set_xlabel(r'$\alpha_0$ (log scale)')
    axb.set_ylabel('ACC@10% (%)')
    axb.set_title(r'$\alpha_0$ sensitivity — flat = invariant', fontsize=11)
    axb.legend(loc='best', fontsize=8.0, framealpha=0.92)
    js.despine(axb)

    # --- (c) per-round gamma trajectory ---
    gstyle = [('fashion', js.P2, 2.4, 'o'),
              ('letter', js.BASELINE_COLORS['CoreSet'], 1.3, 's'),
              ('shuttle', js.BASELINE_COLORS['BALD'], 1.3, '^'),
              ('usps', js.BASELINE_COLORS['QBC'], 1.3, 'D')]
    for ds, col, lw, mk in gstyle:
        d = gam[gam.dataset == ds]
        piv = d.groupby('round').gamma
        mean = piv.mean()
        lo, hi = piv.min(), piv.max()
        axc.fill_between(mean.index.values, lo.values, hi.values,
                         color=col, alpha=0.12, linewidth=0, zorder=2)
        axc.plot(mean.index.values, mean.values, color=col, lw=lw, marker=mk,
                 ms=3.5, zorder=6 if ds == 'fashion' else 4, label=ds)
    axc.annotate(r'fashion: $r_{\mathrm{within}}\approx0.99$ but '
                 r'$\mathrm{PC}\approx0.12$–$0.29$'
                 '\npartition degeneracy, $\\gamma$ stays low',
                 xy=(9, 0.63), xytext=(2.2, 0.70), fontsize=7.6,
                 color=js.P2, style='italic',
                 arrowprops=dict(arrowstyle='-', color='#9A9A9A', lw=0.55,
                                 shrinkA=0, shrinkB=5))
    axc.set_xlabel('selection round')
    axc.set_ylabel(r'block contrast $\gamma$')
    axc.set_title('gamma per selection round', fontsize=11)
    axc.set_ylim(0.55, 1.0)
    axc.legend(loc='lower right', fontsize=7.8, ncol=2, framealpha=0.92)
    js.despine(axc)

    # --- (d) c0 sensitivity ---
    c0ds = ['letter', 'shuttle', 'usps', 'fashion']
    c0x = [0.5, 1.067, 2.0]
    pal = ['#2166AC', '#217A5E', '#E08214', '#762A83']
    for col, ds in zip(pal, c0ds):
        ys = [c0.loc[ds, 'C_0.5'] * 100, c0.loc[ds, 'C_1.067'] * 100,
              c0.loc[ds, 'C_2.0'] * 100]
        axd.plot(c0x, ys, '-o', color=col, lw=1.6, ms=5.5, zorder=5, label=ds)
    axd.set_xticks(c0x); axd.set_xticklabels(['0.5', '1.067', '2.0'])
    axd.set_xlabel(r'$c_0$ (magnitude gate)')
    axd.set_ylabel('ACC@10% (%)')
    axd.set_title(r'$c_0$ sensitivity — perfectly flat', fontsize=11)
    axd.text(0.97, 0.06, 'spread = 0.000 pp\n(magnitude gate inert)',
             transform=axd.transAxes, ha='right', va='bottom', fontsize=8.5,
             style='italic', color='#444444',
             bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#CCCCCC',
                       alpha=0.95))
    axd.legend(loc='center right', fontsize=8.0, framealpha=0.92)
    js.despine(axd)

    for j, ax in enumerate((axa, axb, axc, axd)):
        js.panel_label(ax, f'({chr(97 + j)})', x=-0.13, y=1.06)

    fig.subplots_adjust(left=0.085, right=0.97, top=0.90, bottom=0.09,
                        hspace=0.42, wspace=0.28)
    paths = js.save_both(fig, 'fig_param_invariance')
    print('param_invariance', paths)
    return paths


if __name__ == '__main__':
    build()
