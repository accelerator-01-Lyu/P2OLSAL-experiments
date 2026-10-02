# -*- coding: utf-8 -*-
"""Mechanism figure 2: 2x2 factorial upgrade.

(a) Per-dataset ACC@10% grouped bars: cells A / D / E / B. A = P2 (red);
    D and B are the SAME blue-family two shades (D == B numerically, i.e. the
    block penalty removal and the full penalty removal coincide); E = orange.
(b) Paired lifts D-A and E-A per dataset: point + 95% CI. The two datasets
    where removing the cross-block penalty significantly HURTS (ecoli,
    segment) are ringed red and starred.
Data: results/factorial_cell_acc.csv, results/factorial_paired.csv.
"""
import os, sys
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments\figures')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _journal_style as js

RES = r'D:\P2OLSAL_FSS\experiments\results'
ORDER = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment', 'balance',
         'aggregation', 'compound', 'letter', 'shuttle', 'usps', 'fashion']
BLUE_D = js.BASELINE_COLORS['BADGE']   # #2166AC deep
BLUE_B = '#6BAED6'                      # lighter, same hue (D == B)
ORANGE = js.BASELINE_COLORS['QBC']     # #E08214


def build():
    cell = pd.read_csv(os.path.join(RES, 'factorial_cell_acc.csv'))
    cell = cell.set_index('dataset').loc[ORDER]
    pair = pd.read_csv(os.path.join(RES, 'factorial_paired.csv'))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(js.TEXTWIDTH_IN, 3.9))

    # --- (a) grouped bars ---
    x = np.arange(len(ORDER))
    w = 0.20
    cols = {'A': js.P2, 'D': BLUE_D, 'E': ORANGE, 'B': BLUE_B}
    offs = {'A': -1.5 * w, 'D': -0.5 * w, 'E': 0.5 * w, 'B': 1.5 * w}
    for c in ['A', 'D', 'E', 'B']:
        axA.bar(x + offs[c], cell[c].values * 100.0, width=w * 0.92,
                color=cols[c], edgecolor='white', linewidth=0.4, zorder=3,
                label={'A': 'A (full penalty)', 'D': 'D (drop cross-block)',
                       'E': 'E (drop within-block)', 'B': 'B (drop all)'}[c])
    axA.set_xticks(x)
    axA.set_xticklabels(ORDER, rotation=45, ha='right')
    axA.set_ylabel('ACC@10% (%)')
    axA.set_title('Factorial cells', fontsize=11.5)
    axA.legend(loc='lower right', fontsize=7.6, ncol=1, framealpha=0.92)
    js.despine(axA)

    # --- (b) paired diffs ---
    pD = pair[pair.cell == 'D'].set_index('dataset').loc[ORDER]
    pE = pair[pair.cell == 'E'].set_index('dataset').loc[ORDER]
    yD = pD['diff'].values * 100.0
    yE = pE['diff'].values * 100.0
    eD = np.vstack([(yD - pD.ciL.values * 100.0), (pD.ciH.values * 100.0 - yD)])
    eE = np.vstack([(yE - pE.ciL.values * 100.0), (pE.ciH.values * 100.0 - yE)])

    axB.axhline(0, color='#777777', lw=1.1, zorder=2)
    axB.errorbar(x - 0.18, yD, yerr=eD, fmt='o', ms=4.5, color=BLUE_D,
                 ecolor=BLUE_D, elinewidth=1.1, capsize=2.5, ls='none',
                 zorder=5, label=r'$D-A$')
    axB.errorbar(x + 0.18, yE, yerr=eE, fmt='s', ms=4.2, color=ORANGE,
                 ecolor=ORANGE, elinewidth=1.1, capsize=2.5, ls='none',
                 zorder=5, label=r'$E-A$')
    # significantly-negative D cells: ecoli, segment
    for name in ['ecoli', 'segment']:
        i = ORDER.index(name)
        axB.scatter([i - 0.18], [yD[i], ], s=150, facecolor='none',
                    edgecolor=js.P2, linewidth=1.6, zorder=7)
        axB.annotate('*', xy=(i - 0.18, yD[i]), xytext=(i - 0.18, yD[i] - 2.2),
                     ha='center', fontsize=13, color=js.P2, fontweight='bold')
    axB.set_xticks(x)
    axB.set_xticklabels(ORDER, rotation=45, ha='right')
    axB.set_ylabel(r'$\Delta$ACC@10% vs. $A$ (pp)')
    axB.set_title('Paired lift over $A$', fontsize=11.5)
    axB.legend(loc='lower left', fontsize=8.5, framealpha=0.92)
    axB.text(0.985, 0.965,
             r'$D \equiv B$ on every dataset  $\Rightarrow$  gain is entirely '
             'from removing the\ncross-block penalty',
             transform=axB.transAxes, ha='right', va='top', fontsize=7.8,
             style='italic', color='#444444',
             bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#CCCCCC',
                       alpha=0.95))
    js.despine(axB)

    for j, ax in enumerate((axA, axB)):
        js.panel_label(ax, f'({chr(97 + j)})', x=-0.13, y=1.06)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.22,
                        wspace=0.28)
    paths = js.save_both(fig, 'fig_alpha_factorial')
    print('alpha_factorial', paths)
    return paths


if __name__ == '__main__':
    build()
