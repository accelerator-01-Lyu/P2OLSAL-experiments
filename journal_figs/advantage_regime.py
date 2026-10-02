# -*- coding: utf-8 -*-
"""Core Figure: competitiveness-regime map.

Delta (P2-OLSAL minus the best of all external baselines, ACC@10%, eta=0) vs
block contrast gamma. Honest: ALL 13 points are <= 0 (ecoli is exactly 0 =
tied-best, compound ~ 0); nothing is shifted above zero. The message is the
positive TREND -- competitiveness grows monotonically with block contrast --
"a regime, not dominance". OLS line + bootstrapped 95% CI band.
Data: results/fig2_advantage_data.csv.
"""
import os, sys
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments\figures')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _journal_style as js

RES = r'D:\P2OLSAL_FSS\experiments\results'
EMPH = {'ecoli', 'compound', 'letter', 'fashion'}
LABEL_XY = {  # data coords (gamma, pp)
    'fashion':     (0.650, -11.35),
    'letter':      (0.792, -12.25),
    'balance':     (0.855, -0.10),
    'usps':        (0.798, -3.95),
    'glass':       (0.872, -5.05),
    'segment':     (0.905, -7.55),
    'ecoli':       (0.900, 1.35),
    'shuttle':     (0.948, -3.55),
    'wine':        (0.945, 0.55),
    'seeds':       (0.956, -5.85),
    'iris':        (0.972, -1.35),
    'compound':    (0.985, 1.35),
    'aggregation': (0.995, -1.15),
}
HA = {'fashion': 'left', 'letter': 'left', 'balance': 'left', 'usps': 'right',
      'glass': 'right', 'segment': 'center', 'ecoli': 'center',
      'shuttle': 'left', 'wine': 'left', 'seeds': 'center', 'iris': 'right',
      'compound': 'center', 'aggregation': 'center'}


def build():
    from scipy import stats
    d = pd.read_csv(os.path.join(RES, 'fig2_advantage_data.csv'))
    x = d.gamma.values
    y = 100.0 * d.delta.values
    r, p = stats.pearsonr(x, y)
    rs, _ = stats.spearmanr(x, y)
    slope, icept = np.polyfit(x, y, 1)

    # bootstrap 95% CI of the regression line
    rng = np.random.default_rng(20261001)
    grid = np.linspace(x.min(), x.max(), 200)
    boots = np.empty((10000, grid.size))
    n = len(x)
    for b in range(boots.shape[0]):
        idx = rng.integers(0, n, n)
        s, i = np.polyfit(x[idx], y[idx], 1)
        boots[b] = s * grid + i
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(js.TEXTWIDTH_IN, 4.7))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.12)
    # win / gap background tint
    ax.axhspan(0, 3, color=js.BASELINE_COLORS['BALD'], alpha=0.05, zorder=0)
    ax.axhspan(-14, 0, color=js.P2, alpha=0.04, zorder=0)
    ax.text(0.985, 0.965, 'tied / lead zone ($\\Delta\\geq0$)', ha='right',
            va='top', transform=ax.transAxes, fontsize=8.6,
            color=js.BASELINE_COLORS['BALD'], style='italic')
    ax.text(0.015, 0.05, 'gap to best external baseline ($\\Delta<0$)',
            ha='left', va='bottom', transform=ax.transAxes, fontsize=8.6,
            color=js.P2, style='italic')

    ax.axhline(0, color='#777777', lw=1.1, ls='-', zorder=2)
    ax.fill_between(grid, lo, hi, color=js.P2, alpha=0.13, linewidth=0, zorder=3)
    ax.plot(grid, slope * grid + icept, color=js.P2, lw=2.3, zorder=4,
            label='OLS trend (bootstrap 95% CI)')

    for ds, xi, yi in zip(d.dataset, x, y):
        emph = ds in EMPH
        ax.scatter([xi], [yi], s=105 if emph else 78,
                   color=js.P2 if emph else '#E08E92',
                   edgecolor='white', linewidth=1.1, zorder=6,
                   alpha=1.0 if emph else 0.85)
        tx, ty = LABEL_XY[ds]
        ax.annotate(ds, xy=(xi, yi), xytext=(tx, ty), textcoords='data',
                    ha=HA[ds], va='center', fontsize=8.8,
                    fontweight='bold' if emph else 'normal', color=js.INK,
                    zorder=8,
                    arrowprops=dict(arrowstyle='-', color='#9A9A9A', lw=0.55,
                                    shrinkA=0, shrinkB=5))

    txt = (f'Pearson $r={r:.2f}$, $p={p:.3f}$\n'
           f'Spearman $\\rho={rs:.2f}$\n'
           f'slope $=+{slope:.0f}$ pp per unit $\\gamma$')
    ax.text(0.02, 0.965, txt, transform=ax.transAxes, fontsize=9.6,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='none', alpha=0.92))

    ax.set_xlabel(r'block contrast $\gamma$ (current membership $U$)')
    ax.set_ylabel(r'$\Delta$ACC vs. best external baseline (pp)')
    ax.set_title('P2-OLSAL competitiveness increases with block contrast — a regime, not dominance',
                 fontsize=12, pad=8)
    ax.set_xlim(0.64, 1.03)
    ax.set_ylim(-13.6, 2.4)
    js.despine(ax)
    ax.grid(axis='y')
    ax.legend(loc='lower right', fontsize=9.0, framealpha=0.92)
    paths = js.save_both(fig, 'fig02_advantage_regime')
    print('advantage', paths)
    return paths


if __name__ == '__main__':
    build()
