# -*- coding: utf-8 -*-
"""Fig. 7 (two panels):
 (a) non-identifiability scatter (d_eff vs gamma, size=PC, color=D-A);
 (b) radar comparison of ecoli vs shuttle over the fixed scalar family S,
     showing near-identical statistics but opposite cross-block effects.
Numbers from results/alpha_structure.csv and results/factorial_cell_acc.csv.
"""
import os, sys
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments\figures')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import _journal_style as js

RES = r'D:\P2OLSAL_FSS\experiments\results'
ORDER = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment', 'balance',
         'aggregation', 'compound', 'letter', 'shuttle', 'usps', 'fashion']
KEY_PAIR = ('ecoli', 'shuttle')

LABEL_XY = {
    'fashion':     (1.30, 0.683), 'balance': (1.55, 0.812), 'wine': (1.92, 0.912),
    'seeds':       (2.30, 1.000), 'iris':    (3.32, 1.005), 'usps': (3.30, 0.832),
    'glass':       (3.42, 0.916), 'shuttle': (3.66, 0.972), 'letter': (4.52, 0.772),
    'segment':     (4.72, 0.998), 'ecoli':   (5.24, 0.866),
    'compound':    (5.40, 0.956), 'aggregation': (6.46, 0.992),
}
HA = {ds: 'center' for ds in ORDER}
HA.update({'fashion': 'left', 'balance': 'left', 'wine': 'left',
           'usps': 'left', 'letter': 'left', 'ecoli': 'left'})


def build():
    st = pd.read_csv(os.path.join(RES, 'alpha_structure.csv')).set_index('ds')
    cell = pd.read_csv(os.path.join(RES, 'factorial_cell_acc.csv')).set_index('dataset')
    df = pd.DataFrame({
        'd_eff': st['d_eff'], 'gamma': st['gamma'], 'PC': st['PC'],
        'dpp': 100.0 * (cell['D'] - cell['A']),
    }).loc[ORDER]

    fig = plt.figure(figsize=(js.TEXTWIDTH_IN, 5.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], left=0.055,
                          right=0.975, top=0.885, bottom=0.10, wspace=0.20)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1], polar=True)

    # ---------- (a) scatter ----------
    norm, cmap = js.diverging_norm(df.dpp.values)
    ax.scatter(df.d_eff, df.gamma, s=110 + 430 * df.PC, c=df.dpp, cmap=cmap,
               norm=norm, edgecolor='white', linewidth=1.0, alpha=0.92, zorder=4)
    for ds, col in [('ecoli', js.P2), ('shuttle', js.BASELINE_COLORS['BADGE'])]:
        ax.scatter([df.loc[ds, 'd_eff']], [df.loc[ds, 'gamma']], s=720,
                   facecolor='none', edgecolor=col, linewidth=2.3, zorder=5)

    for ds, r in df.iterrows():
        tx, ty = LABEL_XY[ds]
        bold = ds in KEY_PAIR
        ax.annotate(ds, xy=(r.d_eff, r.gamma), xytext=(tx, ty),
                    textcoords='data', ha=HA[ds], va='center',
                    fontsize=8.6, fontweight='bold' if bold else 'normal',
                    color=js.INK, zorder=8,
                    arrowprops=dict(arrowstyle='-', color='#9A9A9A', lw=0.6,
                                    shrinkA=0, shrinkB=7,
                                    connectionstyle='arc3,rad=0.0'))

    ax.annotate('statistics overlap,\nopposite sign',
                xy=(4.80, 0.905), xytext=(5.55, 0.798), textcoords='data',
                ha='center', va='center', fontsize=9.0, fontweight='bold',
                color=js.INK, zorder=9,
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.3),
                bbox=dict(boxstyle='round,pad=0.35', fc='#FFF6E5',
                          ec='#D4A017', lw=0.9))

    tbl = ('ecoli:  $\\gamma=0.906$, $d_{\\rm eff}=4.88$, PC=0.68, $\\Delta=-2.6$pp\n'
           'shuttle: $\\gamma=0.927$, $d_{\\rm eff}=3.96$, PC=0.77, $\\Delta=+7.9$pp')
    ax.text(0.985, 0.03, tbl, transform=ax.transAxes, fontsize=8.4,
            va='bottom', ha='right', zorder=7,
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='#BBBBBB', lw=0.9))

    cb = fig.colorbar(ax.collections[0], ax=ax, pad=0.03, shrink=0.80, fraction=0.06)
    cb.set_label(r'$\Delta$ACC, remove cross-block (pp)', fontsize=9.6)
    cb.outline.set_linewidth(0.8)
    ax.set_xlabel(r'effective dimension $d_{\mathrm{eff}} = \mathrm{PR}(R)$', fontsize=10.2)
    ax.set_ylabel(r'block contrast $\gamma$ (current membership $U$)', fontsize=10.2)
    ax.set_title('(a) Membership statistics do not predict the sign', loc='left',
                 fontsize=11.0, pad=8)
    ax.set_ylim(0.62, 1.035)
    ax.set_xlim(0.8, 7.15)
    js.despine(ax)
    ax.grid(False)

    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#BBBBBB',
                      markeredgecolor='white', markersize=s, label=lbl)
               for s, lbl in [(7, 'PC=0.2'), (12, 'PC=0.6'), (18, 'PC=1.0')]]
    ax.legend(handles=handles, title='size $\\propto$ PC',
              bbox_to_anchor=(0.012, 0.988), loc='upper left',
              fontsize=8.2, title_fontsize=8.8, ncol=1, framealpha=0.95,
              labelspacing=0.6, borderpad=0.55, handletextpad=0.4)

    # ---------- (b) radar: ecoli vs shuttle over S ----------
    axes_labels = [r'$\gamma$', 'PC', r'$H_{\mathrm{norm}}$', r'$r_{\mathrm{within}}$',
                   r'$r_{\mathrm{cross}}$', r'$d_{\mathrm{eff}}/c$']

    def vec(ds):
        r = st.loc[ds]
        return [r['gamma'], r['PC'], r['H_norm'], r['r_within'], r['r_cross'],
                r['d_eff'] / r['n'] if False else r['d_eff'] / int(r['c'])]

    vals = {'ecoli': vec('ecoli'), 'shuttle': vec('shuttle')}
    ang = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()
    ang_closed = ang + ang[:1]
    styles = {'ecoli': (js.P2, '-'), 'shuttle': (js.BASELINE_COLORS['BADGE'], '--')}
    for ds, vv in vals.items():
        c, ls = styles[ds]
        closed = vv + vv[:1]
        axr.plot(ang_closed, closed, color=c, lw=2.1, ls=ls,
                 label=f'{ds} ({df.loc[ds, "dpp"]:+.1f}pp)', zorder=4)
        axr.fill(ang_closed, closed, color=c, alpha=0.16, zorder=3)
    axr.set_xticks(ang)
    axr.set_xticklabels(axes_labels, fontsize=9.0)
    axr.set_ylim(0, 1)
    axr.set_yticks([0.25, 0.5, 0.75])
    axr.set_yticklabels(['.25', '.5', '.75'], fontsize=7.4, color='#888888')
    axr.tick_params(pad=2)
    axr.set_title('(b) ecoli vs shuttle: statistics overlap,\nopposite cross-block effect',
                  fontsize=10.6, pad=14)
    axr.legend(loc='lower center', bbox_to_anchor=(0.5, -0.16), fontsize=8.6,
               ncol=1, framealpha=0.95, handlelength=2.2, borderpad=0.5)
    axr.spines['polar'].set_color('#CCCCCC')
    axr.grid(color='#DDDDDD', lw=0.7)

    paths = js.save_both(fig, 'fig_nonidentifiability')
    print('nonidentifiability', paths)
    return paths


if __name__ == '__main__':
    build()
