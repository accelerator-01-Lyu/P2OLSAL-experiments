# -*- coding: utf-8 -*-
"""Sample 3: 13-panel learning-curve small multiples (landscape rectangles).

Numbers from results/all_results_merged.csv (eta=0). P2-OLSAL is drawn in brand
red on top; the six external baselines are thin/desaturated. Per-baseline error
bars are omitted in the dense small multiples (variance lives in the main
table); a thin band gives P2's seed spread. Pale vertical bands mark ONLY the
budget points at which P2-OLSAL is (tied) best on that panel, judged directly
from the CSV. The 14th cell is a wide "13-dataset mean" (equal-weighted)
summary, so the 4x4 grid has no blank cells.
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
META = {  # n, d, c
    'iris': (150, 4, 3), 'wine': (178, 13, 3), 'seeds': (210, 7, 3),
    'glass': (214, 9, 6), 'ecoli': (336, 7, 8), 'segment': (2310, 19, 7),
    'balance': (625, 4, 3), 'aggregation': (788, 2, 7), 'compound': (399, 2, 6),
    'letter': (7990, 16, 26), 'shuttle': (4997, 9, 7), 'usps': (9298, 256, 10),
    'fashion': (10000, 50, 10),
}
BASE = ['Random', 'Entropy', 'QBC', 'BALD', 'CoreSet', 'BADGE']
KEY = ['P2_Full'] + BASE
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]


def series(df, ds):
    """Return {method: mean-ACC Series indexed by budget} for one dataset."""
    sub = df[df.dataset == ds]
    out = {}
    for m in KEY:
        s = sub[sub.method == m].groupby('budget_frac').ACC.mean()
        if len(s):
            out[m] = s
    return out


def draw_panel(ax, ser, title, hby, band=True):
    for m in BASE:
        if m not in ser:
            continue
        s = ser[m]; st = js.METHOD_STYLE[m]
        (ln,) = ax.plot(s.index.values, s.values, color=st['color'], ls=st['ls'],
                        lw=1.2, marker=st['marker'], ms=3.7, mfc=st['mfc'],
                        mec='white', mew=0.5, alpha=0.78, zorder=st['zorder'],
                        label=js.METHOD_LABEL.get(m, m))
        hby.setdefault(ln.get_label(), ln)
    if 'P2_Full' in ser:
        s = ser['P2_Full']
        (pl,) = ax.plot(s.index.values, s.values, color=js.P2, lw=2.2,
                        marker='o', ms=5.1, mfc=js.P2, mec='white', mew=0.7,
                        zorder=10, label='P2-OLSAL')
        hby['P2-OLSAL'] = pl
        if band:  # only where P2 is >= best external baseline (CSV-truth)
            bdf = pd.DataFrame({m: ser[m] for m in BASE if m in ser})
            for b in BUDGETS:
                if b in s.index and b in bdf.index and s[b] + 1e-9 >= bdf.loc[b].max():
                    ax.axvspan(b - 0.0085, b + 0.0085, color=js.P2, alpha=0.08,
                               zorder=1, lw=0)
    ax.set_title(title, fontsize=9.6, loc='left', color=js.INK, pad=2)
    ax.set_xticks(BUDGETS)
    js.despine(ax)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.7, length=3)


def build_full():
    df = pd.read_csv(os.path.join(RES, 'all_results_merged.csv'), low_memory=False)
    d = df[(df.eta == 0) & (df.method.isin(KEY))]
    allser = {ds: series(d, ds) for ds in ORDER}

    fig = plt.figure(figsize=(7.2, 6.5))
    gs = fig.add_gridspec(4, 4, left=0.072, right=0.995, top=0.90,
                          bottom=0.075, wspace=0.24, hspace=0.62)
    hby = {}
    glob_lo, glob_hi = [], []
    # 12 datasets fill rows 0-2; fashion takes bottom-left; mean spans the rest
    for k, ds in enumerate(ORDER[:12]):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        n, dd, c = META[ds]
        draw_panel(ax, allser[ds], f'{ds}  (n={n}, {dd}d, {c}c)', hby)
        vals = pd.concat(allser[ds].values())
        glob_lo.append(vals.min()); glob_hi.append(vals.max())
    axf = fig.add_subplot(gs[3, 0])
    n, dd, c = META['fashion']
    draw_panel(axf, allser['fashion'], f'fashion  (n={n}, {dd}d, {c}c)', hby)
    vals = pd.concat(allser['fashion'].values())
    glob_lo.append(vals.min()); glob_hi.append(vals.max())

    # 14th panel: equal-weighted mean across the 13 datasets
    axm = fig.add_subplot(gs[3, 1:4])
    mean_ser = {}
    for m in KEY:
        frames = [allser[ds][m] for ds in ORDER if m in allser[ds]]
        if frames:
            mean_ser[m] = pd.concat(frames, axis=1).mean(axis=1)
    draw_panel(axm, mean_ser, '13-dataset mean (equal-weighted ACC)', hby, band=True)
    vals = pd.concat(mean_ser.values())
    glob_lo.append(vals.min()); glob_hi.append(vals.max())

    lo, hi = min(glob_lo), max(glob_hi)
    pad = (hi - lo) * 0.06
    axes = [fig.axes[i] for i in range(14)]
    for ax in axes:
        ax.set_ylim(lo - pad, hi + pad)
    # summary panel gets its OWN tight y range so the global ranking is legible
    mvals = pd.concat(mean_ser.values())
    mlo, mhi = mvals.min(), mvals.max()
    axm.set_ylim(mlo - (mhi - mlo) * 0.32, mhi + (mhi - mlo) * 0.32)
    axm.set_facecolor('#FAFAFA')
    # outer-only tick labels
    for k in range(12):
        ax = fig.axes[k]
        row, col = k // 4, k % 4
        if row != 2:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(['5', '10', '15', '20', '30'])
        if col != 0:
            ax.set_yticklabels([])
    axf.set_xticklabels(['5', '10', '15', '20', '30'])
    axf.set_yticklabels([])
    axm.set_xticklabels(['5', '10', '15', '20', '30'])
    axm.set_yticklabels([])
    fig.supylabel('clustering ACC', fontsize=11, x=0.015)
    fig.supxlabel('label budget (% of n)', fontsize=11, y=0.008)

    order_lab = ['P2-OLSAL'] + [js.METHOD_LABEL.get(m, m) for m in BASE]
    fin = [hby[lb] for lb in order_lab if lb in hby]
    fig.legend(fin, [h.get_label() for h in fin], loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 0.985), fontsize=9.0, frameon=False,
               columnspacing=1.25, handlelength=1.9, handletextpad=0.4)
    paths = js.save_both(fig, 'fig01_learning_curves_full')
    print('learning_curves_full', paths)
    return paths


MAIN6 = ['ecoli', 'segment', 'letter', 'shuttle', 'fashion', 'usps']


def build_main():
    """Six key-dataset panels for the main text (full 13-panel version is Fig. A1)."""
    df = pd.read_csv(os.path.join(RES, 'all_results_merged.csv'), low_memory=False)
    d = df[(df.eta == 0) & (df.method.isin(KEY))]
    allser = {ds: series(d, ds) for ds in MAIN6}

    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.995, top=0.86,
                          bottom=0.115, wspace=0.20, hspace=0.42)
    hby = {}
    axes = [fig.add_subplot(gs[k // 3, k % 3]) for k in range(6)]
    for k, ds in enumerate(MAIN6):
        ax = axes[k]
        n, dd, c = META[ds]
        draw_panel(ax, allser[ds], f'{ds}  (n={n}, {dd}d, {c}c)', hby)
        ax.set_xticklabels(['5', '10', '15', '20', '30'])
        ax.set_xlabel('')
        ax.grid(axis='y', color=js.GRID if hasattr(js, 'GRID') else '#E6E6E6',
                lw=0.7, zorder=0)
        ax.tick_params(labelsize=9.2)
        ax.set_title(f'{ds}  (n={n}, {dd}d, {c}c)', fontsize=10.6,
                     loc='left', color=js.INK, pad=3)
    for k, ax in enumerate(axes):
        if k % 3 != 0:
            ax.tick_params(axis='y', labelleft=False)
    fig.supylabel('clustering ACC', fontsize=11.5, x=0.012)
    fig.supxlabel('label budget (% of n)', fontsize=11.5, y=0.015)
    order_lab = ['P2-OLSAL'] + [js.METHOD_LABEL.get(m, m) for m in BASE]
    fin = [hby[lb] for lb in order_lab if lb in hby]
    fig.legend(fin, [h.get_label() for h in fin], loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 1.02), fontsize=9.4, frameon=False,
               columnspacing=1.2, handlelength=1.9, handletextpad=0.4)
    paths = js.save_both(fig, 'fig01_learning_curves')
    print('learning_curves_main', paths)
    return paths


if __name__ == '__main__':
    build()
