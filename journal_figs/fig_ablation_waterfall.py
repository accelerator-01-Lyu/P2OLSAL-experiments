# -*- coding: utf-8 -*-
"""Mechanism figure 3: component ablation waterfall.

From results/all_results_merged.csv (eta=0, budget_frac=0.1): seed-average ACC
per (method, dataset), then equal-weight over the 13 datasets. Baseline =
P2_Full; each floating bar is mean(P2_NoX) - mean(P2_Full) (usually <= 0).
Negative bars red, near-zero gray, positive green; dashed connectors track the
running level.
"""
import os, sys
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments\figures')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _journal_style as js

RES = r'D:\P2OLSAL_FSS\experiments\results'
COMP = [('NoRobust', r'$-$ robust'),
        ('NoRedundancy', r'$-$ redundancy'),
        ('NoBlock', r'$-$ block'),
        ('NoExploration', r'$-$ exploration'),
        ('NoFMIS', r'$-$ FMIS')]


def build():
    d = pd.read_csv(os.path.join(RES, 'all_results_merged.csv'))
    sub = d[(d.eta == 0.0) & (d.budget_frac == 0.1)].copy()
    methods = ['P2_Full'] + ['P2_' + k for k, _ in COMP]
    sub = sub[sub.method.isin(methods)]
    # mean over seeds per (method, dataset), then equal-weight over 13 datasets
    g = sub.groupby(['method', 'dataset']).ACC.mean().unstack('method')
    m = g.mean(axis=0) * 100.0
    full = m['P2_Full']
    deltas = [(m['P2_' + k] - full) for k, _ in COMP]

    # assemble waterfall levels
    names = ['full'] + [lab for _, lab in COMP] + ['final']
    bottoms, heights, colors, dlab = [], [], [], []
    running = full
    # bar 0: full solid
    bottoms.append(0.0); heights.append(full)
    colors.append('#3A3A3A'); dlab.append(f'{full:.3f}')
    for (k, lab), dl in zip(COMP, deltas):
        new = running + dl
        if dl < -0.05:
            col = js.P2
        elif dl > 0.05:
            col = js.BASELINE_COLORS['BALD']
        else:
            col = js.BASELINE_COLORS['Random']
        bottoms.append(min(running, new)); heights.append(abs(dl) if abs(dl) > 1e-9 else 0.15)
        colors.append(col)
        dlab.append(f'{dl:+.2f}')
        running = new
    # final summary bar
    bottoms.append(0.0); heights.append(running)
    colors.append('#3A3A3A'); dlab.append(f'{running:.3f}')

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(js.TEXTWIDTH_IN, 4.4))
    bars = ax.bar(x, heights, bottom=bottoms, width=0.62, color=colors,
                  edgecolor='white', linewidth=0.8, zorder=3)
    # dashed connectors at cumulative levels
    running = full
    ax.plot([-0.31, 0.31], [full, full], ls='--', color='#888888', lw=1.0, zorder=2)
    for i, dl in enumerate(deltas, start=1):
        new = running + dl
        ax.plot([i - 0.31, i - 0.31], [running, new], ls='--', color='#888888', lw=1.0, zorder=2)
        ax.plot([i - 0.31, i + 0.31], [new, new], ls='--', color='#888888', lw=1.0, zorder=2)
        running = new
    ax.plot([len(names) - 1 - 0.31, len(names) - 1 + 0.31], [running, running],
            ls='--', color='#888888', lw=1.0, zorder=2)

    # value labels
    for i, (b, h, lab) in enumerate(zip(bottoms, heights, dlab)):
        top = b + h
        if i == 0:
            ax.text(i, top + 0.4, lab, ha='center', va='bottom', fontsize=9.5,
                    fontweight='bold', color='#333333')
        elif i == len(names) - 1:
            ax.text(i, top + 0.4, lab, ha='center', va='bottom', fontsize=9.5,
                    fontweight='bold', color='#333333')
        else:
            ax.text(i, top + 0.45, lab, ha='center', va='bottom', fontsize=9.5,
                    color=colors[i], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel('mean ACC@10% (%)')
    ax.set_ylim(0, max(full, running) * 1.12)
    ax.set_title('Component ablation — 13-dataset mean ACC@10% ($\\eta=0$)',
                 fontsize=12, pad=8)
    js.despine(ax)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.12)
    paths = js.save_both(fig, 'fig_ablation_waterfall')
    print('ablation_waterfall', paths)
    print(f'  full={full:.4f}')
    for (k, _), dl in zip(COMP, deltas):
        print(f'  {k:16s} dDelta={dl:+.3f} pp')
    return paths


if __name__ == '__main__':
    build()
