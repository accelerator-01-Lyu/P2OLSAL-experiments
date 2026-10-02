# -*- coding: utf-8 -*-
"""Regenerate fig2_noise_robustness.png as 6-dataset subplots, dpi=300."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = r'D:\P2OLSAL_FSS\experiments\results'
FIG = r'D:\P2OLSAL_FSS\experiments\figures'

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
})

COLORS = {'P2_Full': '#E74C3C', 'BADGE': '#3498DB', 'QBC': '#F39C12',
          'Random': '#95A5A6'}
MARKERS = {'P2_Full': 'o', 'BADGE': 's', 'QBC': 'v', 'Random': '^'}


def main():
    df = pd.read_csv(os.path.join(RES, 'all_results_merged.csv'))
    datasets = sorted(df['dataset'].unique())
    n_ds = len(datasets)
    ncols = 3
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        dsub = df[(df['dataset'] == ds) & (df['budget_frac'] == 0.20)]
        for m in ['P2_Full', 'BADGE', 'QBC', 'Random']:
            dm = dsub[dsub['method'] == m]
            if len(dm) == 0:
                continue
            agg = dm.groupby('eta')['ACC'].agg(['mean', 'std'])
            ax.errorbar(agg.index, agg['mean'], yerr=agg['std'],
                        label=m, color=COLORS.get(m),
                        marker=MARKERS.get(m, 'o'), markersize=5,
                        capsize=3, linewidth=2)
        ax.set_title(ds, fontsize=14)
        ax.set_xlabel('Label noise rate')
        ax.set_ylabel('ACC @ 20% budget')
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=11)
    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle('Fig.7: Robustness to Label Noise (per-dataset)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, 'fig2_noise_robustness.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('saved', out, os.path.getsize(out), 'bytes')


if __name__ == '__main__':
    main()
