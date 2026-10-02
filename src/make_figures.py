"""Generate core figures from sweep CSV results.

Figures:
1. Learning curves (ACC/NMI/ARI vs budget) per dataset
2. Label noise robustness (ACC vs eta)
3. Misfit proxy validation (delta_mis vs zeta_t scatter)
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

MAIN_METHODS = ['P2_Full', 'BADGE', 'CoreSet', 'BALD', 'QBC', 'Entropy', 'Random']
COLORS = {'P2_Full': '#d62728', 'BADGE': '#1f77b4', 'CoreSet': '#2ca02c',
          'BALD': '#ff7f0e', 'QBC': '#9467bd', 'Entropy': '#8c564b',
          'Random': '#7f7f7f'}
MARKERS = {'P2_Full': 'o', 'BADGE': 's', 'CoreSet': '^', 'BALD': 'D',
           'QBC': 'v', 'Entropy': 'x', 'Random': '+'}


def load_all_results():
    """Load and concatenate all sweep CSVs."""
    dfs = []
    for pat in ['master_sweep_*.csv', 'large_sweep_*.csv', 'newdata_sweep_*.csv']:
        for f in glob.glob(os.path.join(RES, pat)):
            try:
                df = pd.read_csv(f)
                if len(df) > 0 and 'ACC' in df.columns:
                    dfs.append(df)
            except Exception:
                pass
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def fig_learning_curves(df, datasets=None, metric='ACC'):
    """Figure 1: Learning curves per dataset."""
    if datasets is None:
        datasets = sorted(df['dataset'].unique())
    n_ds = len(datasets)
    ncols = min(3, n_ds)
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        dsub = df[(df['dataset'] == ds) & (df['eta'] == 0.0)]
        for m in MAIN_METHODS:
            dm = dsub[dsub['method'] == m]
            if len(dm) == 0:
                continue
            agg = dm.groupby('budget_frac')[metric].agg(['mean', 'std'])
            ax.errorbar(agg.index, agg['mean'], yerr=agg['std'],
                       label=m, color=COLORS.get(m, None),
                       marker=MARKERS.get(m, 'o'), markersize=4,
                       capsize=2, linewidth=1.5)
        ax.set_title(ds, fontsize=12)
        ax.set_xlabel('Annotation budget')
        ax.set_ylabel(metric)
        ax.set_xlim(0.04, 0.32)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc='lower right')
    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIG, f'fig1_learning_curves_{metric}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {out}')


def fig_noise_robustness(df, datasets=None):
    """Figure 2: ACC vs noise rate at budget=20%."""
    if datasets is None:
        datasets = sorted(df['dataset'].unique())
    n_ds = len(datasets)
    ncols = min(3, n_ds)
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
                       label=m, color=COLORS.get(m, None),
                       marker=MARKERS.get(m, 'o'), markersize=5,
                       capsize=3, linewidth=2)
        ax.set_title(ds, fontsize=12)
        ax.set_xlabel('Label noise rate')
        ax.set_ylabel('ACC @ 20% budget')
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9)
    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIG, 'fig2_noise_robustness.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {out}')


def fig_misfit_proxy(df):
    """Figure 5: delta_mis_proxy vs zeta_t scatter."""
    dsub = df[(df['method'] == 'P2_Full') & (df['eta'] == 0.0)]
    if len(dsub) == 0:
        print('no P2_Full data for misfit proxy')
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for ds in sorted(dsub['dataset'].unique()):
        dd = dsub[dsub['dataset'] == ds]
        ax.scatter(dd['delta_mis_proxy'], dd['zeta_t'], alpha=0.3, s=10, label=ds)
    ax.set_xlabel(r'$\hat{\delta}_{\mathrm{mis}}$ (misfit proxy)')
    ax.set_ylabel(r'$\zeta_t$ (realized expert-model disagreement)')
    ax.set_title('Misfit proxy validation')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    # Add correlation line
    valid = dsub.dropna(subset=['delta_mis_proxy', 'zeta_t'])
    if len(valid) > 10:
        corr = valid['delta_mis_proxy'].corr(valid['zeta_t'])
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}',
                transform=ax.transAxes, fontsize=11, va='top')
    plt.tight_layout()
    out = os.path.join(FIG, 'fig5_misfit_proxy.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {out}')


def fig_ablation(df, datasets=None):
    """Ablation: P2 variants vs Full at budget=20%, eta=0."""
    if datasets is None:
        datasets = sorted(df['dataset'].unique())
    variants = ['P2_Full', 'P2_NoRobust', 'P2_NoExploration', 'P2_NoRedundancy',
                'P2_NoFMIS', 'P2_NoBlock']
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    width = 0.13
    for i, v in enumerate(variants):
        means = []
        for ds in datasets:
            dm = df[(df['dataset'] == ds) & (df['method'] == v) &
                    (df['eta'] == 0.0) & (df['budget_frac'] == 0.20)]
            means.append(dm['ACC'].mean() if len(dm) > 0 else 0)
        ax.bar(x + i * width, means, width, label=v)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel('ACC @ 20% budget')
    ax.set_title('Ablation study')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out = os.path.join(FIG, 'fig_ablation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {out}')


if __name__ == '__main__':
    df = load_all_results()
    if df is None:
        print('No results found')
    else:
        print(f'Loaded {len(df)} rows, {df["dataset"].nunique()} datasets')
        print(f'Methods: {sorted(df["method"].unique())}')
        for metric in ['ACC', 'NMI', 'ARI']:
            fig_learning_curves(df, metric=metric)
        fig_noise_robustness(df)
        fig_misfit_proxy(df)
        fig_ablation(df)
        print('All figures generated.')
