"""Merge all CSVs and generate core figures + results table (13 datasets)."""
import sys, os, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALL_DATASETS = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment',
                 'balance', 'aggregation', 'compound', 'letter', 'shuttle', 'usps', 'fashion']
DATASETS = ALL_DATASETS
METHODS = ['P2_Full', 'BADGE', 'CoreSet', 'BALD', 'QBC', 'Entropy', 'Random',
           'SSFCM', 'CEFCM', 'GRFCM', 'PLCFCMPassive',
           'P2_NoRobust', 'P2_NoExploration', 'P2_NoRedundancy', 'P2_NoFMIS', 'P2_NoBlock']
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]
COLORS = {'P2_Full': '#e74c3c', 'BADGE': '#3498db', 'CoreSet': '#2ecc71',
          'BALD': '#9b59b6', 'QBC': '#f39c12', 'Entropy': '#95a5a6',
          'Random': '#bdc3c7', 'SSFCM': '#1abc9c', 'CEFCM': '#e67e22',
          'GRFCM': '#34495e', 'PLCFCMPassive': '#7f8c8d'}


METHOD_NAME_MAP = {
    'BADGESampler': 'BADGE', 'BALDSampler': 'BALD', 'CoreSetSampler': 'CoreSet',
    'EntropySampler': 'Entropy', 'QBCSampler': 'QBC', 'RandomSampler': 'Random',
    'P2OLSAL': 'P2_Full', 'P2_OLSAL': 'P2_Full',
}


def merge_all():
    """Merge all result CSVs into all_results_merged.csv."""
    dfs = []
    all_patterns = ['master_sweep_*.csv', 'p2_tuned_sweep_*.csv', 'baseline_sweep_*.csv',
                    'rerun_missing_*.csv', 'segment_ablation_*.csv', 'segment_other_*.csv',
                    'large_sweep_*.csv', 'newdata_*.csv']
    for pat in all_patterns:
        for f in sorted(glob.glob(os.path.join('results', pat))):
            if os.path.getsize(f) == 0:
                continue
            try:
                r = pd.read_csv(f, low_memory=False)
                if 'error' in r.columns:
                    r = r[r['error'].isna() | (r['error'] == '')]
                if len(r) > 0:
                    r['method'] = r['method'].replace(METHOD_NAME_MAP)
                    dfs.append(r)
            except Exception as e:
                print(f'  Skip {f}: {e}')
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['dataset', 'method', 'eta', 'seed', 'budget_frac'],
                             keep='last')
    print(f'Merged: {len(df)} rows ({len(df) // 5} jobs)')
    print(f'Datasets ({df.dataset.nunique()}): {sorted(df.dataset.unique())}')
    print(f'Methods ({df.method.nunique()}): {sorted(df.method.unique())}')
    df.to_csv('results/all_results_merged.csv', index=False)
    return df


def fig1_learning_curves(df):
    """Main learning curves: ACC vs budget for key methods (13 datasets, 4x4 grid)."""
    key_methods = ['P2_Full', 'BADGE', 'CoreSet', 'BALD', 'QBC', 'Entropy', 'Random']
    n_ds = len(DATASETS)
    ncols = 4
    nrows = int(np.ceil(n_ds / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        sub = df[(df.dataset == ds) & (df.eta == 0.0) & (df.method.isin(key_methods))]
        n_seeds = sub['seed'].nunique()
        for m in key_methods:
            ms = sub[sub.method == m]
            if len(ms) == 0:
                continue
            agg = ms.groupby('budget_frac')['ACC'].agg(['mean', 'std'])
            ax.errorbar(agg.index, agg['mean'],
                       yerr=agg['std'] / np.sqrt(max(n_seeds, 1)),
                       marker='o', label=m, color=COLORS.get(m), capsize=3, linewidth=2)
        ax.set_title(ds, fontsize=13, fontweight='bold')
        ax.set_xlabel('Label Budget')
        ax.set_ylabel('ACC')
        ax.set_xticks(BUDGETS)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right', ncol=2)

    # Hide unused subplots
    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Fig.1: Active Learning Curves (ACC vs Label Budget)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig1_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Fig1 saved')


def fig2_noise_robustness(df):
    """Label noise robustness: ACC at 10% budget vs noise level."""
    fig, ax = plt.subplots(figsize=(10, 6))
    key_methods = ['P2_Full', 'BADGE', 'QBC', 'Entropy', 'Random']
    n_ds = df['dataset'].nunique()
    for m in key_methods:
        sub = df[(df.method == m) & (df.budget_frac == 0.10)]
        n_seeds = sub['seed'].nunique()
        agg = sub.groupby('eta')['ACC'].agg(['mean', 'std'])
        ax.errorbar(agg.index, agg['mean'],
                   yerr=agg['std'] / np.sqrt(max(n_seeds * n_ds, 1)),
                   marker='s', label=m, color=COLORS.get(m), capsize=4, linewidth=2.5)
    ax.set_xlabel('Label Noise Level', fontsize=13)
    ax.set_ylabel('ACC @ 10% Budget', fontsize=13)
    ax.set_title(f'Fig.2: Robustness to Label Noise (averaged over {n_ds} datasets)',
                 fontsize=15, fontweight='bold')
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig2_noise_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Fig2 saved')


def fig3_ablation(df):
    """Ablation study: P2 variants comparison (13 datasets)."""
    fig, ax = plt.subplots(figsize=(18, 6))
    abl_methods = ['P2_Full', 'P2_NoRobust', 'P2_NoExploration',
                   'P2_NoRedundancy', 'P2_NoFMIS', 'P2_NoBlock']
    abl_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    x = np.arange(len(DATASETS))
    width = 0.13
    for i, (m, c) in enumerate(zip(abl_methods, abl_colors)):
        sub = df[(df.method == m) & (df.eta == 0.0) & (df.budget_frac == 0.10)]
        means = [sub[sub.dataset == ds]['ACC'].mean() for ds in DATASETS]
        ax.bar(x + i * width, means, width, label=m, color=c, alpha=0.85)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(DATASETS, fontsize=10, rotation=30, ha='right')
    ax.set_ylabel('ACC @ 10% Budget', fontsize=13)
    ax.set_title('Fig.3: Ablation Study — P2 Variants Comparison',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/fig3_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Fig3 saved')


def table1_main_results(df):
    """Generate main results table (ACC@10% for all methods, all 13 datasets)."""
    rows = []
    for m in METHODS:
        row = {'Method': m}
        for ds in DATASETS:
            sub = df[(df.method == m) & (df.dataset == ds) &
                     (df.eta == 0.0) & (df.budget_frac == 0.10)]
            if len(sub) > 0:
                row[ds] = f"{sub['ACC'].mean():.3f}+/-{sub['ACC'].std() / np.sqrt(len(sub)):.3f}"
            else:
                row[ds] = 'N/A'
        # Average across datasets
        vals = []
        for ds in DATASETS:
            sub = df[(df.method == m) & (df.dataset == ds) &
                     (df.eta == 0.0) & (df.budget_frac == 0.10)]
            if len(sub) > 0:
                vals.append(sub['ACC'].mean())
        row['Average'] = f"{np.mean(vals):.3f}" if vals else 'N/A'
        rows.append(row)
    tdf = pd.DataFrame(rows)
    tdf.to_csv('results/table1_main_results.csv', index=False)
    print('Table1 saved')
    print(tdf.to_string(index=False))
    return tdf


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    df = merge_all()
    fig1_learning_curves(df)
    fig2_noise_robustness(df)
    fig3_ablation(df)
    table1_main_results(df)
    print('\n=== ALL DONE ===')
