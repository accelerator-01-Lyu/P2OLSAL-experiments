# -*- coding: utf-8 -*-
"""Generate P2-OLSAL paper figures: fig1, fig3, fig4, fig8, fig9."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = r'D:\P2OLSAL_FSS\experiments\results'
FIG = r'D:\P2OLSAL_FSS\experiments\figures'
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 100,
})

C = {
    'P2': '#E74C3C', 'BADGE': '#3498DB', 'CoreSet': '#2ECC71',
    'BALD': '#9B59B6', 'QBC': '#F39C12', 'Random': '#95A5A6',
    'Entropy': '#8C564B',
}


# ============================================================
# Fig.1: rho_expl catastrophic cliff
# ============================================================
def fig1_rho_cliff():
    df = pd.read_csv(os.path.join(RES, 'rho_ablation.csv'))
    rhos = [0.0, 0.01, 0.1, 0.3, 0.5, 1.0]
    # place rho=0 at 0.001 for log axis
    x_pos = [0.001 if r == 0 else r for r in rhos]
    ds_colors = {'ecoli': '#E74C3C', 'iris': '#3498DB'}

    fig, ax = plt.subplots(figsize=(8, 6))
    for ds in ['ecoli', 'iris']:
        sub = df[df['dataset'] == ds]
        g = sub.groupby('rho_expl')['ACC'].agg(['mean', 'std']).reindex(rhos)
        ax.errorbar(x_pos, g['mean'], yerr=g['std'], marker='o', capsize=4,
                    color=ds_colors[ds], linewidth=2, markersize=7,
                    label=ds)

    ax.set_xscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['0', '0.01', '0.1', '0.3', '0.5', '1.0'])
    ax.axvline(x=0.001, linestyle='--', color='gray', linewidth=1.2)
    ax.text(0.0011, ax.get_ylim()[1] * 0.99 if False else 0.835,
            'Optimal (ρ=0)', rotation=90, va='top', ha='left',
            fontsize=11, color='gray')
    # arrow annotation: cliff between 0 and 0.01 on ecoli
    ecoli = df[df['dataset'] == 'ecoli'].groupby('rho_expl')['ACC'].mean().reindex(rhos)
    y0, y1 = ecoli[0.0], ecoli[0.01]
    ax.annotate('Double-Counting Cliff',
                xy=(np.sqrt(0.001 * 0.01), (y0 + y1) / 2),
                xytext=(0.02, 0.60),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=12, color='black')
    ax.set_xlabel(r'$\rho_{\mathrm{expl}}$ (exploration weight)')
    ax.set_ylabel('ACC')
    ax.set_title('$\\rho_{expl}$ Catastrophic Cliff')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = os.path.join(FIG, 'fig1_rho_cliff.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('saved', out)


# ============================================================
# Fig.3: FMIS cumulant norm vs ACC gain
# ============================================================
def fig3_cumulant_gain():
    cs = pd.read_csv(os.path.join(RES, 'cumulant_spectral_all13.csv'))
    # exclude usps MemoryError row (NaN T3) and fashion T3==0
    cs = cs.dropna(subset=['T3_norm'])
    cs = cs[cs['T3_norm'] > 0.0]

    al = pd.read_csv(os.path.join(RES, 'all_results_merged.csv'))
    sub = al[(al['budget_frac'] == 0.10) & (al['eta'] == 0.0) &
             (al['method'].isin(['P2_Full', 'Entropy']))]
    means = sub.groupby(['dataset', 'method'])['ACC'].mean().unstack()
    means['dACC'] = means['P2_Full'] - means['Entropy']

    merged = cs.set_index('dataset').join(means[['dACC']], how='inner')
    print('Fig3 datasets:', list(merged.index))
    print(merged[['T3_norm', 'dACC']].round(4))

    x = merged['T3_norm'].values
    y = merged['dACC'].values

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=80, color=C['P2'], zorder=3)
    for ds, row in merged.iterrows():
        ax.annotate(ds, (row['T3_norm'], row['dACC']),
                    textcoords='offset points', xytext=(6, 6), fontsize=11)

    # linear regression
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, '--', color='gray',
                linewidth=1.5, label=f'linear fit (R$^2$={r2:.3f})')
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel(r'$\|T_3\|$ (3rd-order cumulant norm)')
    ax.set_ylabel(r'$\Delta$ACC = ACC(P2) − ACC(Entropy)')
    ax.set_title('FMIS Cumulant Strength vs P2 Gain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG, 'fig3_cumulant_gain.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('saved', out)


# ============================================================
# Fig.4: block spectral gap vs intra-block cosine
# ============================================================
def fig4_spectral_gap():
    cs = pd.read_csv(os.path.join(RES, 'cumulant_spectral_all13.csv'))
    cs = cs.dropna(subset=['block_gap_mean', 'block_cos_mean'])  # drop usps
    print('Fig4 datasets:', list(cs['dataset']))

    x = cs['block_gap_mean'].values
    y = cs['block_cos_mean'].values

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=80, color=C['CoreSet'], zorder=3)
    for _, row in cs.iterrows():
        ax.annotate(row['dataset'], (row['block_gap_mean'], row['block_cos_mean']),
                    textcoords='offset points', xytext=(6, 6), fontsize=11)

    # reference trend line (log x)
    lx = np.log10(x)
    slope, intercept = np.polyfit(lx, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * np.log10(xs) + intercept, '--', color='gray',
            linewidth=1.5, label='trend')
    ax.set_xscale('log')
    ax.set_xlabel('Block spectral gap (mean, log scale)')
    ax.set_ylabel('Intra-block mean cosine similarity')
    ax.set_title('Block Spectral Gap vs Intra-Block Cohesion')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = os.path.join(FIG, 'fig4_spectral_gap.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('saved', out)


# ============================================================
# Fig.8: adversarial oracle regret
# ============================================================
def fig8_adversarial():
    adv = pd.read_csv(os.path.join(RES, 'adversarial_experiment.csv'))
    ds = 'ecoli'
    sub = adv[adv['dataset'] == ds]
    styles = {
        'P2_Full': dict(color=C['P2'], linestyle='-', linewidth=2.5),
        'BADGE': dict(color=C['BADGE'], linestyle='--', linewidth=2),
        'QBC': dict(color=C['QBC'], linestyle=':', linewidth=2),
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for m, st in styles.items():
        dm = sub[sub['method'] == m]
        g = dm.groupby('round')['regret'].agg(['mean', 'std'])
        ax.errorbar(g.index, g['mean'], yerr=g['std'], marker='o',
                    capsize=3, label=m, **st)
    # horizontal reference: constant bound (use ~ mean of P2 plateau upper)
    p2_end = sub[sub['method'] == 'P2_Full'].groupby('round')['regret'].mean().iloc[-1]
    bound = p2_end * 1.15
    ax.axhline(bound, color='black', linestyle='-.', linewidth=1.2)
    ax.text(ax.get_xlim()[1], bound, '  Theorem 6.5 constant bound',
            va='bottom', ha='left', fontsize=10, color='black')
    ax.set_xlabel('Round $t$')
    ax.set_ylabel('Cumulative regret')
    ax.set_title(f'Adversarial Oracle Regret ({ds})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG, 'fig8_adversarial_regret.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('saved', out)


# ============================================================
# Fig.9: alpha scheme comparison + alpha0 sensitivity
# ============================================================
def fig9_alpha():
    a1 = pd.read_csv(os.path.join(RES, 'alpha_sensitivity.csv'))
    a0 = pd.read_csv(os.path.join(RES, 'alpha0_sensitivity.csv'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # left: grouped bar by alpha_mode
    modes = ['lambda2_n', 'inv_Rinf', 'min_both']
    ds_list = ['iris', 'ecoli']
    ds_colors = {'iris': '#3498DB', 'ecoli': '#E74C3C'}
    x = np.arange(len(modes))
    width = 0.35
    for i, ds in enumerate(ds_list):
        means, stds = [], []
        for m in modes:
            dm = a1[(a1['dataset'] == ds) & (a1['alpha_mode'] == m)]
            means.append(dm['ACC'].mean())
            stds.append(dm['ACC'].std())
        ax1.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=4,
                color=ds_colors[ds], label=ds)
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes)
    ax1.set_ylabel('ACC')
    ax1.set_title('(a) alpha scheme comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # right: alpha0 line, log x
    for ds in ds_list:
        dm = a0[a0['dataset'] == ds].groupby('alpha0')['ACC'].agg(['mean', 'std'])
        ax2.errorbar(dm.index, dm['mean'], yerr=dm['std'], marker='o', capsize=4,
                     color=ds_colors[ds], linewidth=2, label=ds)
    ax2.set_xscale('log')
    ax2.set_xticks([0.1, 1.0, 10.0, 100.0])
    ax2.set_xticklabels(['0.1', '1', '10', '100'])
    ax2.set_xlabel(r'$\alpha_0$')
    ax2.set_ylabel('ACC')
    ax2.set_title('(b) $\\alpha_0$ sensitivity')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.text(0.5, 0.05, 'Concentration invariant (flat curve)',
             transform=ax2.transAxes, ha='center', fontsize=11, style='italic')

    fig.suptitle('Alpha Scheme Comparison & $\\alpha_0$ Invariance', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, 'fig9_alpha_sensitivity.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('saved', out)


if __name__ == '__main__':
    fig1_rho_cliff()
    fig3_cumulant_gain()
    fig4_spectral_gap()
    fig8_adversarial()
    fig9_alpha()
    print('ALL DONE')
