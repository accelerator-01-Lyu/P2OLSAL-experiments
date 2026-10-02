"""Generate 4 supplementary experiment figures."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ============================================================
# Fig 1: Cumulant + spectral diagnosis (12 datasets)
# ============================================================
df = pd.read_csv(os.path.join(RES, 'cumulant_spectral_all13.csv'))
df = df[df['error'].isna() | (df['error']=='')].sort_values('T3_norm')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: T3 norm bar
colors = ['#2196F3' if v < 0.05 else '#FF9800' if v < 0.07 else '#F44336' for v in df['T3_norm']]
bars = ax1.bar(range(len(df)), df['T3_norm'], color=colors, edgecolor='white', linewidth=0.5)
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(df['dataset'], rotation=45, ha='right')
ax1.set_ylabel(r'$\|T_3\|$ (third cumulant norm)')
ax1.set_title('(a) FMIS truncation error proxy')
ax1.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='small-remainder threshold')
ax1.legend()
for bar, val in zip(bars, df['T3_norm']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# Right: block spectral gap vs mean cosine
sc = ax2.scatter(df['block_gap_mean'], df['block_cos_mean'],
                  c=df['T3_norm'], cmap='YlOrRd', s=80, edgecolors='black', linewidth=0.5)
for _, row in df.iterrows():
    ax2.annotate(row['dataset'], (row['block_gap_mean'], row['block_cos_mean']),
                 textcoords="offset points", xytext=(5, 3), fontsize=7)
ax2.set_xlabel(r'Within-block spectral gap $\lambda_2(L^{R^{(\beta)}})$')
ax2.set_ylabel(r'Within-block mean cosine similarity $\bar{r}$')
ax2.set_title('(b) Block near-orthogonality (Prop. 7.2)')
ax2.set_xscale('log')
plt.colorbar(sc, ax=ax2, label=r'$\|T_3\|$')

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_cumulant_spectral.png'))
plt.close()
print('Fig1 saved: fig_cumulant_spectral.png')

# ============================================================
# Fig 2: alpha sensitivity
# ============================================================
a = pd.read_csv(os.path.join(RES, 'alpha_sensitivity.csv'))
a = a[a['error'].isna() | (a['error']=='')]

fig, ax = plt.subplots(figsize=(8, 5))
datasets = ['iris', 'ecoli']
modes = ['lambda2_n', 'inv_Rinf', 'min_both']
mode_labels = [r'$\lambda_2/n$ (default)', r'$1/\|R\|_\infty$', r'$\min(\cdot)$']
x = np.arange(len(datasets))
width = 0.25

for i, (mode, label) in enumerate(zip(modes, mode_labels)):
    means = [a[(a.dataset==ds)&(a.alpha_mode==mode)]['ACC'].mean() for ds in datasets]
    stds = [a[(a.dataset==ds)&(a.alpha_mode==mode)]['ACC'].std() for ds in datasets]
    bars = ax.bar(x + i*width - width, means, width, yerr=stds, label=label,
                  capsize=3, edgecolor='white', linewidth=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{m:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(['Iris', 'Ecoli'])
ax.set_ylabel('ACC @ 10% budget')
ax.set_title(r'Sensitivity to redundancy coefficient $\alpha$ (5 seeds)')
ax.legend(loc='lower right')
ax.set_ylim(0.6, 0.85)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_alpha_sensitivity.png'))
plt.close()
print('Fig2 saved: fig_alpha_sensitivity.png')

# ============================================================
# Fig 3: alpha0 sensitivity
# ============================================================
a0 = pd.read_csv(os.path.join(RES, 'alpha0_sensitivity.csv'))
a0 = a0[a0['error'].isna() | (a0['error']=='')]

fig, ax = plt.subplots(figsize=(8, 5))
for ds, marker, color in [('iris', 'o', '#2196F3'), ('ecoli', 's', '#F44336')]:
    sub = a0[a0.dataset==ds].groupby('alpha0')['ACC'].agg(['mean','std'])
    ax.errorbar(sub.index, sub['mean'], yerr=sub['std'], marker=marker,
                color=color, label=ds.capitalize(), capsize=3, linewidth=2, markersize=8)

ax.set_xscale('log')
ax.set_xlabel(r'Dirichlet concentration $\alpha_0$')
ax.set_ylabel('ACC @ 10% budget')
ax.set_title(r'Sensitivity to Dirichlet concentration $\alpha_0$ (5 seeds)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0.65, 0.85)

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_alpha0_sensitivity.png'))
plt.close()
print('Fig3 saved: fig_alpha0_sensitivity.png')

# ============================================================
# Fig 4: rho_expl ablation (KEY FIGURE)
# ============================================================
r = pd.read_csv(os.path.join(RES, 'rho_ablation.csv'))
r = r[r['error'].isna() | (r['error']=='')]

fig, ax = plt.subplots(figsize=(9, 5.5))
for ds, marker, color in [('iris', 'o', '#2196F3'), ('ecoli', 's', '#F44336')]:
    sub = r[r.dataset==ds].groupby('rho_expl')['ACC'].agg(['mean','std'])
    ax.errorbar(sub.index, sub['mean'], yerr=sub['std'], marker=marker,
                color=color, label=ds.capitalize(), capsize=4, linewidth=2.5, markersize=9)
    # Highlight rho=0
    val0 = sub.loc[0.0, 'mean']
    ax.scatter([0], [val0], s=200, facecolors='none', edgecolors=color, linewidths=2.5, zorder=5)
    ax.annotate(f'optimal\n{val0:.3f}', xy=(0, val0), xytext=(0.15, val0+0.02),
                fontsize=9, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

ax.set_xlabel(r'Exploration reward weight $\rho_{\mathrm{expl}}$')
ax.set_ylabel('ACC @ 10% budget')
ax.set_title(r'Ablation: exploration reward is harmful ($\rho=0$ is optimal)')
ax.legend(loc='center right')
ax.grid(alpha=0.3)
ax.set_ylim(0.6, 0.85)
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_rho_ablation.png'))
plt.close()
print('Fig4 saved: fig_rho_ablation.png')

print('\nAll 4 figures saved to', FIG)
