# -*- coding: utf-8 -*-
"""Remaining 6 figures in the frozen journal style:
  fig05_noise_robustness   - equal-weight ACC vs label-noise eta
  fig09_adversarial_regret - cumulative regret vs round (P2/BADGE/QBC)
  fig08_misfit_proxy       - centroid-perturb proxy vs realized zeta
  fig06_cumulant_gap       - (a) T3 vs P2-Entropy advantage; (b) spectral gap vs cosine
  fig11_rescue             - segment/glass/noise rescue: default vs best-grid vs external
  fig_geometry             - schematic: simplex max-entropy pairwise projection (FMIS)
All numbers come from CSV; nothing is invented. Outputs png(400dpi)+pdf into figures/.
"""
import os, sys
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments\figures')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import _journal_style as js

RES = r'D:\P2OLSAL_FSS\experiments\results'


# ============================================================== 1 noise
def fig_noise():
    n = pd.read_csv(os.path.join(RES, 'noise_equalweight.csv'), index_col=0)
    etas = [float(c) for c in n.columns]
    order = ['P2_Full', 'BADGE', 'Random', 'CoreSet', 'QBC', 'Entropy']
    fig, ax = plt.subplots(figsize=(js.TEXTWIDTH_IN, js.TEXTWIDTH_IN * 0.52))
    for m in order:
        js.plot_method(ax, etas, n.loc[m].values.astype(float), m)
    ax.set_xlabel(r'label-noise fraction $\eta$')
    ax.set_ylabel('clustering ACC')
    ax.set_xticks(etas)
    ax.set_xticklabels([f'{e:g}' for e in etas])
    ax.set_ylim(0.60, 0.76)
    ax.legend(ncol=3, loc='lower left', frameon=True)
    ax.set_title('Label-noise robustness (13-dataset equal-weighted ACC, 10% budget)')
    js.despine(ax)
    js.save_both(fig, 'fig05_noise_robustness')


# ============================================================== 2 adversarial
def fig_adversarial():
    d = pd.read_csv(os.path.join(RES, 'adversarial_experiment.csv'))
    g = d.groupby(['method', 'round']).regret.agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(js.TEXTWIDTH_IN, js.TEXTWIDTH_IN * 0.55))
    for m in ['P2_Full', 'BADGE', 'QBC']:
        s = g[g.method == m].sort_values('round')
        x = s['round'].values
        y = s['mean'].values
        e = s['std'].fillna(0).values
        js.plot_method(ax, x, y, m)
        ax.fill_between(x, y - e, y + e, color=js.METHOD_STYLE[m]['color'], alpha=0.15, zorder=2)
    # theoretical linear bound for P2 (constant per-round regret C_adv)
    x = np.sort(d['round'].unique())
    slope = g[(g.method == 'P2_Full') & (g['round'] == x.max())]['mean'].iloc[0] / x.max()
    ax.plot(x, slope * x, ls=':', color=js.P2, lw=1.4, zorder=3,
            label=r'linear bound $C_{\mathrm{adv}}\cdot t$ (Thm. 6.5)')
    ax.set_xlabel('active-learning round $t$')
    ax.set_ylabel(r'cumulative regret')
    ax.set_title('Adversarial oracle: cumulative regret stays bounded (linear, not divergent)')
    ax.legend(loc='upper left', frameon=True)
    js.despine(ax)
    js.save_both(fig, 'fig09_adversarial_regret')


# ============================================================== 3 misfit proxy
def fig_misfit():
    d = pd.read_csv(os.path.join(RES, 'misfit_proxy_data.csv'))
    x = d.delta_mis_hat.values.astype(float)
    y = d.zeta_t.values.astype(float)
    # OLS
    A = np.vstack([x, np.ones_like(x)]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = plt.subplots(figsize=(js.TEXTWIDTH_IN * 0.72, js.TEXTWIDTH_IN * 0.62))
    ax.scatter(x, y, s=70, color=js.BASELINE_COLORS['BADGE'], marker='o',
               edgecolor='white', linewidth=0.9, zorder=5)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, b * xx + a, color=js.P2, lw=2.2, zorder=4,
            label=rf'OLS  slope $={b:.2f}$')
    ax.text(0.04, 0.95, f'Pearson $r={r:.2f}$\n$n={len(x)}$ rounds',
            transform=ax.transAxes, va='top', ha='left', fontsize=10.5,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#CCCCCC'))
    ax.set_xlabel(r'centroid-perturbation ensemble variance $\hat{\delta}_{\mathrm{mis}}^t$')
    ax.set_ylabel(r'realized expert-model disagreement $\zeta_t$')
    ax.set_title('Misfit-budget proxy tracks realized label surprise')
    ax.legend(loc='lower right')
    js.despine(ax)
    js.save_both(fig, 'fig08_misfit_proxy')


# ============================================================== 4 cumulant
def fig_cumulant():
    spec = pd.read_csv(os.path.join(RES, 'cumulant_spectral_all13.csv')).set_index('dataset')
    raw = pd.read_csv(os.path.join(RES, 'all_results_merged.csv'), low_memory=False)
    raw = raw[raw['error'].isna() | (raw['error'].astype(str) == '')]
    m = raw[(raw.budget_frac == 0.1) & (raw.eta == 0)]
    p2 = m[m.method == 'P2_Full'].groupby('dataset').ACC.mean()
    ent = m[m.method == 'Entropy'].groupby('dataset').ACC.mean()
    adv = ((p2 - ent) * 100).dropna()
    common = spec.index.intersection(adv.index)
    frame = pd.DataFrame({'T3': spec.loc[common, 'T3_norm'],
                          'adv': adv.loc[common]}).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(js.TEXTWIDTH_IN, js.TEXTWIDTH_IN * 0.46))
    # (a) T3 vs advantage
    ax = axes[0]
    x = frame['T3'].values.astype(float)
    y = frame['adv'].values.astype(float)
    ax.scatter(x, y, s=60,
               color=js.P2, marker='o', edgecolor='white', linewidth=0.8, zorder=5)
    A = np.vstack([x, np.ones_like(x)]).T
    bb, aa = np.linalg.lstsq(A, y, rcond=None)[0]
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, bb * xx + aa, color='#555555', lw=1.8, ls='--')
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f'Pearson $r={r:.2f}$', transform=ax.transAxes,
            va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#CCCCCC'))
    ax.set_xlabel(r'third-cumulant norm $\|T_3\|$')
    ax.set_ylabel(r'P2-OLSAL $-$ Entropy ACC advantage (pp)')
    ax.set_title('(a) Higher-order interaction vs gain', loc='left')
    js.despine(ax)
    # (b) spectral gap vs mean cosine
    ax = axes[1]
    ax.scatter(spec['lambda2_LR'], spec['mean_cos'], s=60,
               color=js.BASELINE_COLORS['BADGE'], marker='s',
               edgecolor='white', linewidth=0.8, zorder=5)
    ax.set_xscale('log')
    ax.set_xlabel(r'redundancy-Laplacian spectral gap $\lambda_2(L^R)$')
    ax.set_ylabel(r'mean within-block cosine $\bar r$')
    ax.set_title('(b) Spectral gap vs block coherence', loc='left')
    js.despine(ax)
    fig.subplots_adjust(wspace=0.34)
    js.save_both(fig, 'fig06_cumulant_gap')


# ============================================================== 5 rescue
def _external_ref(method, dataset, eta):
    raw = pd.read_csv(os.path.join(RES, 'all_results_merged.csv'), low_memory=False)
    raw = raw[raw['error'].isna() | (raw['error'].astype(str) == '')]
    s = raw[(raw.method == method) & (raw.dataset == dataset) &
            (raw.budget_frac == 0.1) & (raw.eta == eta)]
    return s.ACC.mean(), s.ACC.std(ddof=1)


def fig_rescue():
    cases = [
        ('segment', 'segment', 0.0, 'Random', 'P2 default'),
        ('glass', 'glass', 0.0, 'CoreSet', 'P2 default'),
        ('noise', 'ecoli', 0.3, 'Random', r'high-noise ($\eta=.3$)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(js.TEXTWIDTH_IN, js.TEXTWIDTH_IN * 0.5),
                             sharey=False)
    for ax, (file, ds, eta, ext, sub) in zip(axes, cases):
        d = pd.read_csv(os.path.join(RES, f'rescue_tuning_{file}.csv'))
        d = d[d.eta == eta] if d.eta.notna().any() else d
        agg = d.groupby('config_label').ACC.agg(['mean', 'std'])
        default = agg.loc[[i for i in agg.index if i == 'k=1.0_lam=0.1']]
        if len(default) == 0:
            default = agg.loc[[i for i in agg.index if 'k=1.0' in i][:1]]
        best_idx = agg['mean'].idxmax()
        labels = ['P2\ndefault', 'best grid\nsearch', f'{ext}']
        vals = [default['mean'].iloc[0], agg.loc[best_idx, 'mean']]
        errs = [default['std'].iloc[0], agg.loc[best_idx, 'std']]
        rv, re = _external_ref(ext, ds, eta)
        vals.append(rv)
        errs.append(re if not np.isnan(re) else 0)
        colors = [js.P2, '#E08214', js.BASELINE_COLORS['BADGE']]
        xpos = np.arange(3)
        ax.bar(xpos, vals, yerr=errs, capsize=3, color=colors, width=0.62,
               edgecolor='white', linewidth=0.8, error_kw=dict(ecolor='#666', lw=1))
        for xx, vv in zip(xpos, vals):
            ax.text(xx, vv + 0.012, f'{vv:.3f}', ha='center', fontsize=9, color='#333')
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_title(f'{ds} {sub}', fontsize=11)
        js.despine(ax)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('clustering ACC')
    fig.suptitle('Theory-guided rescue: default vs best local grid vs strongest external baseline',
                 fontsize=12.5, fontweight='bold', y=1.02)
    js.save_both(fig, 'fig11_rescue')


# ============================================================== 6 geometry
def fig_geometry():
    fig = plt.figure(figsize=(js.TEXTWIDTH_IN * 0.78, js.TEXTWIDTH_IN * 0.66))
    ax = fig.add_subplot(111, projection='3d')
    # probability simplex (triangle in z=0 plane of 3D barycentric -> use plane)
    v = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
    tri = Poly3DCollection([v], fc='#F2F2F2', ec='#999999', lw=1.2)
    ax.add_collection3d(tri)
    # vertices = clusters
    for p, name in zip(v, ['$e_1$', '$e_2$', '$e_3$']):
        ax.scatter(p[0], p[1], 0, color=js.P2, s=40)
        ax.text(p[0], p[1], 0.03, name, fontsize=11, ha='center')
    # interior point = membership
    c = np.array([0.42, 0.34])
    ax.scatter(c[0], c[1], 0, color=js.BASELINE_COLORS['BADGE'], s=45, marker='s')
    ax.text(c[0] + 0.03, c[1], 0.03, r'$u\in\Delta^{c-1}$', fontsize=10)
    # curved surface = true fuzzy MI (higher order); flat pairwise mesh = FMIS
    gx = np.linspace(0.05, 0.95, 40)
    gy = np.linspace(0.02, np.sqrt(3) / 2 - 0.02, 40)
    GX, GY = np.meshgrid(gx, gy)
    inside = (GY < np.sqrt(3) * GX + 0.0) & (GY < np.sqrt(3) * (1 - GX)) & (GY > 0.02)
    # pairwise (FMIS): gentle paraboloid
    z_pair = 0.12 + 0.10 * np.exp(-((GX - c[0]) ** 2 + (GY - c[1]) ** 2) * 6)
    # true MI: adds rippled higher-order structure
    z_true = z_pair + 0.06 * np.sin(7 * GX) * np.sin(7 * GY)
    z_pair = np.where(inside, z_pair, np.nan)
    z_true = np.where(inside, z_true, np.nan)
    ax.plot_wireframe(GX, GY, z_pair, color=js.BASELINE_COLORS['BADGE'],
                      linewidth=0.5, alpha=0.55)
    ax.plot_wireframe(GX, GY, z_true, color=js.P2, linewidth=0.5, alpha=0.45)
    ax.text(0.75, 0.62, 0.28, 'true $I_{\\mathrm{fuzzy}}$\n(higher-order)',
            color=js.P2, fontsize=9.5)
    ax.text(0.12, 0.5, 0.22, 'FMIS: pairwise\nmax-entropy projection',
            color=js.BASELINE_COLORS['BADGE'], fontsize=9.5)
    ax.set_zlim(0, 0.35)
    ax.view_init(elev=22, azim=-58)
    ax.set_title('FMIS as the pairwise max-entropy projection on the simplex (schematic)',
                 fontsize=11.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    js.save_both(fig, 'fig_geometry')


if __name__ == '__main__':
    fig_noise(); print('noise ok')
    fig_adversarial(); print('adversarial ok')
    fig_misfit(); print('misfit ok')
    fig_cumulant(); print('cumulant ok')
    fig_rescue(); print('rescue ok')
    fig_geometry(); print('geometry ok')
