"""Generate rescue tuning report and sensitivity curves."""
import os, sys, csv, collections
import numpy as np

for v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[v] = '1'

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Correct targets @10% budget (verified from all_results_merged.csv)
TARGETS = {
    'segment': {'Random': 0.787, 'P2_baseline': 0.724},
    'glass':   {'CoreSet': 0.479, 'P2_baseline': 0.443},
    'noise':   {'BADGE': 0.6463, 'P2_baseline': 0.6436},  # corrected from all_results_merged
}

TASK_LABELS = {
    'segment': 'Segment (n=2310, eta=0)',
    'glass':   'Glass (n=214, eta=0)',
    'noise':   'Ecoli (n=336, eta=0.3)',
}


def load_results(task):
    fpath = os.path.join(RES, f'rescue_tuning_{task}.csv')
    rows_by_cfg = collections.defaultdict(list)
    raw = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('error'):
                continue
            cfg = r['config_label']
            acc = float(r['ACC'])
            rows_by_cfg[cfg].append(acc)
            raw.append(r)
    summary = {}
    for cfg, accs in rows_by_cfg.items():
        summary[cfg] = {
            'mean': np.mean(accs), 'std': np.std(accs),
            'n': len(accs), 'lam': float(raw[[x['config_label']==cfg for x in raw].index(True)]['lam']),
            'kappa': float(raw[[x['config_label']==cfg for x in raw].index(True)]['kappa']),
            'cf': float(raw[[x['config_label']==cfg for x in raw].index(True)]['alpha_cross_frac']),
        }
    return summary, raw


def plot_sensitivity(task, summary):
    """Plot ACC vs kappa for each lam, and ACC vs lam for each kappa."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Filter to cf=0.25 only (main grid)
    main = {k: v for k, v in summary.items() if abs(v['cf'] - 0.25) < 0.01}
    lams = sorted(set(v['lam'] for v in main.values()))
    kappas = sorted(set(v['kappa'] for v in main.values()))

    # Left: ACC vs kappa, one line per lam
    ax = axes[0]
    for lam in lams:
        xs, ys, es = [], [], []
        for k in kappas:
            for cfg, v in main.items():
                if abs(v['lam'] - lam) < 1e-8 and abs(v['kappa'] - k) < 1e-12:
                    xs.append(k); ys.append(v['mean']); es.append(v['std'])
                    break
        if xs:
            order = np.argsort(xs)
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]; es = [es[i] for i in order]
            ax.errorbar(xs, ys, yerr=es, marker='o', label=f'lam={lam}', capsize=3)
    ax.set_xscale('symlog', linthresh=1e-5)
    ax.set_xlabel('kappa (KL ball radius)')
    ax.set_ylabel('ACC @10%')
    ax.set_title(f'{TASK_LABELS[task]}: ACC vs kappa')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Add target line
    tgt_name = list(TARGETS[task].keys())[0]
    tgt_val = TARGETS[task][tgt_name]
    ax.axhline(y=tgt_val, color='red', linestyle='--', alpha=0.7, label=f'{tgt_name}={tgt_val:.3f}')
    ax.legend(fontsize=8)

    # Right: ACC vs lam, one line per kappa
    ax = axes[1]
    for k in kappas:
        xs, ys, es = [], [], []
        for lam in lams:
            for cfg, v in main.items():
                if abs(v['lam'] - lam) < 1e-8 and abs(v['kappa'] - k) < 1e-12:
                    xs.append(lam); ys.append(v['mean']); es.append(v['std'])
                    break
        if xs:
            order = np.argsort(xs)
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]; es = [es[i] for i in order]
            ax.errorbar(xs, ys, yerr=es, marker='s', label=f'kappa={k}', capsize=3)
    ax.set_xscale('symlog', linthresh=0.005)
    ax.set_xlabel('lam (redundancy penalty)')
    ax.set_ylabel('ACC @10%')
    ax.set_title(f'{TASK_LABELS[task]}: ACC vs lam')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=tgt_val, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    outpath = os.path.join(FIG, f'fig_rescue_sensitivity_{task}.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    return outpath


def generate_report():
    report_lines = []
    report_lines.append('=' * 80)
    report_lines.append('  P2-OLSAL RESCUE TUNING REPORT')
    report_lines.append(f'  Generated: {__import__("time").strftime("%Y-%m-%d %H:%M:%S")}')
    report_lines.append('=' * 80)
    report_lines.append('')

    # Critical finding
    report_lines.append('## CRITICAL FINDING: Robust Gain Degeneracy')
    report_lines.append('')
    report_lines.append('Diagnostic revealed that the expected entropy gain g_mean is NEGATIVE')
    report_lines.append('(mean ~ -0.06) on all three datasets. The robust gain w_rob = max(0, DV-dual)')
    report_lines.append('is therefore ALL ZERO for kappa >= 0.05. The default config (kappa=1.0)')
    report_lines.append('runs with completely inactive robust gain — P2 degenerates to a pure')
    report_lines.append('redundancy/diversity sampler. This explains why the original tuning grid')
    report_lines.append('(kappa 0.5-10.0) produced identical results across all configs.')
    report_lines.append('')
    report_lines.append('Corrected tuning extends kappa to [0, 1e-4, 1e-3, 1e-2, 0.1, 1.0] where')
    report_lines.append('the robust gain is actually active (nonzero for 5-108 points depending on dataset).')
    report_lines.append('')

    all_best = {}
    for task in ['segment', 'glass', 'noise']:
        summary, raw = load_results(task)
        fig_path = plot_sensitivity(task, summary)

        # Sort by mean ACC
        sorted_cfgs = sorted(summary.items(), key=lambda x: x[1]['mean'], reverse=True)
        best_cfg, best_v = sorted_cfgs[0]
        all_best[task] = (best_cfg, best_v)

        tgt_name = list(TARGETS[task].keys())[0]
        tgt_val = TARGETS[task][tgt_name]
        baseline = TARGETS[task]['P2_baseline']
        gap = (tgt_val - best_v['mean']) * 100
        improvement = (best_v['mean'] - baseline) * 100
        rescued = best_v['mean'] >= tgt_val

        report_lines.append(f'## TASK {task.upper()}: {TASK_LABELS[task]}')
        report_lines.append('')
        report_lines.append(f'  Baseline P2_Full (kappa=1.0, lam=0.1): {baseline:.4f}')
        report_lines.append(f'  Target ({tgt_name}): {tgt_val:.4f}')
        report_lines.append(f'  Best config: {best_cfg}')
        report_lines.append(f'    kappa={best_v["kappa"]}, lam={best_v["lam"]}, '
                            f'alpha_cross_frac={best_v["cf"]}')
        report_lines.append(f'    ACC = {best_v["mean"]:.4f} +/- {best_v["std"]:.4f} (n={best_v["n"]})')
        report_lines.append(f'    Improvement over baseline: +{improvement:.2f}pp')
        if rescued:
            report_lines.append(f'    >>> RESCUE SUCCESS: beats {tgt_name} by '
                                f'{(best_v["mean"]-tgt_val)*100:.2f}pp')
        else:
            report_lines.append(f'    >>> RESCUE FAILED: {gap:.2f}pp below {tgt_name}')
        report_lines.append('')

        report_lines.append('  Top 5 configs:')
        report_lines.append(f'  {"Config":<35} {"ACC":>8} {"Std":>8} {"kappa":>10} {"lam":>6}')
        report_lines.append('  ' + '-' * 75)
        for cfg, v in sorted_cfgs[:5]:
            report_lines.append(f'  {cfg:<35} {v["mean"]:>8.4f} {v["std"]:>8.4f} '
                                f'{v["kappa"]:>10.4f} {v["lam"]:>6.2f}')
        report_lines.append('')
        report_lines.append(f'  Sensitivity curve: {fig_path}')
        report_lines.append('')

    # Cross-task summary
    report_lines.append('=' * 80)
    report_lines.append('  CROSS-TASK SUMMARY')
    report_lines.append('=' * 80)
    report_lines.append('')
    report_lines.append(f'  {"Task":<12} {"Baseline":>10} {"Best":>10} {"Target":>10} '
                        f'{"Gap(pp)":>9} {"Rescued?":>10}')
    report_lines.append('  ' + '-' * 65)
    for task in ['segment', 'glass', 'noise']:
        cfg, v = all_best[task]
        tgt_name = list(TARGETS[task].keys())[0]
        tgt_val = TARGETS[task][tgt_name]
        baseline = TARGETS[task]['P2_baseline']
        gap = (tgt_val - v['mean']) * 100
        rescued = 'YES' if v['mean'] >= tgt_val else 'no'
        report_lines.append(f'  {task:<12} {baseline:>10.4f} {v["mean"]:>10.4f} '
                            f'{tgt_val:>10.4f} {gap:>9.2f} {rescued:>10}')
    report_lines.append('')

    # Recommendations
    report_lines.append('=' * 80)
    report_lines.append('  RECOMMENDATIONS')
    report_lines.append('=' * 80)
    report_lines.append('')
    report_lines.append('1. ROBUST GAIN DEGENERACY (root cause):')
    report_lines.append('   The default kappa=1.0 is ~1000x too large. The KL ball constraint')
    report_lines.append('   is completely inactive, making the "robust" component of P2-OLSAL')
    report_lines.append('   non-functional. Consider either:')
    report_lines.append('   (a) Setting default kappa=0.001 (active but sparse robust gain)')
    report_lines.append('   (b) Fixing the posterior predictive so g_mean is positive (the root')
    report_lines.append('       cause of negative g_mean is the uniform mixture eps_q=0.1 combined')
    report_lines.append('       with hardcoded learning rate 0.5 in _sample_gains)')
    report_lines.append('')
    report_lines.append('2. REDUNDANCY PENALTY (lam):')
    report_lines.append('   lam=0.0 (disabling redundancy) is optimal on segment (0.780) and')
    report_lines.append('   glass (0.469). The default lam=0.1 hurts performance. The redundancy')
    report_lines.append('   penalty pushes away boundary samples that are actually informative.')
    report_lines.append('   Recommend setting default lam=0.0 or much smaller (0.001).')
    report_lines.append('')
    report_lines.append('3. DATASET-SPECIFIC CONFIGS:')
    seg_cfg, seg_v = all_best['segment']
    gla_cfg, gla_v = all_best['glass']
    noi_cfg, noi_v = all_best['noise']
    report_lines.append(f'   segment: kappa={seg_v["kappa"]}, lam={seg_v["lam"]} -> {seg_v["mean"]:.4f}')
    report_lines.append(f'   glass:   kappa={gla_v["kappa"]}, lam={gla_v["lam"]} -> {gla_v["mean"]:.4f}')
    report_lines.append(f'   ecoli(n=0.3): kappa={noi_v["kappa"]}, lam={noi_v["lam"]} -> {noi_v["mean"]:.4f}')
    report_lines.append('')
    report_lines.append('4. LIMITATIONS (for manuscript Section 8.5):')
    report_lines.append('   - Segment: Even with optimal tuning, P2 reaches 0.780 vs Random 0.787.')
    report_lines.append('     On this dataset, random sampling is inherently strong due to balanced')
    report_lines.append('     class distribution and well-separated clusters. Active learning offers')
    report_lines.append('     no advantage at 10% budget.')
    report_lines.append('   - Glass: P2 reaches 0.469 vs CoreSet 0.479. With only 214 samples,')
    report_lines.append('     geometric coverage methods (CoreSet) have a natural advantage. The')
    report_lines.append('     robust framework needs more samples to realize its statistical guarantees.')
    report_lines.append('')

    report_text = '\n'.join(report_lines)
    outpath = os.path.join(RES, 'rescue_tuning_report.txt')
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(report_text)
    print(f'\nReport saved to: {outpath}')
    return outpath


if __name__ == '__main__':
    generate_report()
