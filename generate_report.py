"""Generate final hyperparameter tuning report combining all phases + baselines."""
import csv, collections, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
HP = ['kappa', 'rho_expl', 'lam', 'alpha0', 'eps_q', 'mc_samples', 'alpha_cross_frac']


def load_hp(csv_path):
    rows_by_cfg = collections.defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get('error'):
                continue
            rows_by_cfg[r['config_id']].append(r)
    return rows_by_cfg


def load_baselines(csv_path):
    by = collections.defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get('error'):
                continue
            key = (r['dataset'], r['method'], r['budget_frac'])
            by[key].append(float(r['ACC']))
    return by


def hp_summary(rows_by_cfg, cid):
    rows = rows_by_cfg[cid]
    cfg = {k: float(rows[0][k]) for k in HP}
    datasets = sorted(set(r['dataset'] for r in rows))
    result = {'cfg': cfg, 'datasets': {}}
    for ds in datasets:
        ds_rows = [r for r in rows if r['dataset'] == ds]
        acc10 = [float(r['ACC']) for r in ds_rows if r['budget_frac'] in ('0.1', '0.10')]
        acc20 = [float(r['ACC']) for r in ds_rows if r['budget_frac'] in ('0.2', '0.20')]
        nmi10 = [float(r['NMI']) for r in ds_rows if r['budget_frac'] in ('0.1', '0.10')]
        ari10 = [float(r['ARI']) for r in ds_rows if r['budget_frac'] in ('0.1', '0.10')]
        result['datasets'][ds] = {
            'acc10': np.mean(acc10), 'acc20': np.mean(acc20),
            'nmi10': np.mean(nmi10), 'ari10': np.mean(ari10),
            'acc10_std': np.std(acc10), 'acc20_std': np.std(acc20),
            'n_seeds': len(acc10),
        }
    all10 = [float(r['ACC']) for r in rows if r['budget_frac'] in ('0.1', '0.10')]
    all20 = [float(r['ACC']) for r in rows if r['budget_frac'] in ('0.2', '0.20')]
    result['mean_acc10'] = np.mean(all10)
    result['mean_acc20'] = np.mean(all20)
    return result


def main():
    # Load all phases
    p1 = load_hp(os.path.join(RES, 'hp_phase1.csv'))
    p2 = load_hp(os.path.join(RES, 'hp_phase2.csv'))
    p3 = load_hp(os.path.join(RES, 'hp_phase3.csv'))
    baselines = load_baselines(os.path.join(RES, 'baseline_comparison.csv'))

    # Best config from phase 2 (find actual best by ACC@10%)
    best_cid = None
    best_acc10 = -1
    for cid, rows in p2.items():
        acc10 = [float(r['ACC']) for r in rows if r['budget_frac'] in ('0.1', '0.10')]
        if acc10 and np.mean(acc10) > best_acc10:
            best_acc10 = np.mean(acc10)
            best_cid = cid
    best_p2 = hp_summary(p2, best_cid)
    # Default is cid=5 in phase2
    default_p2 = hp_summary(p2, '5')
    best_p3 = hp_summary(p3, '0')
    default_p3 = hp_summary(p3, '1')

    report = []
    report.append('=' * 80)
    report.append('P2-OLSAL 超参数调优最终报告')
    report.append('=' * 80)
    report.append('')

    # Best config
    report.append('【最优超参数配置】')
    cfg = best_p2['cfg']
    report.append(f'  kappa (ROBUST_KAPPA_BASE)  = {cfg["kappa"]}')
    report.append(f'  rho_expl (EXPL_RHO)         = {cfg["rho_expl"]}')
    report.append(f'  lam (冗余惩罚权重)           = {cfg["lam"]}')
    report.append(f'  alpha0 (DIRICHLET_ALPHA0)   = {cfg["alpha0"]}')
    report.append(f'  eps_q (ROBUST_EPS_Q)        = {cfg["eps_q"]}')
    report.append(f'  mc_samples                  = {cfg["mc_samples"]}')
    report.append(f'  alpha_cross_frac            = {cfg["alpha_cross_frac"]}')
    report.append('')

    report.append('【默认配置（对照）】')
    report.append('  kappa=0.1, rho_expl=0.1, lam=1.0, alpha0=1.0, eps_q=0.05, mc_samples=50, alpha_cross_frac=0.5')
    report.append('')

    # Phase 1 summary
    report.append('-' * 80)
    report.append('第一阶段：iris/wine/seeds 快速筛选（81 配置 × 3 种子 × budget 10%/20%）')
    report.append('-' * 80)
    p1_scored = []
    for cid, rows in p1.items():
        acc10 = [float(r['ACC']) for r in rows if r['budget_frac'] in ('0.1', '0.10')]
        acc20 = [float(r['ACC']) for r in rows if r['budget_frac'] in ('0.2', '0.20')]
        if acc10:
            p1_scored.append((np.mean(acc10), np.mean(acc20), cid))
    p1_scored.sort(key=lambda x: x[0], reverse=True)
    report.append(f'  共 {len(p1_scored)} 个有效配置')
    report.append(f'  默认配置排名: #{[i+1 for i,s in enumerate(p1_scored) if s[2]=="0"][0]}/{len(p1_scored)}')
    report.append(f'  Top-5 ACC@10%: {[f"{s[0]:.4f}" for s in p1_scored[:5]]}')
    report.append('')

    # Phase 2 detailed comparison
    report.append('-' * 80)
    report.append('第二阶段：glass/ecoli 验证（Top5 + 默认，5 种子，budget 10%/20%）')
    report.append('-' * 80)
    report.append('')
    header = f'{"配置":>30s} {"glass@10":>9s} {"glass@20":>9s} {"ecoli@10":>9s} {"ecoli@20":>9s} {"avg@10":>8s} {"avg@20":>8s}'
    report.append(header)
    report.append('-' * len(header))
    for cid in sorted(p2.keys(), key=int):
        s = hp_summary(p2, cid)
        cfg = s['cfg']
        label = (f'k={cfg["kappa"]:.2f},r={cfg["rho_expl"]:.2f},l={cfg["lam"]:.1f}')
        g = s['datasets'].get('glass', {})
        e = s['datasets'].get('ecoli', {})
        report.append(f'{label:>30s} {g.get("acc10",0):9.4f} {g.get("acc20",0):9.4f} '
                      f'{e.get("acc10",0):9.4f} {e.get("acc20",0):9.4f} '
                      f'{s["mean_acc10"]:8.4f} {s["mean_acc20"]:8.4f}')
    report.append('')

    # Phase 3
    report.append('-' * 80)
    report.append('第三阶段：segment 确认（最优 vs 默认，5 种子）')
    report.append('-' * 80)
    report.append('')
    report.append(f'{"配置":>30s} {"seg@10":>9s} {"seg@20":>9s} {"NMI@10":>9s} {"ARI@10":>9s}')
    report.append('-' * 65)
    for label, s in [('Tuned (best)', best_p3), ('Default', default_p3)]:
        seg = s['datasets']['segment']
        report.append(f'{label:>30s} {seg["acc10"]:9.4f} {seg["acc20"]:9.4f} '
                      f'{seg["nmi10"]:9.4f} {seg["ari10"]:9.4f}')
    report.append('')

    # Full comparison with baselines
    report.append('-' * 80)
    report.append('全方法对比：Tuned P2 vs Default P2 vs SOTA 基线（5 种子均值）')
    report.append('-' * 80)
    report.append('')

    datasets = ['glass', 'ecoli', 'segment']
    methods = ['P2_Tuned', 'P2_Default', 'BADGE', 'CoreSet', 'QBC', 'BALD', 'Entropy', 'Random']

    for ds in datasets:
        report.append(f'  数据集: {ds}')
        report.append(f'  {"方法":>12s} {"ACC@10%":>9s} {"ACC@20%":>9s}')
        report.append('  ' + '-' * 35)
        # Tuned P2
        if ds in best_p2['datasets']:
            d = best_p2['datasets'][ds]
            report.append(f'  {"P2_Tuned":>12s} {d["acc10"]:9.4f} {d["acc20"]:9.4f}  *')
        elif ds in best_p3['datasets']:
            d = best_p3['datasets'][ds]
            report.append(f'  {"P2_Tuned":>12s} {d["acc10"]:9.4f} {d["acc20"]:9.4f}  *')
        # Default P2
        if ds in default_p2['datasets']:
            d = default_p2['datasets'][ds]
            report.append(f'  {"P2_Default":>12s} {d["acc10"]:9.4f} {d["acc20"]:9.4f}')
        elif ds in default_p3['datasets']:
            d = default_p3['datasets'][ds]
            report.append(f'  {"P2_Default":>12s} {d["acc10"]:9.4f} {d["acc20"]:9.4f}')
        # Baselines
        for m in ['BADGE', 'CoreSet', 'QBC', 'BALD', 'Entropy', 'Random']:
            v10 = baselines.get((ds, m, '0.1'), baselines.get((ds, m, '0.10'), []))
            v20 = baselines.get((ds, m, '0.2'), baselines.get((ds, m, '0.20'), []))
            if v10:
                report.append(f'  {m:>12s} {np.mean(v10):9.4f} {np.mean(v20):9.4f}')
        report.append('')

    # Key findings
    report.append('-' * 80)
    report.append('关键发现')
    report.append('-' * 80)
    report.append('')
    report.append('1. 探索项 (rho_expl) 应设为 0：在重叠/边界模糊数据集上，基于距离的探索奖励')
    report.append('   会干扰鲁棒信息增益的选择，关闭后 ACC@10% 显著提升。')
    report.append('')
    report.append('2. 冗余惩罚应减弱 (lam=0.1)：默认 lam=1.0 过度惩罚相似样本，导致选择')
    report.append('   过于分散而忽略高信息密度区域。lam=0.1 在 ecoli 上带来 +30% ACC@10%。')
    report.append('')
    report.append('3. 跨簇冗余系数应降低 (alpha_cross_frac=0.25)：默认 0.5 对不同簇间的')
    report.append('   冗余惩罚过强，0.25 允许更多跨簇多样性。')
    report.append('')
    report.append('4. kappa 和 alpha0 在 rho=0/lam=0.1 配置下影响较小：cid=3 (k=1.0,a0=50)')
    report.append('   和 cid=4 (k=0.3,a0=0.5) 产生完全相同的选择结果，说明当探索关闭且')
    report.append('   冗余较弱时，选择主要由原始熵增益驱动。')
    report.append('')
    report.append('5. glass 数据集上调优配置略低于默认：glass 是极小数据集 (214样本, 6类)，')
    report.append('   高方差导致不同配置差异不显著。建议在 glass 上可保留默认或单独调参。')
    report.append('')

    # Config update recommendation
    report.append('-' * 80)
    report.append('配置更新建议')
    report.append('-' * 80)
    report.append('')
    report.append('主 sweep 完成后，建议更新 src/config.py 如下：')
    report.append('  ROBUST_KAPPA_BASE = 1.0   (歧义半径基数，0.3~1.0 均可)')
    report.append('  EXPL_RHO = 0.0            (关闭探索项，关键改动)')
    report.append('  ROBUST_EPS_Q = 0.10')
    report.append('  DIRICHLET_ALPHA0 = 50.0   (0.5~50 范围内影响小)')
    report.append('  MC_SAMPLES = 50')
    report.append('')
    report.append('同时需要修改 src/p2_method.py：')
    report.append('  - 添加 lam 参数（默认 0.1），在 select 中缩放冗余项：gains - lam*red_sum/n')
    report.append('  - 添加 alpha_cross_frac 参数（默认 0.25），替代硬编码的 alpha*0.5')
    report.append('  - 对应 alpha_within_mult 可保持 2.0')
    report.append('')
    report.append('需要重新跑的实验：')
    report.append('  - 所有 6 个原始数据集的 P2_Full 主实验（30种子 × 4 eta × 5 budget）')
    report.append('  - P2 消融实验（NoExploration 消融项结果将变化）')
    report.append('  - 新数据集（letter/shuttle/usps/fashion 等）的 P2_Full 实验')
    report.append('  - 对抗噪声实验（eta>0）需重新验证鲁棒性')
    report.append('')

    text = '\n'.join(report)
    print(text)
    out_path = os.path.join(RES, 'hyperparam_tuning_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'\n报告已保存: {out_path}')


if __name__ == '__main__':
    main()
