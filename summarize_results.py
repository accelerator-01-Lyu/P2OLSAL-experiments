"""Summarize all completed experiment results."""
import pandas as pd, glob, numpy as np

HERE = r'D:\P2OLSAL_FSS\experiments'

# Load all valid CSVs
all_dfs = []
for f in sorted(glob.glob(f'{HERE}/results/master_sweep_*.csv')):
    try:
        df = pd.read_csv(f)
        df = df[df['error'].isna() | (df['error']=='')]
        if len(df)>0:
            all_dfs.append(df)
    except: pass

hp = pd.read_csv(f'{HERE}/results/hyperparam_sweep.csv')
hp = hp[hp['error'].isna() | (hp['error']=='')]

adv = pd.read_csv(f'{HERE}/results/adversarial_experiment.csv')
cum = pd.read_csv(f'{HERE}/results/cumulant_spectral_diagnosis.csv')

print('='*60)
print('已完成实验汇总')
print('='*60)

if all_dfs:
    main = pd.concat(all_dfs, ignore_index=True)
    print(f'\n【主 sweep】{len(main)} 行')
    print(f'  数据集: {sorted(main.dataset.unique())}')
    print(f'  方法: {sorted(main.method.unique())}')
    print(f'  预算: {sorted(main.budget_frac.unique())}')
    print(f'  噪声: {sorted(main.eta.unique())}')
    print(f'\n  各数据集完成度:')
    for ds in sorted(main.dataset.unique()):
        sub = main[main.dataset==ds]
        methods = sub.method.nunique()
        seeds = sub.seed.nunique()
        print(f'    {ds}: {len(sub)}行, {methods}方法, {seeds}种子')

print(f'\n【调参实验】{len(hp)} 行')
print(f'  数据集: {sorted(hp.dataset.unique())}')
n_configs = hp[['kappa','rho_expl','lam']].drop_duplicates().shape[0]
print(f'  配置数: {n_configs}')

print(f'\n【对抗实验】{len(adv)} 行')
print(f'  数据集: {sorted(adv.dataset.unique())}')
print(f'  方法: {sorted(adv.method.unique())}')

print(f'\n【累积量诊断】{len(cum)} 行')
print(f'  数据集: {sorted(cum.dataset.unique())}')

# Key results from hyperparam tuning
print('\n' + '='*60)
print('调参关键结果（ACC@10%, 5种子均值）')
print('='*60)
for ds in ['glass','ecoli','segment']:
    sub = hp[(hp.dataset==ds)&(hp.budget_frac==0.10)&(hp.eta==0.0)]
    if len(sub)>0:
        best = sub.loc[sub['ACC'].idxmax()]
        print(f'\n  {ds}:')
        print(f'    最优配置: kappa={best.kappa}, rho={best.rho_expl}, lam={best.lam}')
        print(f'    最优ACC: {best.ACC:.4f}')
        # Compare with default
        default = sub[(sub.kappa==0.1)&(sub.rho_expl==0.1)&(sub.lam==1.0)]
        if len(default)>0:
            print(f'    默认ACC: {default.ACC.mean():.4f}')
            print(f'    提升: {(best.ACC-default.ACC.mean())*100:+.1f} pp')

# Adversarial results
print('\n' + '='*60)
print('对抗Oracle结果（最终轮累积遗憾，越低越好）')
print('='*60)
final = adv[adv['round']==adv['round'].max()]
for ds in sorted(adv.dataset.unique()):
    sub = final[final.dataset==ds]
    print(f'\n  {ds}:')
    for m in sorted(sub.method.unique()):
        r = sub[sub.method==m]['regret'].mean()
        print(f'    {m}: {r:.2f}')
