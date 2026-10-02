"""Extract best-performing methods/configs per dataset."""
import pandas as pd, glob, numpy as np

HERE = r'D:\P2OLSAL_FSS\experiments'

# Load main sweep (deduplicate method names)
all_dfs = []
for f in sorted(glob.glob(f'{HERE}/results/master_sweep_*.csv')):
    try:
        df = pd.read_csv(f)
        df = df[df['error'].isna() | (df['error']=='')]
        if len(df)>0:
            # Normalize method names
            df['method'] = df['method'].replace({
                'P2OLSAL':'P2_Full','BADGESampler':'BADGE','CoreSetSampler':'CoreSet',
                'BALDSampler':'BALD','QBCSampler':'QBC','EntropySampler':'Entropy',
                'RandomSampler':'Random'})
            all_dfs.append(df)
    except: pass
main = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

hp = pd.read_csv(f'{HERE}/results/hyperparam_sweep.csv')
hp = hp[hp['error'].isna() | (hp['error']=='')]

adv = pd.read_csv(f'{HERE}/results/adversarial_experiment.csv')

print('='*70)
print('各数据集最优结果汇总（ACC@10%, eta=0）')
print('='*70)

# === Main sweep: iris (complete) ===
for ds in ['iris','wine']:
    sub = main[(main.dataset==ds)&(main.budget_frac==0.10)&(main.eta==0.0)]
    if len(sub)==0: continue
    print(f'\n【{ds}】主sweep（{sub.seed.nunique()}种子）')
    rank = sub.groupby('method')['ACC'].agg(['mean','std','count']).sort_values('mean',ascending=False)
    for i,(m,row) in enumerate(rank.iterrows(),1):
        marker = ' 🏆' if i==1 else ''
        print(f'  {i}. {m:20s} ACC={row["mean"]:.4f} ± {row["std"]:.4f} (n={int(row["count"])}){marker}')

# === Hyperparam: glass/ecoli/segment ===
for ds in ['glass','ecoli','segment']:
    sub = hp[(hp.dataset==ds)&(hp.budget_frac==0.10)&(hp.eta==0.0)]
    if len(sub)==0: continue
    print(f'\n【{ds}】调参实验（{sub.seed.nunique()}种子，{sub[["kappa","rho_expl","lam"]].drop_duplicates().shape[0]}配置）')
    
    # Top 5 configs
    cfg = sub.groupby(['kappa','rho_expl','lam'])['ACC'].agg(['mean','count']).sort_values('mean',ascending=False).head(5)
    print('  Top-5 配置:')
    for i,((k,r,l),row) in enumerate(cfg.iterrows(),1):
        marker = ' 🏆' if i==1 else ''
        print(f'    {i}. k={k:.2f} rho={r:.2f} lam={l:.2f}  ACC={row["mean"]:.4f} (n={int(row["count"])}){marker}')
    
    # Best P2 vs SOTA baselines (from hyperparam sweep which includes baselines)
    best_p2 = cfg.iloc[0]['mean']
    print(f'\n  P2最优({best_p2:.4f}) vs SOTA基线:')
    method_col = 'method' if 'method' in sub.columns else None
    if method_col:
        for m in ['BADGE','CoreSet','QBC','BALD','Entropy','Random']:
            bsub = sub[sub[method_col]==m]
            if len(bsub)>0:
                diff = (best_p2 - bsub['ACC'].mean())*100
                win = '✅' if diff>0 else '❌'
                print(f'    {m:10s} ACC={bsub["ACC"].mean():.4f}  P2领先={diff:+.1f}pp {win}')
    else:
        print('    (调参CSV无基线方法列，SOTA对比见hyperparam_tuning_report.txt)')

# === Adversarial ===
print('\n' + '='*70)
print('对抗Oracle最优（最终轮累积遗憾，越低越好）')
print('='*70)
final = adv[adv['round']==adv['round'].max()]
for ds in sorted(adv.dataset.unique()):
    sub = final[final.dataset==ds]
    rank = sub.groupby('method')['regret'].mean().sort_values()
    print(f'\n  {ds}:')
    for i,(m,v) in enumerate(rank.items(),1):
        marker = ' 🏆' if i==1 else ''
        print(f'    {i}. {m:10s} regret={v:.2f}{marker}')

# === Cumulant diagnosis ===
print('\n' + '='*70)
print('累积量诊断（T3范数越小=越接近高斯=FMIS越精确）')
print('='*70)
cum = pd.read_csv(f'{HERE}/results/cumulant_spectral_diagnosis.csv')
cum_sorted = cum.sort_values('T3_norm')
for _,row in cum_sorted.iterrows():
    cls = '小余项(近高斯)' if row['T3_norm']<0.1 else '大余项(非高斯)'
    print(f'  {row["dataset"]:15s} T3={row["T3_norm"]:.3f}  lambda2={row["lambda2_LR"]:.2f}  cos={row["mean_cos"]:.3f}  [{cls}]')
