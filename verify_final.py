import pandas as pd
df = pd.read_csv('results/all_results_merged.csv', low_memory=False)
df = df[df.error.isna() | (df.error=='')]
print('总行数:', len(df), '数据集:', df.dataset.nunique(), '方法:', df.method.nunique())
print()
# ACC@10%, eta=0, all 13 datasets
sub = df[(df.budget_frac==0.1)&(df.eta==0)]
piv = sub.groupby('method').ACC.mean().sort_values(ascending=False)
print('=== 13数据集平均 ACC@10% η=0 ===')
for m,v in piv.items():
    print(f'  {m:20s} {v:.4f}')
print()
# per-dataset best
print('=== 各数据集最佳方法 ===')
for ds in sorted(df.dataset.unique()):
    d = sub[sub.dataset==ds]
    best = d.groupby('method').ACC.mean().sort_values(ascending=False)
    top = best.head(2)
    p2 = best.get('P2_Full', float('nan'))
    print(f'  {ds:12s} best={top.index[0]:18s}({top.iloc[0]:.3f})  2nd={top.index[1]:18s}({top.iloc[1]:.3f})  P2={p2:.3f}')
print()
# noise robustness (13-dataset avg)
print('=== 13数据集平均噪声鲁棒性 ===')
ns = df[df.budget_frac==0.1].groupby(['method','eta']).ACC.mean().unstack()
for m in ['P2_Full','BADGE','QBC','Random','CoreSet','Entropy']:
    if m in ns.index:
        row = ns.loc[m]
        deg = (row[0.0]-row[0.3])/row[0.0]*100
        print(f'  {m:12s} η0={row[0.0]:.3f} η.3={row[0.3]:.3f} 退化{deg:.1f}%')
print()
# ablation
print('=== 消融 (13数据集平均 ACC@10% η=0) ===')
for m in ['P2_Full','P2_NoRobust','P2_NoRedundancy','P2_NoBlock','P2_NoExploration','P2_NoFMIS']:
    if m in piv.index:
        print(f'  {m:18s} {piv[m]:.4f}  diff={piv[m]-piv["P2_Full"]:+.4f}')
