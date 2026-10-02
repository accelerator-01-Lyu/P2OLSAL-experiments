import pandas as pd
df = pd.read_csv('results/all_results_merged.csv', low_memory=False)
df = df[df.error.isna() | (df.error=='')]
sub = df[(df.budget_frac==0.1)&(df.eta==0)]
# 正确等权：每数据集先对种子平均，再13集平均
eq = sub.groupby(['method','dataset']).ACC.mean().groupby('method').mean().sort_values(ascending=False)
print('=== 数据集等权平均 ACC@10% η=0（正确口径）===')
for i,(m,v) in enumerate(eq.items(),1):
    print(f'  {i:2d}. {m:18s} {v*100:.3f}%')
print()
# 噪声等权
ns = df[df.budget_frac==0.1].groupby(['method','dataset','eta']).ACC.mean().groupby(['method','eta']).mean().unstack()
print('=== 等权噪声曲线 ===')
for m in ['P2_Full','BADGE','CoreSet','QBC','Random','Entropy']:
    r=ns.loc[m]; deg=(r[0.0]-r[0.3])/r[0.0]*100
    print(f'  {m:10s} η0={r[0.0]*100:.2f} η.3={r[0.3]*100:.2f} 退化{deg:.1f}%')
