import pandas as pd
df = pd.read_csv('results/all_results_merged.csv', low_memory=False)
df = df[df.error.isna() | (df.error=='')]
sub = df[(df.budget_frac==0.1)&(df.eta==0)]
print('=== 主表口径 @10% eta=0, 13集等权平均 ===')
for m in ['P2_Full','P2_NoRobust','P2_NoRedundancy','P2_NoBlock','P2_NoExploration','P2_NoFMIS']:
    v = sub[sub.method==m].groupby('dataset').ACC.mean().mean()
    print(f'  {m:18s} {v*100:.3f}%  diff_vs_full={(v-sub[sub.method=="P2_Full"].groupby("dataset").ACC.mean().mean())*100:+.2f}pp')
print()
# 其它可能口径
print('=== 可能的错误口径排查 ===')
allb = df[df.eta==0]
print('含全部budget平均 P2_Full:', round(allb[allb.method=="P2_Full"].ACC.mean()*100,3),'%')
print('含全部eta平均 P2_Full:', round(df[df.method=="P2_Full"].ACC.mean()*100,3),'%')
# 每行直接平均(不按数据集等权)
print('行直接平均@10%η0 P2_Full:', round(sub[sub.method=="P2_Full"].ACC.mean()*100,3),'%  (种子数不等会偏)')
print('每数据集种子数:')
print(sub[sub.method=="P2_Full"].groupby('dataset').size())
