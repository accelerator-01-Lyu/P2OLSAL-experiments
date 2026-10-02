import pandas as pd
df = pd.read_csv('results/all_results_merged.csv', low_memory=False)
df = df[df.error.isna() | (df.error=='')]
sub = df[df.budget_frac==0.1]
# 6-dataset average noise profile
piv = sub.groupby(['method','eta']).ACC.mean().unstack()
print('=== 6-dataset average ACC @10% by eta ===')
print(piv.loc[['P2_Full','BADGE','QBC','Random','CoreSet'], [0.0,0.1,0.2,0.3]].round(3).to_string())
