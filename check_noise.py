import pandas as pd
df = pd.read_csv('results/all_results_merged.csv', low_memory=False)
df = df[df.error.isna() | (df.error=='')]
# P2_Full and BADGE noise profile on ecoli at budget 0.1
for ds in ['ecoli','segment','glass']:
    sub = df[(df.dataset==ds)&(df.budget_frac==0.1)]
    print(f'=== {ds} @10% budget: ACC by method x eta ===')
    piv = sub.groupby(['method','eta']).ACC.mean().unstack()
    cols = [c for c in [0.0,0.1,0.2,0.3] if c in piv.columns]
    print(piv.loc[[m for m in ['P2_Full','BADGE','QBC','Random','CoreSet'] if m in piv.index], cols].round(3).to_string())
    print()
