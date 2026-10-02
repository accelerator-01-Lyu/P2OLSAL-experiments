import pandas as pd
df = pd.read_csv('results/all_results_merged.csv', low_memory=False)
df = df[df['error'].isna() | (df['error']=='')]
agg = df.groupby('method')[['sel_time','retrain_time','total_time']].mean()
agg = agg.sort_values('total_time')
print(agg.to_string())
