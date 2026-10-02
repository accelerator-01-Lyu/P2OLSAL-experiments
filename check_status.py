import pandas as pd
df = pd.read_csv('results/master_sweep_20261001_005028.csv', low_memory=False)
err = df[df['error'].notna() & (df['error']!='')]
print(f'总错误: {len(err)//5}jobs')
print(err.groupby(['dataset','method']).size().to_string())
print()
seg = df[df.dataset=='segment']
valid = seg[seg['error'].isna() | (seg['error']=='')]
print(f'segment 有效: {len(valid)//5}jobs')
print(valid.groupby('method').size().to_string())
