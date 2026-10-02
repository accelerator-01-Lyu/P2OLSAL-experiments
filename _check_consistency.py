import pandas as pd
df = pd.read_csv('results/speedup_consistency_check.csv')
print('tests:', df['test'].unique())
for t in df['test'].unique():
    sub = df[df['test'] == t]
    print(f'  {t}: {sub["pass"].sum()}/{len(sub)} pass')
fa = df[df['test'] == 'full_al']
if len(fa) > 0:
    print(f'  full_al max ACC diff: {fa["ACC_diff"].max():.6f}')
    print(f'  full_al max NMI diff: {fa["NMI_diff"].max():.6f}')
    print(f'  full_al max ARI diff: {fa["ARI_diff"].max():.6f}')
al = df[df['test'] == 'alpha_eigsh']
if len(al) > 0:
    print(f'  alpha max diff: {al["alpha_diff"].max():.2e}')
pp = df[df['test'] == 'plcfcm_precision']
if len(pp) > 0:
    print(f'  plcfcm max U diff: {pp["U_max_diff"].max():.2e}')
print(f'\nOVERALL: {"PASS" if df["pass"].all() else "FAIL"}')
