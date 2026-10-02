import pandas as pd, numpy as np

df = pd.read_csv('results/all_results_merged.csv', low_memory=False)
print(f"Total rows: {len(df)} ({len(df)//5} jobs)")
print(f"Datasets: {df.dataset.nunique()} - {sorted(df.dataset.unique())}")
print(f"Methods: {df.method.nunique()} - {sorted(df.method.unique())}")

# Check errors
if 'error' in df.columns:
    errs = df[df['error'].notna() & (df['error'] != '')]
    print(f"Errors: {len(errs)}")
else:
    print("No error column")

# Coverage matrix
print("\n=== Coverage (jobs per dataset x method) ===")
coverage = df.groupby(['dataset', 'method']).size().div(5).unstack(fill_value=0)
print(coverage.to_string())

# Check for missing cells
ALL_DS = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment',
          'balance', 'aggregation', 'compound', 'letter', 'shuttle', 'usps', 'fashion']
ALL_M = ['P2_Full', 'P2_NoRobust', 'P2_NoExploration', 'P2_NoRedundancy',
         'P2_NoFMIS', 'P2_NoBlock', 'BADGE', 'CoreSet', 'BALD', 'QBC',
         'Entropy', 'Random', 'SSFCM', 'CEFCM', 'GRFCM', 'PLCFCMPassive']
missing = []
for ds in ALL_DS:
    for m in ALL_M:
        if ds not in coverage.index or m not in coverage.columns or coverage.loc[ds, m] == 0:
            missing.append((ds, m))
print(f"\nMissing dataset-method combos: {len(missing)}")
for ds, m in missing:
    print(f"  {ds} - {m}")

# Consistency check
cc = pd.read_csv('results/speedup_consistency_check.csv')
print(f"\n=== Consistency check ===")
for t in cc['test'].unique():
    sub = cc[cc['test'] == t]
    print(f"  {t}: {sub['pass'].sum()}/{len(sub)} pass")
fa = cc[cc['test'] == 'full_al']
print(f"  Full AL max ACC diff: {fa['ACC_diff'].max():.6f}")
print(f"  Full AL max NMI diff: {fa['NMI_diff'].max():.6f}")
print(f"  Full AL max ARI diff: {fa['ARI_diff'].max():.6f}")

# Key results
print("\n=== Key results (ACC@10%, eta=0) ===")
t1 = pd.read_csv('results/table1_main_results.csv')
print(t1[['Method', 'Average']].sort_values('Average', ascending=False).to_string(index=False))
