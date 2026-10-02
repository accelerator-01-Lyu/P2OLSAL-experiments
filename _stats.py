import pandas as pd, numpy as np
df = pd.read_csv('results/all_results_merged.csv', low_memory=False)
print("=== Noise robustness (ACC@10% vs eta) ===")
for m in ['P2_Full','BADGE','QBC','Entropy','Random','CoreSet']:
    vals = []
    for eta in [0,0.1,0.2,0.3]:
        sub = df[(df.method==m)&(df.budget_frac==0.10)&(df.eta==eta)]
        vals.append(sub['ACC'].mean())
    deg = (vals[0]-vals[3])/vals[0]*100
    print(f'{m}: {vals[0]:.3f} -> {vals[3]:.3f} ({deg:.1f}% deg)')

print("\n=== P2_Full vs best non-P2 @10% eta=0 ===")
for ds in sorted(df.dataset.unique()):
    p2 = df[(df.method=='P2_Full')&(df.dataset==ds)&(df.eta==0)&(df.budget_frac==0.10)]['ACC'].mean()
    others = df[(df.dataset==ds)&(df.eta==0)&(df.budget_frac==0.10)&(~df.method.str.startswith('P2_'))]
    best = others.groupby('method')['ACC'].mean().max()
    best_m = others.groupby('method')['ACC'].mean().idxmax()
    print(f'{ds}: P2={p2:.3f}, best={best_m}={best:.3f}, gap={p2-best:+.3f}')

print("\n=== Average ACC ranking ===")
avgs = []
for m in df.method.unique():
    sub = df[(df.method==m)&(df.eta==0)&(df.budget_frac==0.10)]
    avg = sub.groupby('dataset')['ACC'].mean().mean()
    avgs.append((m, avg))
for m, a in sorted(avgs, key=lambda x: -x[1]):
    print(f'  {m}: {a:.3f}')
