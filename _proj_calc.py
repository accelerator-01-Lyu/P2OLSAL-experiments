import pandas as pd, numpy as np
df = pd.read_csv(r'D:\P2OLSAL_FSS\experiments\results\all_results_merged.csv', low_memory=False)
ds_all = ['iris','wine','seeds','glass','ecoli','segment','balance','aggregation','compound',
          'letter','shuttle','usps','fashion']
def acc(m, d, eta=0.0, b=0.10):
    s = df[(df.method==m)&(df.dataset==d)&(df.eta==eta)&(df.budget_frac==b)]
    return s.ACC.mean() if len(s) else np.nan
cur=[]; proj=[]
for d in ds_all:
    a_full=acc('P2_Full',d); a_nored=acc('P2_NoRedundancy',d)
    cur.append(a_full)
    relax = d in ('letter','shuttle','usps','fashion')
    proj.append(a_nored if relax else a_full)
    if relax:
        print(f'{d:>11}: Full={a_full:.3f} NoRed={a_nored:.3f} delta=+{100*(a_nored-a_full):.1f}pp')
print(f'\ncurrent  P2 mean ACC@10 eta=0 : {np.mean(cur):.4f}')
print(f'projected C mean (gamma=0 relax on 4, NoRed values): {np.mean(proj):.4f}')
# ranking check
avgs={}
for m in df.method.unique():
    vals=[acc(m,d) for d in ds_all]
    avgs[m]=np.nanmean(vals)
avgs['P2_Full_projectedC']=np.mean(proj)
for i,(m,v) in enumerate(sorted(avgs.items(), key=lambda x:-x[1]),1):
    tag=' <==' if 'projected' in m else ''
    print(f'{i:2d}. {m:20s} {v:.4f}{tag}')
