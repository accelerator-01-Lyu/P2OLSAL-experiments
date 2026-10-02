"""2x2 factorial attribution: cells
  A : cross=0.25, within=2     (baseline; from main table)
  D : cross=0,    within=2     (remove cross-block penalty ONLY)
  E : cross=0.25, within=wB    (raise within via mass conservation ONLY)
  B : cross=0,    within=wB    (both = current scheme B; C==B empirically)
Paired-seed stats vs A at ACC@10% and 4-budget mean. NI margin -0.005.
Small non-inferiority rule (user): paired-diff 95% CI LOWER bound > -0.005.
"""
import os, glob
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, pandas as pd
from scipy import stats
R=r'D:\P2OLSAL_FSS\experiments\results'
SMALL=['iris','wine','seeds','glass','ecoli','segment','balance','aggregation','compound']
LARGE=['letter','shuttle','usps','fashion']; ALL=SMALL+LARGE
BUD=[.05,.10,.15,.20]; NI=-.005
main=pd.read_csv(os.path.join(R,'all_results_merged.csv'),low_memory=False)
A=main[(main.method=='P2_Full')&(main.eta==0)&(main.budget_frac.isin(BUD))].copy()
A['scheme']='A'
fr=pd.concat([pd.read_csv(f,low_memory=False) for f in
              glob.glob(os.path.join(R,'_alpha_stage1_*.csv'))],ignore_index=True)
cols=['scheme','dataset','eta','seed','budget_frac','ACC','NMI','ARI']
cmp=pd.concat([A[cols],fr[cols]],ignore_index=True)

def paired(ds,sc,bf=.10):
    x=cmp[(cmp.dataset==ds)&(cmp.scheme==sc)&(abs(cmp.budget_frac-bf)<1e-9)].set_index('seed').ACC
    y=cmp[(cmp.dataset==ds)&(cmp.scheme=='A')&(abs(cmp.budget_frac-bf)<1e-9)].set_index('seed').ACC
    j=pd.concat([x,y],axis=1,keys=['x','A']).dropna()
    d=(j.x-j.A).values; n=len(d)
    m=d.mean(); sd=d.std(ddof=1); se=sd/np.sqrt(n)
    tcrit=stats.t.ppf(.975,n-1); lo=m-tcrit*se; hi=m+tcrit*se
    tp=stats.ttest_1samp(d,0).pvalue if n>1 else np.nan
    try: wp=stats.wilcoxon(d).pvalue if not np.allclose(d,0) else 1.0
    except Exception: wp=np.nan
    dz=m/sd if sd>0 else np.nan
    return dict(dataset=ds,cell=sc,n=n,diff=m,se=se,ciL=lo,ciH=hi,
                t_p=tp,wilc_p=wp,d=dz,noninf_ci=bool(lo>NI),
                acc=j.x.mean())

for metric,bf in [('@10%',.10)]:
    print(f'===== Paired vs A  ACC{metric} (95% CI; NI if lower>{NI}) =====')
    rows=[]
    for sc in ['D','E','B']:
        for ds in ALL:
            r=paired(ds,sc,bf); rows.append(r)
    pt=pd.DataFrame(rows)
    for sc in ['D','E','B']:
        sub=pt[pt.cell==sc].set_index('dataset')
        print(f'\n--- cell {sc} ---')
        print(sub[['diff','se','ciL','ciH','t_p','wilc_p','d','noninf_ci','acc']].round(4).to_string())
        sl_ok=all(sub.loc[d,'ciL']>NI for d in SMALL if d in sub.index)
        lg={d:sub.loc[d,'diff'] for d in LARGE if d in sub.index}
        print(f'  small NI(CI lower>{NI}) ALL? {sl_ok}; large diffs: '+
              ' '.join(f'{k}={100*v:+.1f}pp' for k,v in lg.items()))
    pt.to_csv(os.path.join(R,'factorial_paired.csv'),index=False,encoding='utf-8')

# 4-cell ACC@10% matrix + 13-dataset means
print('\n===== ACC@10% four-cell matrix =====')
mat=cmp[abs(cmp.budget_frac-.10)<1e-9].groupby(['dataset','scheme']).ACC.mean().unstack('scheme')
mat=mat[['A','D','E','B']]
print(mat.round(4).to_string())
print('\n13-dataset equal-weight means:')
for c in ['A','D','E','B']:
    print(f'  {c}: {mat[c].mean():.4f}')
mat.to_csv(os.path.join(R,'factorial_cell_acc.csv'),encoding='utf-8')
