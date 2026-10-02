import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import sys
import numpy as np, pandas as pd
from scipy import stats
R=r'D:\P2OLSAL_FSS\experiments\results'
cmp=pd.read_csv(os.path.join(R,'alpha_scheme_comparison.csv'))
SMALL=['iris','wine','seeds','glass','ecoli','segment','balance','aggregation','compound']
LARGE=['letter','shuttle','usps','fashion']; ALL=SMALL+LARGE
# 4-budget mean ACC per seed, paired C-A
rows=[]
for ds in ALL:
    x=cmp[(cmp.dataset==ds)&(cmp.scheme=='C')].groupby('seed').ACC.mean()
    y=cmp[(cmp.dataset==ds)&(cmp.scheme=='A')].groupby('seed').ACC.mean()
    j=pd.concat([x,y],axis=1,keys=['C','A']).dropna()
    d=(j.C-j.A).values
    wp=stats.wilcoxon(d).pvalue if not np.allclose(d,0) else 1.0
    rows.append((ds,len(d),d.mean(),d.std(ddof=1)/np.sqrt(len(d)),wp))
df=pd.DataFrame(rows,columns=['dataset','n','C-A (4bud mean)','SE','wilcoxon_p'])
print(df.round(4).to_string(index=False))
# per-budget C-A for the three failing small sets
print('\nper-budget C-A:')
for ds in ['ecoli','segment','compound']:
    line=f'  {ds}: '
    for b in [.05,.10,.15,.20]:
        x=cmp[(cmp.dataset==ds)&(cmp.scheme=='C')&(abs(cmp.budget_frac-b)<1e-9)].set_index('seed').ACC
        y=cmp[(cmp.dataset==ds)&(cmp.scheme=='A')&(abs(cmp.budget_frac-b)<1e-9)].set_index('seed').ACC
        j=pd.concat([x,y],axis=1,keys=['C','A']).dropna()
        line+=f'{int(b*100)}%={(j.C-j.A).mean():+.3f}  '
    print(line)
