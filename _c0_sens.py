import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import sys
import numpy as np, pandas as pd
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments')
from src.data import load_dataset
from src.p2_method import P2OLSAL
from src.runner_generic import run_one
R=r'D:\P2OLSAL_FSS\experiments\results'
BUD=[0.05,.10,.15,.20]; SEEDS=[0,1,2]
DS=['letter','shuttle','usps','fashion','ecoli','segment']
def a10(rows): return [r['ACC'] for r in rows if abs(r['budget_frac']-.10)<1e-9][0]
out=[]
for ds in DS:
    _,_,info=load_dataset(ds); c=info['c']
    by={0.5:[],1.067:[],2.0:[]}
    for s in SEEDS:
        for c0 in [0.5,1.067,2.0]:
            m=P2OLSAL(scheme='C',c0=c0,name='P2_Full')
            rows=run_one(ds,m,{'base_solver':'plcfcm','c':c,'seed':s},
                         expert='clean',eta=0.0,seed=s,budgets=BUD)
            by[c0].append(a10(rows))
    out.append({'dataset':ds,'C_0.5':np.mean(by[0.5]),
                'C_1.067':np.mean(by[1.067]),'C_2.0':np.mean(by[2.0]),
                'spread_2minus0.5':np.mean(by[2.0])-np.mean(by[0.5])})
df=pd.DataFrame(out)
print(df.round(4).to_string(index=False))
print(f"\nmax |ACC(c0=2)-ACC(c0=.5)| over datasets = {df['spread_2minus0.5'].abs().max():.2e}")
df.to_csv(os.path.join(R,'c0_sensitivity_stage1.csv'),index=False,encoding='utf-8')
