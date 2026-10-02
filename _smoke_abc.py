import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, time
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments')
import numpy as np
import pandas as pd
from src.data import load_dataset
from src.p2_method import P2OLSAL
from src.runner_generic import run_one

B = [0.05, 0.10, 0.15, 0.20]

def run(ds, scheme, seed=0, eta=0.0, c0=1.07):
    _, _, info = load_dataset(ds)
    m = P2OLSAL(scheme=scheme, c0=c0, name='P2_Full')
    state = {'base_solver': 'plcfcm', 'c': info['c'], 'seed': seed}
    rows = run_one(ds, m, state, expert='clean', eta=eta, seed=seed, budgets=B)
    tr = m.alpha_trace
    return rows, tr

# 1) A identity on shuttle seed0 vs existing
rowsA, trA = run('shuttle', 'A')
df = pd.read_csv(r'D:\P2OLSAL_FSS\experiments\results\all_results_merged.csv', low_memory=False)
old = df[(df.dataset=='shuttle')&(df.method=='P2_Full')&(df.eta==0)&(df.seed==0)]
print('=== A identity check (shuttle seed0) ===')
maxdiff=0
for r in rowsA:
    o = old[old.budget_frac==r['budget_frac']].iloc[0]
    d = max(abs(r['ACC']-o['ACC']),abs(r['NMI']-o['NMI']),abs(r['ARI']-o['ARI']))
    maxdiff=max(maxdiff,d)
print(f'max metric diff A vs existing = {maxdiff:.2e} (must be 0)')

# 2) B and C on a small dataset (ecoli) and large (shuttle, letter)
for ds in ['ecoli','shuttle','letter']:
    for sc in ['B','C']:
        t0=time.time()
        rows,tr = run(ds,sc)
        el=time.time()-t0
        accs=[f"{r['ACC']:.3f}" for r in rows]
        g=[f"{t['gamma_raw']:.3f}" for t in tr]
        am=[f"{t['alpha']:.4f}" for t in tr]
        print(f'{ds:>8} {sc}: {el:.1f}s ACC={accs} gamma={g} alpha={am} d_eff={tr[0]["d_eff"]:.2f} PC={tr[0]["PC"]:.3f}')
