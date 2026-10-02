import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, time
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments')
import pandas as pd
from src.data import load_dataset
from src.p2_method import P2OLSAL
from src.runner_generic import run_one

ds = 'shuttle'
_, _, info = load_dataset(ds)
t0 = time.time()
m = P2OLSAL(lam=0.1, name='P2_Full')
state = {'base_solver': 'plcfcm', 'c': info['c'], 'seed': 0}
rows = run_one(ds, m, state, expert='clean', eta=0.0, seed=0,
               budgets=[0.05, 0.10, 0.15, 0.20])
print(f'elapsed {time.time()-t0:.1f}s')
for r in rows:
    print(f"  b={r['budget_frac']:.2f} ACC={r['ACC']:.4f} NMI={r['NMI']:.4f} ARI={r['ARI']:.4f}")

# Compare with existing merged results
df = pd.read_csv(r'D:\P2OLSAL_FSS\experiments\results\all_results_merged.csv', low_memory=False)
old = df[(df.dataset == ds) & (df.method == 'P2_Full') & (df.eta == 0) & (df.seed == 0)]
print('existing rows:')
for _, r in old.iterrows():
    print(f"  b={r['budget_frac']:.2f} ACC={r['ACC']:.4f} NMI={r['NMI']:.4f} ARI={r['ARI']:.4f}")
