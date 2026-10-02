import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments')
import numpy as np
from src.data import load_dataset
from src.p2_method import P2OLSAL
from src.runner_generic import run_one

B = [0.05, 0.10, 0.15, 0.20]
SEEDS = [0, 1, 2]
DS = ['iris', 'ecoli', 'letter', 'shuttle', 'usps', 'fashion']

def acc10(rows):
    return [r['ACC'] for r in rows if abs(r['budget_frac']-0.10) < 1e-9][0]

for ds in DS:
    _, _, info = load_dataset(ds)
    res = {'A': [], 'B': [], 'C': [], 'NoRed': []}
    gammas = []
    for s in SEEDS:
        for scheme, key, kw in [
            ('A', 'A', dict()),
            ('B', 'B', dict()),
            ('C', 'C', dict(c0=1.07)),
            ('A', 'NoRed', dict(redundancy=False, name='P2_NoRedundancy')),
        ]:
            m = P2OLSAL(scheme=scheme, name=kw.pop('name', 'P2_Full'), **kw)
            state = {'base_solver': 'plcfcm', 'c': info['c'], 'seed': s}
            rows = run_one(ds, m, state, expert='clean', eta=0.0, seed=s, budgets=B)
            res[key].append(acc10(rows))
            if key == 'C' and m.alpha_trace:
                gammas.append(np.mean([t['gamma_raw'] for t in m.alpha_trace]))
    g = np.mean(gammas) if gammas else float('nan')
    print(f"{ds:>9} gamma_C={g:.3f} | " + " ".join(
        f"{k}={np.mean(v):.3f}" for k, v in res.items()) +
          f" | C-A={100*(np.mean(res['C'])-np.mean(res['A'])):+.1f}pp "
          f"C-NoRed={100*(np.mean(res['C'])-np.mean(res['NoRed'])):+.1f}pp")
