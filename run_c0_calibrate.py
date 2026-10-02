"""c0 calibration via penalty-mass conservation on 6 ORIGINAL datasets only.

c0 = sum m_A / sum (gamma * m_A),  m_A = max(lambda2,1)/m,
over (dataset x calibration-seed x round). Uses ONLY spectral statistics
along the scheme-A trajectory (ACC never read). gamma is the raw block
contrast from the CURRENT membership U each round.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import numpy as np, pandas as pd
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments')
from src.data import load_dataset
from src.p2_method import P2OLSAL
from src.runner_generic import run_one

ORIG6 = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment']
BUDGETS = [0.05, 0.10, 0.15, 0.20]
CALSEEDS = list(range(10))
rows = []
for ds in ORIG6:
    _, _, info = load_dataset(ds)
    c = info['c']
    for s in CALSEEDS:
        m = P2OLSAL(scheme='A', collect_stats=True)
        state = {'base_solver': 'plcfcm', 'c': c, 'seed': s}
        run_one(ds, m, state, expert='clean', eta=0.0, seed=s, budgets=BUDGETS)
        for t in m.alpha_trace:
            mA = max(t['lambda2'], 1.0) / t['m']
            rows.append(dict(dataset=ds, seed=s, gamma=float(t['gamma_raw']),
                             m=float(t['m']), lambda2=float(t['lambda2']),
                             mA=float(mA), gmA=float(t['gamma_raw'] * mA)))
df = pd.DataFrame(rows)
num = df.mA.sum(); den = df.gmA.sum(); c0 = num / den
per = df.groupby('dataset').apply(lambda g: g.mA.sum() / g.gmA.sum(),
                                  include_groups=False)
df.to_csv(r'D:\P2OLSAL_FSS\experiments\results\c0_calibration_detail.csv',
          index=False, encoding='utf-8')
with open(r'D:\P2OLSAL_FSS\experiments\results\c0_value.txt', 'w') as f:
    f.write(f"{c0:.6f}\n")
print(f"FROZEN c0 = {c0:.4f}  (N rounds={len(df)})\nper-dataset c0:")
print(per.round(3).to_string())
