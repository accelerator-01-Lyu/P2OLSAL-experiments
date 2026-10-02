import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np, time, sys
sys.path.insert(0, '.')
from src.data import load_dataset
from src.registry import build_registry
from src.runner_generic import run_one

X, y, info = load_dataset('segment')
n, c = X.shape[0], info['c']
reg = build_registry()

for name in ['P2_NoRobust','P2_NoExploration','P2_NoRedundancy','P2_NoFMIS','P2_NoBlock']:
    factory, statef = reg[name]
    m = factory()
    state = statef('segment', c, 0)
    print(f'Testing {name}...', end=' ', flush=True)
    t0 = time.perf_counter()
    try:
        rows = run_one('segment', m, state, expert='clean', eta=0.0, seed=0,
                       budgets=[0.05, 0.10, 0.15, 0.20, 0.30])
        t = time.perf_counter() - t0
        acc = rows[1]['ACC']
        print(f'OK in {t:.1f}s, ACC@10%={acc:.3f}')
    except Exception as e:
        t = time.perf_counter() - t0
        print(f'ERROR after {t:.1f}s: {repr(e)[:150]}')
