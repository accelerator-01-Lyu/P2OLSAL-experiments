"""Quick test: P2_Full on usps to verify redundancy fix."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_dataset
from src.registry import build_registry
from src.runner_generic import run_one

X, y, info = load_dataset('usps')
print(f"usps: n={X.shape[0]}, d={X.shape[1]}, c={info['c']}")

reg = build_registry()
factory, statef = reg['P2_Full']
m = factory()
state = statef('usps', info['c'], 0)

t0 = time.perf_counter()
rows = run_one('usps', m, state, expert='clean', eta=0.0, seed=0,
               budgets=[0.05, 0.10])
el = time.perf_counter() - t0
print(f"P2_Full usps seed=0: {len(rows)} budgets in {el:.1f}s")
for r in rows:
    print(f"  budget={r['budget_frac']:.2f} ACC={r['ACC']:.4f} "
          f"sel_time={r['sel_time']:.2f}s retrain={r['retrain_time']:.2f}s")
print("SUCCESS - no MemoryError")
