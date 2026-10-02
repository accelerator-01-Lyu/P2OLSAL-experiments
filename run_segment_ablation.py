"""Single-process runner for segment P2 ablation (avoids multiprocessing issues)."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_dataset
from src.registry import build_registry
from src.runner_generic import run_one

DATASETS = ['segment']
ABLATION_METHODS = ['P2_NoRobust', 'P2_NoExploration', 'P2_NoRedundancy',
                    'P2_NoFMIS', 'P2_NoBlock']
ETAS = [0.0, 0.20]
SEEDS = list(range(30))
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]

FIELDS = ['dataset', 'method', 'ablation', 'study', 'expert', 'eta', 'seed',
          'budget_frac', 'budget_count', 'ACC', 'NMI', 'ARI', 'zeta_t',
          'kappa', 'delta_mis_proxy', 'sel_boundary_margin',
          'argmax_retention', 'sel_time', 'retrain_time', 'total_time', 'error']

# Load existing to skip
existing = set()
import pandas as pd
if os.path.exists('results/master_sweep_20261001_005028.csv'):
    df = pd.read_csv('results/master_sweep_20261001_005028.csv', low_memory=False)
    valid = df[df['error'].isna() | (df['error']=='')]
    for _, row in valid.iterrows():
        existing.add((row['dataset'], row['method'], row['eta'], row['seed']))
# Also check rerun files
for f in sorted(__import__('glob').glob('results/rerun_missing_*.csv')):
    try:
        df = pd.read_csv(f, low_memory=False)
        valid = df[df['error'].isna() | (df['error']=='')]
        for _, row in valid.iterrows():
            existing.add((row['dataset'], row['method'], row['eta'], row['seed']))
    except: pass

jobs = []
for ds in DATASETS:
    for m in ABLATION_METHODS:
        for eta in ETAS:
            for s in SEEDS:
                if (ds, m, eta, s) not in existing:
                    jobs.append((ds, m, eta, s))

print(f'Segment P2 ablation: {len(jobs)} missing jobs')
reg = build_registry()

ts = time.strftime('%Y%m%d_%H%M%S')
out = f'results/segment_ablation_{ts}.csv'
t0 = time.perf_counter()
done = 0
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for ds, method_name, eta, seed in jobs:
        try:
            factory, statef = reg[method_name]
            m = factory()
            _, _, info = load_dataset(ds)
            state = statef(ds, info['c'], seed)
            rows = run_one(ds, m, state, expert='clean', eta=eta, seed=seed,
                           budgets=BUDGETS)
            for r in rows:
                r['study'] = 'ablation'
                r['error'] = ''
                w.writerow({k: r.get(k, '') for k in FIELDS})
        except Exception as e:
            w.writerow({'error': repr(e), 'study': 'ablation', 'dataset': ds,
                        'method': method_name, 'seed': seed, 'eta': eta})
        f.flush()
        done += 1
        if done % 10 == 0 or done == len(jobs):
            el = time.perf_counter() - t0
            print(f'  {done}/{len(jobs)} ({100*done/len(jobs):.0f}%) elapsed {el/60:.1f}m, avg {el/done:.1f}s/job')

print(f'DONE in {(time.perf_counter()-t0)/60:.1f}m -> {out}')
