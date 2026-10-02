"""Rerun segment non-P2 methods that failed (BADGE, CoreSet, GRFCM, etc.)."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys, csv, time, multiprocessing as mp
import pandas as pd, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_dataset
from src.registry import build_registry
from src.runner_generic import run_one

DATASETS = ['segment']
METHODS = ['BADGE', 'CoreSet', 'BALD', 'QBC', 'Entropy', 'Random',
           'SSFCM', 'CEFCM', 'GRFCM', 'PLCFCMPassive']
ETAS = [0.0, 0.10, 0.20, 0.30]
SEEDS = list(range(30))
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]

FIELDS = ['dataset', 'method', 'ablation', 'study', 'expert', 'eta', 'seed',
          'budget_frac', 'budget_count', 'ACC', 'NMI', 'ARI', 'zeta_t',
          'kappa', 'delta_mis_proxy', 'sel_boundary_margin',
          'argmax_retention', 'sel_time', 'retrain_time', 'total_time', 'error']

# Load existing
existing = set()
for f in ['results/master_sweep_20261001_005028.csv'] + sorted(glob.glob('results/rerun_missing_*.csv')):
    try:
        df = pd.read_csv(f, low_memory=False)
        valid = df[df['error'].isna() | (df['error']=='')]
        for _, row in valid.iterrows():
            existing.add((row['dataset'], row['method'], row['eta'], row['seed']))
    except: pass

jobs = []
for ds in DATASETS:
    for m in METHODS:
        for eta in ETAS:
            for s in SEEDS:
                if (ds, m, eta, s) not in existing:
                    jobs.append((ds, m, eta, s))

print(f'Segment non-P2 missing: {len(jobs)} jobs')
from collections import Counter
for m, cnt in Counter(j[1] for j in jobs).most_common():
    print(f'  {m}: {cnt}')

def _worker(job):
    ds, method_name, eta, seed = job
    try:
        reg = build_registry()
        factory, statef = reg[method_name]
        m = factory()
        _, _, info = load_dataset(ds)
        state = statef(ds, info['c'], seed)
        rows = run_one(ds, m, state, expert='clean', eta=eta, seed=seed,
                       budgets=BUDGETS)
        for r in rows:
            r['study'] = 'rerun2'
            r['error'] = ''
        return rows
    except Exception as e:
        return [dict(error=repr(e), study='rerun2', dataset=ds,
                     method=method_name, seed=seed, eta=eta)]

if __name__ == '__main__':
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = f'results/segment_other_{ts}.csv'
    t0 = time.perf_counter()
    done = 0
    with mp.Pool(4) as pool, open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rows in pool.imap_unordered(_worker, jobs, chunksize=1):
            for r in rows:
                w.writerow({k: r.get(k, '') for k in FIELDS})
            f.flush()
            done += 1
            if done % 25 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                print(f'  {done}/{len(jobs)} ({100*done/len(jobs):.0f}%) elapsed {el/60:.1f}m')
    print(f'DONE in {(time.perf_counter()-t0)/60:.1f}m -> {out}')
