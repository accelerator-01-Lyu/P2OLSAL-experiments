"""Rerun only missing/failed jobs from the main sweep."""
import os
# Limit BLAS threads to avoid oversubscription in multiprocessing
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import sys, csv, time, multiprocessing as mp
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXISTING_CSV = 'results/master_sweep_20261001_005028.csv'
DATASETS = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment']
MAIN_METHODS = ['P2_Full', 'BADGE', 'CoreSet', 'BALD', 'QBC', 'Entropy',
                'Random', 'SSFCM', 'CEFCM', 'GRFCM', 'PLCFCMPassive']
ABLATION_METHODS = ['P2_NoRobust', 'P2_NoExploration', 'P2_NoRedundancy',
                    'P2_NoFMIS', 'P2_NoBlock']
ETAS = [0.0, 0.10, 0.20, 0.30]
ABL_ETAS = [0.0, 0.20]
SEEDS = list(range(30))
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]

FIELDS = ['dataset', 'method', 'ablation', 'study', 'expert', 'eta', 'seed',
          'budget_frac', 'budget_count', 'ACC', 'NMI', 'ARI', 'zeta_t',
          'kappa', 'delta_mis_proxy', 'sel_boundary_margin',
          'argmax_retention', 'sel_time', 'retrain_time', 'total_time', 'error']


def _worker(job):
    from src.data import load_dataset
    from src.registry import build_registry
    from src.runner_generic import run_one
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
            r['study'] = 'rerun'
            r['error'] = ''
        return rows
    except Exception as e:
        return [dict(error=repr(e), study='rerun', dataset=ds,
                     method=method_name, seed=seed, eta=eta)]


def main():
    existing = set()
    if os.path.exists(EXISTING_CSV):
        df = pd.read_csv(EXISTING_CSV, low_memory=False)
        valid = df[df['error'].isna() | (df['error']=='')]
        for _, row in valid.iterrows():
            existing.add((row['dataset'], row['method'], row['eta'], row['seed']))
        print(f'Existing valid jobs: {len(existing)}')

    jobs = []
    for ds in DATASETS:
        for m in MAIN_METHODS:
            for eta in ETAS:
                for s in SEEDS:
                    if (ds, m, eta, s) not in existing:
                        jobs.append((ds, m, eta, s))
        for m in ABLATION_METHODS:
            for eta in ABL_ETAS:
                for s in SEEDS:
                    if (ds, m, eta, s) not in existing:
                        jobs.append((ds, m, eta, s))

    print(f'Total missing jobs to rerun: {len(jobs)}')
    ds_count = Counter(j[0] for j in jobs)
    for ds, cnt in ds_count.most_common():
        print(f'  {ds}: {cnt} jobs')

    ts = time.strftime('%Y%m%d_%H%M%S')
    out = f'results/rerun_missing_{ts}.csv'
    t0 = time.perf_counter()
    done = 0
    with mp.Pool(8) as pool, open(out, 'w', newline='') as f:
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


if __name__ == '__main__':
    main()

