"""Run baseline methods (multiprocessing) for large datasets.

Usage: python run_baselines_large.py
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys, csv, time, glob, multiprocessing as mp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_dataset
from src.registry import build_registry
from src.runner_generic import run_one

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

LARGE_DATASETS = ['letter', 'shuttle', 'usps', 'fashion']
BASELINE_METHODS = ['BADGE', 'CoreSet', 'BALD', 'QBC', 'Entropy', 'Random',
                    'SSFCM', 'CEFCM', 'GRFCM', 'PLCFCMPassive']
ETAS = [0.0, 0.10, 0.20, 0.30]
SEEDS_LARGE = list(range(8))
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]

FIELDS = ['dataset', 'method', 'ablation', 'study', 'expert', 'eta', 'seed',
          'budget_frac', 'budget_count', 'ACC', 'NMI', 'ARI', 'zeta_t',
          'kappa', 'delta_mis_proxy', 'sel_boundary_margin',
          'argmax_retention', 'sel_time', 'retrain_time', 'total_time', 'error']


def load_existing():
    existing = set()
    for pat in ['master_sweep_*.csv', 'p2_tuned_sweep_*.csv', 'baseline_sweep_*.csv',
                'rerun_missing_*.csv', 'segment_ablation_*.csv', 'segment_other_*.csv',
                'large_sweep_*.csv', 'newdata_*.csv', 'all_results_merged.csv']:
        for f in sorted(glob.glob(os.path.join(RES, pat))):
            try:
                if os.path.getsize(f) == 0:
                    continue
                df = pd.read_csv(f, low_memory=False)
                if 'error' in df.columns:
                    df = df[df['error'].isna() | (df['error'] == '')]
                for _, row in df.iterrows():
                    existing.add((row['dataset'], row['method'], row['eta'], row['seed']))
            except Exception:
                pass
    return existing


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
            r['study'] = 'newdata'
            r['error'] = ''
        return rows
    except Exception as e:
        import traceback
        return [dict(error=repr(e) + '\n' + traceback.format_exc(),
                     study='newdata', dataset=ds, method=method_name,
                     eta=eta, seed=seed, budget_frac=0, ACC=0, NMI=0, ARI=0)]


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    existing = load_existing()

    jobs = []
    for ds in LARGE_DATASETS:
        for m in BASELINE_METHODS:
            for eta in ETAS:
                for s in SEEDS_LARGE:
                    if (ds, m, eta, s) not in existing:
                        jobs.append((ds, m, eta, s))

    total = len(jobs)
    print(f'Baseline large: {total} jobs, {workers} workers', flush=True)

    ts = time.strftime('%Y%m%d_%H%M%S')
    out = os.path.join(RES, f'newdata_base_large_{ts}.csv')
    done = 0
    t0 = time.time()

    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        with mp.Pool(workers) as pool:
            for rows in pool.imap_unordered(_worker, jobs, chunksize=1):
                for r in rows:
                    w.writerow(r)
                f.flush()
                done += 1
                if done % 10 == 0 or done == total:
                    el = time.time() - t0
                    rate = done / el if el > 0 else 0
                    eta_min = (total - done) / rate / 60 if rate > 0 else 0
                    print(f'[baseline] {done}/{total} ({100*done/total:.1f}%) '
                          f'elapsed {el/60:.1f}m eta {eta_min:.1f}m', flush=True)

    print(f'Baseline large DONE -> {out}', flush=True)


if __name__ == '__main__':
    main()
