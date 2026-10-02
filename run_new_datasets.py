"""Unified runner for the 7 new datasets.

Usage:
  python run_new_datasets.py small     # balance, aggregation, compound (30 seeds, 4 workers)
  python run_new_datasets.py large     # letter, shuttle, usps, fashion (8 seeds)
  python run_new_datasets.py all       # all 7 datasets

For large datasets: P2 methods run single-process; baselines run with 4 workers.
Skip-existing logic: completed (dataset,method,eta,seed) tuples are not re-run.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys, csv, time, glob, multiprocessing as mp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_dataset
from src.registry import build_registry
from src.runner_generic import run_one

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

P2_METHODS = ['P2_Full', 'P2_NoRobust', 'P2_NoExploration',
              'P2_NoRedundancy', 'P2_NoFMIS', 'P2_NoBlock']
BASELINE_METHODS = ['BADGE', 'CoreSet', 'BALD', 'QBC', 'Entropy', 'Random',
                    'SSFCM', 'CEFCM', 'GRFCM', 'PLCFCMPassive']
ALL_METHODS = P2_METHODS + BASELINE_METHODS
ETAS = [0.0, 0.10, 0.20, 0.30]
ABL_ETAS = [0.0, 0.20]
SEEDS_SMALL = list(range(30))
SEEDS_LARGE = list(range(8))
LARGE_THRESHOLD = 2500
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]

SMALL_DATASETS = ['balance', 'aggregation', 'compound']
LARGE_DATASETS = ['letter', 'shuttle', 'usps', 'fashion']

FIELDS = ['dataset', 'method', 'ablation', 'study', 'expert', 'eta', 'seed',
          'budget_frac', 'budget_count', 'ACC', 'NMI', 'ARI', 'zeta_t',
          'kappa', 'delta_mis_proxy', 'sel_boundary_margin',
          'argmax_retention', 'sel_time', 'retrain_time', 'total_time', 'error']


def load_existing():
    """Load all completed job keys from existing CSVs in results/."""
    existing = set()
    patterns = ['master_sweep_*.csv', 'p2_tuned_sweep_*.csv', 'baseline_sweep_*.csv',
                'rerun_missing_*.csv', 'segment_ablation_*.csv', 'segment_other_*.csv',
                'large_sweep_*.csv', 'newdata_*.csv', 'all_results_merged.csv']
    for pat in patterns:
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


def build_jobs(datasets, methods, existing):
    jobs = []
    for ds in datasets:
        X, y, info = load_dataset(ds)
        n = X.shape[0]
        seeds = SEEDS_LARGE if n >= LARGE_THRESHOLD else SEEDS_SMALL
        for m in methods:
            etas = ABL_ETAS if (m in P2_METHODS and m != 'P2_Full') else ETAS
            for eta in etas:
                for s in seeds:
                    if (ds, m, eta, s) not in existing:
                        jobs.append((ds, m, eta, s))
    return jobs


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


def run_mp(jobs, workers, outpath, label):
    """Run jobs with multiprocessing pool."""
    total = len(jobs)
    if total == 0:
        print(f'[{label}] No jobs to run.')
        return outpath
    print(f'[{label}] {total} jobs, {workers} workers -> {outpath}')
    done = 0
    t0 = time.time()
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
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
                    msg = (f'[{label}] {done}/{total} ({100*done/total:.1f}%) '
                           f'elapsed {el/60:.1f}m eta {eta_min:.1f}m')
                    print(msg, flush=True)
    print(f'[{label}] DONE -> {outpath}')
    return outpath


def run_single(jobs, outpath, label):
    """Run jobs single-process (for P2 on large datasets)."""
    total = len(jobs)
    if total == 0:
        print(f'[{label}] No jobs to run.')
        return outpath
    print(f'[{label}] {total} jobs, single-process -> {outpath}')
    reg = build_registry()
    done = 0
    t0 = time.time()
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
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
                    r['study'] = 'newdata'
                    r['error'] = ''
                    w.writerow(r)
            except Exception as e:
                import traceback
                w.writerow(dict(error=repr(e) + '\n' + traceback.format_exc(),
                                 study='newdata', dataset=ds, method=method_name,
                                 eta=eta, seed=seed, budget_frac=0, ACC=0, NMI=0, ARI=0))
            f.flush()
            done += 1
            if done % 5 == 0 or done == total:
                el = time.time() - t0
                avg = el / done
                eta_min = (total - done) * avg / 60
                msg = (f'[{label}] {done}/{total} ({100*done/total:.1f}%) '
                       f'elapsed {el/60:.1f}m avg {avg:.1f}s/job eta {eta_min:.1f}m')
                print(msg, flush=True)
    print(f'[{label}] DONE -> {outpath}')
    return outpath


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_new_datasets.py [small|large|all]')
        sys.exit(1)

    mode = sys.argv[1]
    if mode == 'small':
        datasets = SMALL_DATASETS
    elif mode == 'large':
        datasets = LARGE_DATASETS
    elif mode == 'all':
        datasets = SMALL_DATASETS + LARGE_DATASETS
    else:
        datasets = [mode]  # single dataset name

    ts = time.strftime('%Y%m%d_%H%M%S')
    print(f'=== New dataset sweep: {datasets} ===')
    print(f'Timestamp: {ts}')

    # Load existing completed jobs
    print('Loading existing results...')
    existing = load_existing()
    print(f'  {len(existing)} completed jobs found')

    # Separate small and large datasets
    small_ds = [d for d in datasets if d in SMALL_DATASETS]
    large_ds = [d for d in datasets if d in LARGE_DATASETS]

    outputs = []

    # --- Small datasets: all methods, multiprocessing ---
    if small_ds:
        jobs = build_jobs(small_ds, ALL_METHODS, existing)
        out = os.path.join(RES, f'newdata_small_{ts}.csv')
        run_mp(jobs, workers=4, outpath=out, label='small-all')
        outputs.append(out)

    # --- Large datasets: P2 single-process, baselines multiprocessing ---
    if large_ds:
        # P2 methods (single-process)
        p2_jobs = build_jobs(large_ds, P2_METHODS, existing)
        out_p2 = os.path.join(RES, f'newdata_large_p2_{ts}.csv')
        run_single(p2_jobs, out_p2, label='large-P2')
        outputs.append(out_p2)

        # Baseline methods (4 workers)
        base_jobs = build_jobs(large_ds, BASELINE_METHODS, existing)
        out_base = os.path.join(RES, f'newdata_large_base_{ts}.csv')
        run_mp(base_jobs, workers=4, outpath=out_base, label='large-baseline')
        outputs.append(out_base)

    print('\n=== ALL SWEEPS COMPLETE ===')
    for o in outputs:
        print(f'  {o}')


if __name__ == '__main__':
    main()
