"""Sweep driver for large datasets: 8 seeds instead of 30.

Usage: python -m src.sweep_large letter shuttle usps fashion ng20
"""
import os, csv, time, multiprocessing as mp
from .data import load_dataset
from .registry import build_registry
from .runner_generic import run_one

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')

MAIN_METHODS = ['P2_Full', 'BADGE', 'CoreSet', 'BALD', 'QBC', 'Entropy',
                'Random', 'SSFCM', 'CEFCM', 'GRFCM', 'PLCFCMPassive']
ABLATION_METHODS = ['P2_NoRobust', 'P2_NoExploration', 'P2_NoRedundancy',
                    'P2_NoFMIS', 'P2_NoBlock']
ETAS = [0.0, 0.10, 0.20, 0.30]
ABL_ETAS = [0.0, 0.20]
SEEDS = list(range(8))  # 8 seeds for large datasets
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]

FIELDS = ['dataset', 'method', 'ablation', 'study', 'expert', 'eta', 'seed',
          'budget_frac', 'budget_count', 'ACC', 'NMI', 'ARI', 'zeta_t',
          'kappa', 'delta_mis_proxy', 'sel_boundary_margin',
          'argmax_retention', 'sel_time', 'retrain_time', 'total_time', 'error']


def build_jobs(datasets):
    jobs = []
    for ds in datasets:
        for m in MAIN_METHODS:
            for eta in ETAS:
                for s in SEEDS:
                    jobs.append((ds, m, 'clean', eta, s))
        for m in ABLATION_METHODS:
            for eta in ABL_ETAS:
                for s in SEEDS:
                    jobs.append((ds, m, 'clean', eta, s))
    return jobs


def _worker(job):
    ds, method_name, expert, eta, seed = job
    try:
        reg = build_registry()
        factory, statef = reg[method_name]
        m = factory()
        _, _, info = load_dataset(ds)
        state = statef(ds, info['c'], seed)
        rows = run_one(ds, m, state, expert=expert, eta=eta, seed=seed,
                       budgets=BUDGETS)
        for r in rows:
            r['study'] = 'large'
            r['error'] = ''
        return rows
    except Exception as e:
        return [dict(error=repr(e), study='large', dataset=ds,
                     method=method_name, seed=seed, eta=eta)]


def main(datasets, workers=8):
    jobs = build_jobs(datasets)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = os.path.join(RES, f'large_sweep_{ts}.csv')
    log = os.path.join(RES, 'large_sweep.log')
    n = len(jobs)
    with open(log, 'w') as lf:
        lf.write(f'[{time.strftime("%H:%M:%S")}] start {n} jobs -> {out}\n')
    t0 = time.perf_counter()
    done = 0
    with mp.Pool(workers, maxtasksperchild=1) as pool, \
         open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rows in pool.imap_unordered(_worker, jobs, chunksize=1):
            for r in rows:
                w.writerow({k: r.get(k, '') for k in FIELDS})
            f.flush()
            done += 1
            if done % 10 == 0 or done == n:
                el = time.perf_counter() - t0
                msg = (f'[{time.strftime("%H:%M:%S")}] {done}/{n} '
                       f'({100*done/n:.1f}%) elapsed {el/60:.1f}m '
                       f'eta {(n-done)*el/max(done,1)/60:.1f}m')
                with open(log, 'a') as lf:
                    lf.write(msg + '\n')
    with open(log, 'a') as lf:
        lf.write(f'[{time.strftime("%H:%M:%S")}] DONE {out}\n')
    return out


if __name__ == '__main__':
    import sys
    ds = sys.argv[1:] if len(sys.argv) > 1 else None
    if ds:
        main(ds)
