"""P2-only accelerated sweep with tuned hyperparameters.

Only runs P2_Full + 5 ablation variants. Baseline methods reuse old CSV data.
Uses more workers for speed.
"""
import os, csv, time, multiprocessing as mp
from .data import load_dataset
from .registry import build_registry
from .runner_generic import run_one

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')

P2_METHODS = ['P2_Full', 'P2_NoRobust', 'P2_NoExploration',
              'P2_NoRedundancy', 'P2_NoFMIS', 'P2_NoBlock']
ETAS = [0.0, 0.10, 0.20, 0.30]
ABL_ETAS = [0.0, 0.20]
SEEDS_SMALL = list(range(30))
SEEDS_LARGE = list(range(8))
LARGE_THRESHOLD = 2500
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]

FIELDS = ['dataset', 'method', 'ablation', 'study', 'expert', 'eta', 'seed',
          'budget_frac', 'budget_count', 'ACC', 'NMI', 'ARI', 'zeta_t',
          'kappa', 'delta_mis_proxy', 'sel_boundary_margin',
          'argmax_retention', 'sel_time', 'retrain_time', 'total_time', 'error']


def build_jobs(datasets):
    jobs = []
    for ds in datasets:
        X, y, info = load_dataset(ds)
        n = X.shape[0]
        seeds = SEEDS_LARGE if n >= LARGE_THRESHOLD else SEEDS_SMALL
        for m in P2_METHODS:
            etas = ABL_ETAS if m != 'P2_Full' else ETAS
            for eta in etas:
                for s in seeds:
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
            r['study'] = 'p2_tuned'
            r['error'] = ''
        return rows
    except Exception as e:
        import traceback
        return [dict(error=repr(e), study='p2_tuned', dataset=ds,
                     method=method_name, eta=eta, seed=seed,
                     budget_frac=0, ACC=0, NMI=0, ARI=0)]


def run(datasets, workers=14):
    jobs = build_jobs(datasets)
    total = len(jobs)
    print(f'P2-only tuned sweep: {total} jobs, {workers} workers')
    print(f'Datasets: {datasets}')
    print(f'Methods: {P2_METHODS}')

    ts = time.strftime('%Y%m%d_%H%M%S')
    outpath = os.path.join(RES, f'p2_tuned_sweep_{ts}.csv')
    logpath = os.path.join(RES, 'p2_tuned_sweep.log')

    done = 0
    t0 = time.time()
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        with mp.Pool(workers) as pool:
            for rows in pool.imap_unordered(_worker, jobs, chunksize=1):
                for r in rows:
                    w.writerow(r)
                done += 1
                if done % 25 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta_min = (total - done) / rate / 60 if rate > 0 else 0
                    msg = (f'[{time.strftime("%H:%M:%S")}] {done}/{total} '
                           f'({100*done/total:.1f}%) elapsed {elapsed/60:.1f}m '
                           f'eta {eta_min:.1f}m')
                    print(msg, flush=True)
                    with open(logpath, 'a') as lf:
                        lf.write(msg + '\n')
                f.flush()

    print(f'Done. Results: {outpath}')
    return outpath


if __name__ == '__main__':
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else None
    if datasets is None:
        from .config import ALL_DATASETS
        datasets = ALL_DATASETS
    run(datasets)
