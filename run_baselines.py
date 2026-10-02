"""Run baseline methods (BADGE, CoreSet, Random, QBC, BALD) on target datasets
for comparison with tuned P2-OLSAL.
"""
import os, sys, csv, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.registry import build_registry
from src.data import load_dataset
from src.runner_generic import run_one

RESULTS_DIR = os.path.join(HERE, 'results')
BUDGETS = [0.10, 0.20]
SEEDS = [0, 1, 2, 3, 4]
DATASETS = ['glass', 'ecoli', 'segment']
METHODS = ['BADGE', 'CoreSet', 'Random', 'QBC', 'BALD', 'Entropy']

FIELDS = ['dataset', 'method', 'seed', 'eta', 'budget_frac', 'budget_count',
          'ACC', 'NMI', 'ARI', 'sel_time', 'retrain_time', 'total_time', 'error']


def _worker(job):
    ds, method_name, seed = job
    try:
        reg = build_registry()
        factory, statef = reg[method_name]
        m = factory()
        _, _, info = load_dataset(ds)
        state = statef(ds, info['c'], seed)
        rows = run_one(ds, m, state, expert='clean', eta=0.0, seed=seed,
                       budgets=BUDGETS)
        out = []
        for r in rows:
            out.append({
                'dataset': r['dataset'], 'method': method_name,
                'seed': r['seed'], 'eta': r['eta'],
                'budget_frac': r['budget_frac'], 'budget_count': r['budget_count'],
                'ACC': r['ACC'], 'NMI': r['NMI'], 'ARI': r['ARI'],
                'sel_time': r.get('sel_time', 0),
                'retrain_time': r.get('retrain_time', 0),
                'total_time': r.get('total_time', 0), 'error': '',
            })
        return out
    except Exception as e:
        return [{'dataset': ds, 'method': method_name, 'seed': seed,
                 'error': repr(e), 'ACC': '', 'NMI': '', 'ARI': ''}]


def main():
    jobs = []
    for ds in DATASETS:
        for m in METHODS:
            for s in SEEDS:
                jobs.append((ds, m, s))
    out_csv = os.path.join(RESULTS_DIR, 'baseline_comparison.csv')
    print(f'[{time.strftime("%H:%M:%S")}] {len(jobs)} baseline jobs -> {out_csv}')
    t0 = time.perf_counter()
    done = 0
    with mp.Pool(4, maxtasksperchild=20) as pool, \
         open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rows in pool.imap_unordered(_worker, jobs, chunksize=1):
            for r in rows:
                w.writerow({k: r.get(k, '') for k in FIELDS})
            f.flush()
            done += 1
            if done % 10 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                print(f'  [{time.strftime("%H:%M:%S")}] {done}/{len(jobs)} '
                      f'({100*done/len(jobs):.1f}%) {el/60:.1f}m')
    print(f'[{time.strftime("%H:%M:%S")}] DONE -> {out_csv}')


if __name__ == '__main__':
    main()
