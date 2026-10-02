"""Dataset-specific lambda grid search for large high-dimensional datasets.

Grid: lam in {0, 0.01, 0.02, 0.05, 0.10}, eta=0, 8 seeds, 4 budgets.
Single-process per dataset (P2 on large datasets must not be multiprocessed).

Usage: python run_lambda_tune_large.py fashion
Output: results/_lambda_tune_<dataset>.csv (concat later into lambda_tune_large.csv)
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys, csv, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_dataset
from src.p2_method import P2OLSAL
from src.runner_generic import run_one

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

LAMBDA_GRID = [0.0, 0.01, 0.02, 0.05, 0.10]
SEEDS_LARGE = list(range(8))
TUNE_BUDGETS = [0.05, 0.10, 0.15, 0.20]

FIELDS = ['dataset', 'method', 'ablation', 'study', 'expert', 'eta', 'seed',
          'budget_frac', 'budget_count', 'ACC', 'NMI', 'ARI', 'zeta_t',
          'kappa', 'delta_mis_proxy', 'sel_boundary_margin',
          'argmax_retention', 'sel_time', 'retrain_time', 'total_time',
          'error', 'lam']


def main():
    ds = sys.argv[1]
    _, _, info = load_dataset(ds)

    # Resume support: skip (lam, seed) combos already in the per-dataset file
    out = os.path.join(RES, f'_lambda_tune_{ds}.csv')
    done = set()
    write_header = True
    if os.path.exists(out) and os.path.getsize(out) > 0:
        old = pd.read_csv(out, low_memory=False)
        old = old[old['error'].isna() | (old['error'] == '')]
        for _, r in old.iterrows():
            done.add((float(r['lam']), int(r['seed'])))
        write_header = False

    total = len(LAMBDA_GRID) * len(SEEDS_LARGE)
    print(f'[lambda-tune {ds}] {total} jobs, {len(done)} already done', flush=True)

    t0 = time.time()
    n_done = 0
    with open(out, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if write_header:
            w.writeheader()
        for lam in LAMBDA_GRID:
            for seed in SEEDS_LARGE:
                if (float(lam), seed) in done:
                    n_done += 1
                    continue
                try:
                    m = P2OLSAL(lam=lam, name='P2_Full')
                    state = {'base_solver': 'plcfcm', 'c': info['c'], 'seed': seed}
                    rows = run_one(ds, m, state, expert='clean', eta=0.0,
                                   seed=seed, budgets=TUNE_BUDGETS)
                    for r in rows:
                        r['study'] = 'lambda_tune'
                        r['error'] = ''
                        r['lam'] = lam
                        w.writerow(r)
                except Exception as e:
                    import traceback
                    w.writerow(dict(error=repr(e) + '\n' + traceback.format_exc(),
                                    study='lambda_tune', dataset=ds, method='P2_Full',
                                    eta=0.0, seed=seed, budget_frac=0,
                                    ACC=0, NMI=0, ARI=0, lam=lam))
                f.flush()
                n_done += 1
                el = time.time() - t0
                print(f'[lambda-tune {ds}] {n_done}/{total} lam={lam} seed={seed} '
                      f'elapsed {el/60:.1f}m', flush=True)

    print(f'[lambda-tune {ds}] DONE -> {out}', flush=True)


if __name__ == '__main__':
    main()
