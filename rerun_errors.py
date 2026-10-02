"""Rerun failed jobs (MemoryError) after main batch completes.

Reads all newdata_*.csv, finds error rows, and reruns those specific jobs
when memory is free (no other processes running).
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys, csv, time, glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_dataset
from src.registry import build_registry
from src.runner_generic import run_one

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]

FIELDS = ['dataset', 'method', 'ablation', 'study', 'expert', 'eta', 'seed',
          'budget_frac', 'budget_count', 'ACC', 'NMI', 'ARI', 'zeta_t',
          'kappa', 'delta_mis_proxy', 'sel_boundary_margin',
          'argmax_retention', 'sel_time', 'retrain_time', 'total_time', 'error']


def find_errors():
    """Find all error jobs across newdata CSVs, dedup by (dataset, method, eta, seed)."""
    errors = set()
    for f in sorted(glob.glob(os.path.join(RES, 'newdata_*.csv'))):
        if os.path.getsize(f) == 0:
            continue
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'error' not in df.columns:
                continue
            err = df[df['error'].notna() & (df['error'] != '')]
            for _, row in err.iterrows():
                errors.add((row['dataset'], row['method'], row['eta'], row['seed']))
        except Exception:
            pass
    return sorted(errors)


def main():
    errors = find_errors()
    print(f"Found {len(errors)} failed jobs to rerun:")
    for ds, m, eta, seed in errors:
        print(f"  {ds} {m} eta={eta} seed={seed}")

    if not errors:
        print("No errors to rerun.")
        return

    ts = time.strftime('%Y%m%d_%H%M%S')
    out = os.path.join(RES, f'newdata_rerun_{ts}.csv')
    reg = build_registry()

    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()

        for i, (ds, method_name, eta, seed) in enumerate(errors):
            print(f"[{i+1}/{len(errors)}] {ds} {method_name} eta={eta} seed={seed}")
            try:
                factory, statef = reg[method_name]
                m = factory()
                _, _, info = load_dataset(ds)
                state = statef(ds, info['c'], seed)
                rows = run_one(ds, m, state, expert='clean', eta=eta, seed=seed,
                               budgets=BUDGETS)
                for r in rows:
                    r['study'] = 'rerun'
                    r['error'] = ''
                    w.writerow(r)
                f.flush()
                print(f"  OK ({len(rows)} rows)")
            except Exception as e:
                import traceback
                w.writerow(dict(error=repr(e) + '\n' + traceback.format_exc(),
                                 study='rerun', dataset=ds, method=method_name,
                                 eta=eta, seed=seed, budget_frac=0,
                                 ACC=0, NMI=0, ARI=0))
                f.flush()
                print(f"  FAILED: {e}")

    print(f"\nRerun DONE -> {out}")


if __name__ == '__main__':
    main()
