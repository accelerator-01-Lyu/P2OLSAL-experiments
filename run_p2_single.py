"""Run P2 methods (single-process) for a single large dataset.

Usage: python run_p2_single.py letter
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

P2_METHODS = ['P2_Full', 'P2_NoRobust', 'P2_NoExploration',
              'P2_NoRedundancy', 'P2_NoFMIS', 'P2_NoBlock']
ETAS = [0.0, 0.10, 0.20, 0.30]
ABL_ETAS = [0.0, 0.20]
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


def main():
    ds = sys.argv[1]
    existing = load_existing()

    jobs = []
    for m in P2_METHODS:
        etas = ABL_ETAS if m != 'P2_Full' else ETAS
        for eta in etas:
            for s in SEEDS_LARGE:
                if (ds, m, eta, s) not in existing:
                    jobs.append((ds, m, eta, s))

    total = len(jobs)
    print(f'[{ds}] P2 single-process: {total} jobs', flush=True)

    reg = build_registry()
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = os.path.join(RES, f'newdata_p2_{ds}_{ts}.csv')
    done = 0
    t0 = time.time()

    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for ds_name, method_name, eta, seed in jobs:
            try:
                factory, statef = reg[method_name]
                m = factory()
                _, _, info = load_dataset(ds_name)
                state = statef(ds_name, info['c'], seed)
                rows = run_one(ds_name, m, state, expert='clean', eta=eta, seed=seed,
                               budgets=BUDGETS)
                for r in rows:
                    r['study'] = 'newdata'
                    r['error'] = ''
                    w.writerow(r)
            except Exception as e:
                import traceback
                w.writerow(dict(error=repr(e) + '\n' + traceback.format_exc(),
                                 study='newdata', dataset=ds_name, method=method_name,
                                 eta=eta, seed=seed, budget_frac=0, ACC=0, NMI=0, ARI=0))
            f.flush()
            done += 1
            if done % 2 == 0 or done == total:
                el = time.time() - t0
                avg = el / done
                eta_min = (total - done) * avg / 60
                print(f'[{ds_name}] {done}/{total} ({100*done/total:.0f}%) '
                      f'elapsed {el/60:.1f}m avg {avg:.1f}s/job eta {eta_min:.1f}m',
                      flush=True)

    print(f'[{ds}] P2 DONE -> {out}', flush=True)


if __name__ == '__main__':
    main()
