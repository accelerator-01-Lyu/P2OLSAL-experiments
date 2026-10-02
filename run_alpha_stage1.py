"""Stage-1 alpha-scheme comparison: A/B/C at eta=0, budgets .05-.20.

Usage:
  python run_alpha_stage1.py --tag small --workers 4 \
      --datasets iris wine seeds glass ecoli segment balance aggregation compound \
      --schemes B C --seeds 30
  python run_alpha_stage1.py --tag letter --workers 1 \
      --datasets letter --schemes B C --seeds 8
Output: results/_alpha_stage1_<tag>.csv  (resume-aware)
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, argparse, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, r'D:\P2OLSAL_FSS\experiments')
from src.data import load_dataset
from src.p2_method import P2OLSAL
from src.runner_generic import run_one

BUDGETS = [0.05, 0.10, 0.15, 0.20]
C0 = float(open(r'D:\P2OLSAL_FSS\experiments\results\c0_value.txt').read().strip())
TRACE_KEYS = ['alpha', 'gamma_raw', 'd_eff', 'lambda2', 'r_within', 'r_cross',
              'r_W', 'p', 'w_B', 'PC', 'H_norm']


def job(a):
    ds, scheme, seed, c = a
    m = P2OLSAL(scheme=scheme, c0=C0, name='P2_Full')
    state = {'base_solver': 'plcfcm', 'c': c, 'seed': seed}
    rows = run_one(ds, m, state, expert='clean', eta=0.0, seed=seed, budgets=BUDGETS)
    tr = m.alpha_trace
    for i, r in enumerate(rows):
        r['scheme'] = scheme
        if i < len(tr):
            for k in TRACE_KEYS:
                r[k] = tr[i].get(k, np.nan)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--schemes', nargs='+', default=['B', 'C'])
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    out = rf'D:\P2OLSAL_FSS\experiments\results\_alpha_stage1_{args.tag}.csv'

    tasks, cmap = [], {}
    for ds in args.datasets:
        _, _, info = load_dataset(ds)
        cmap[ds] = info['c']
    done = set()
    if os.path.exists(out):
        old = pd.read_csv(out, low_memory=False)
        for ds, sc, sd in zip(old.dataset, old.scheme, old.seed):
            done.add((ds, sc, int(sd)))
    for ds in args.datasets:
        for sc in args.schemes:
            for sd in range(args.seeds):
                if (ds, sc, sd) not in done:
                    tasks.append((ds, sc, sd, cmap[ds]))
    print(f'c0={C0:.4f}  tasks={len(tasks)} workers={args.workers} -> {out}', flush=True)
    if not tasks:
        print('all done'); return

    t0 = time.time(); n = 0; errs = 0
    if args.workers <= 1:
        for a in tasks:
            try:
                rws = job(a)
                pd.DataFrame(rws).to_csv(out, mode='a',
                    header=not os.path.exists(out), index=False)
                n += 1
            except Exception as e:
                errs += 1; print(f'ERROR {a[:3]}: {e}', flush=True)
            if n % 2 == 0:
                print(f'  {n}/{len(tasks)} done {time.time()-t0:.0f}s errs={errs}',
                      flush=True)
    else:
        with Pool(processes=args.workers) as pool:
            for rws in pool.imap_unordered(job, tasks):
                try:
                    pd.DataFrame(rws).to_csv(out, mode='a',
                        header=not os.path.exists(out), index=False)
                    n += 1
                except Exception as e:
                    errs += 1; print(f'WRITE ERROR: {e}', flush=True)
                if n % 20 == 0:
                    print(f'  {n}/{len(tasks)} done {time.time()-t0:.0f}s', flush=True)
    print(f'FINISHED {n} jobs, errs={errs}, {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
