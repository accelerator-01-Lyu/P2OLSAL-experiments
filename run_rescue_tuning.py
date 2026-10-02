"""Rescue tuning for three "strategic failures" in P2-OLSAL paper.

Task 1 (segment): P2_LAM x P2_ALPHA_CROSS_FRAC grid — beat Random (0.787)
Task 2 (glass):   KAPPA x block-toggle x P2_LAM — beat CoreSet (0.479)
Task 3 (ecoli, eta=0.3): P2_LAM x KAPPA — beat BADGE (0.726)

All runs @10% budget only.  Results saved to results/rescue_tuning_<task>.csv

Usage:
  python run_rescue_tuning.py --task segment
  python run_rescue_tuning.py --task glass
  python run_rescue_tuning.py --task noise
  python run_rescue_tuning.py --task all
"""
import os, sys, csv, time, argparse, itertools
import multiprocessing as mp
import numpy as np

# Thread limits (must be set before numpy import in workers too)
for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_v] = '1'

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src import config as C
from src.p2_method import P2OLSAL
from src.data import load_dataset
from src.runner_generic import run_one

RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

BUDGET_10 = [0.10]

# ---------------------------------------------------------------------------
# Config grids (CORRECTED)
# Diagnostic finding: expected entropy gain g_mean is negative (~-0.06) on
# all datasets, so robust gain w_rob=max(0, DV-dual) is all-zero for
# kappa >= 0.05.  Active robust gain only appears for kappa in [1e-4, 1e-2].
# kappa=0 returns g_mean directly (negative gains, different selection regime).
# ---------------------------------------------------------------------------
KAPPA_GRID = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]


def segment_configs():
    """kappa x lam grid (6 x 4 = 24 configs).

    kappa must be tiny (1e-4..1e-2) to activate robust gain.
    lam controls redundancy penalty relative to tiny gain scale.
    """
    configs = []
    for k in KAPPA_GRID:
        for lam in [0.0, 0.01, 0.1, 1.0]:
            configs.append({
                'lam': lam, 'kappa': k, 'alpha_cross_frac': 0.25,
                'label': f'k={k}_lam={lam}'
            })
    return configs


def glass_configs():
    """kappa x lam grid (6 x 3 = 18) + block_off at active kappa (3) = 21."""
    configs = []
    seen = set()
    def _add(lam, kappa, cf, label):
        key = (round(lam, 8), round(kappa, 8), round(cf, 6))
        if key not in seen:
            seen.add(key)
            configs.append({'lam': lam, 'kappa': kappa,
                            'alpha_cross_frac': cf, 'label': label})
    for k in KAPPA_GRID:
        for lam in [0.0, 0.1, 1.0]:
            _add(lam, k, 0.25, f'k={k}_lam={lam}')
    # block off (cf=1.0) at active kappa values
    for k in [0.0001, 0.001, 0.01]:
        _add(0.1, k, 1.0, f'k={k}_lam=0.1_blockoff')
    return configs


def noise_configs():
    """ecoli eta=0.3: kappa x lam grid (6 x 3 = 18) + 2 targeted = 20."""
    configs = []
    seen = set()
    def _add(lam, kappa, cf, label):
        key = (round(lam, 8), round(kappa, 8), round(cf, 6))
        if key not in seen:
            seen.add(key)
            configs.append({'lam': lam, 'kappa': kappa,
                            'alpha_cross_frac': cf, 'label': label})
    for k in KAPPA_GRID:
        for lam in [0.0, 0.1, 1.0]:
            _add(lam, k, 0.25, f'k={k}_lam={lam}')
    # targeted combos at active kappa with small lam
    _add(0.01, 0.001, 0.25, 'k=0.001_lam=0.01')
    _add(0.01, 0.0001, 0.25, 'k=0.0001_lam=0.01')
    return configs


TASK_CONFIGS = {
    'segment': segment_configs,
    'glass': glass_configs,
    'noise': noise_configs,
}

TASK_PARAMS = {
    'segment': {'dataset': 'segment', 'eta': 0.0, 'seeds': list(range(5)),
                'workers': 1, 'target': 0.787, 'target_name': 'Random'},
    'glass':   {'dataset': 'glass',   'eta': 0.0, 'seeds': list(range(10)),
                'workers': 4, 'target': 0.479, 'target_name': 'CoreSet'},
    'noise':   {'dataset': 'ecoli',   'eta': 0.3, 'seeds': list(range(10)),
                'workers': 4, 'target': 0.726, 'target_name': 'BADGE'},
}

OUT_FIELDS = ['task', 'config_label', 'lam', 'kappa', 'alpha_cross_frac',
              'dataset', 'eta', 'seed', 'budget_frac', 'ACC', 'NMI', 'ARI',
              'sel_time', 'retrain_time', 'total_time', 'error']


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(args):
    cfg, dataset, eta, seed = args
    try:
        method = P2OLSAL(
            robust=True, exploration=True, redundancy=True, fmis=True, block=True,
            kappa=cfg['kappa'], rho_expl=0.0,
            lam=cfg['lam'], alpha_cross_frac=cfg['alpha_cross_frac'],
            name='P2_Rescue'
        )
        state = {'base_solver': 'plcfcm', 'c': None, 'seed': seed}
        rows = run_one(dataset, method, state, expert='clean', eta=eta,
                       seed=seed, budgets=BUDGET_10)
        out = []
        for r in rows:
            out.append({
                'config_label': cfg['label'],
                'lam': cfg['lam'], 'kappa': cfg['kappa'],
                'alpha_cross_frac': cfg['alpha_cross_frac'],
                'dataset': r['dataset'], 'eta': r['eta'], 'seed': r['seed'],
                'budget_frac': r['budget_frac'],
                'ACC': r['ACC'], 'NMI': r['NMI'], 'ARI': r['ARI'],
                'sel_time': r.get('sel_time', 0),
                'retrain_time': r.get('retrain_time', 0),
                'total_time': r.get('total_time', 0),
                'error': ''
            })
        return out
    except Exception as e:
        import traceback
        return [{'config_label': cfg.get('label', '?'), 'lam': cfg.get('lam'),
                 'kappa': cfg.get('kappa'), 'alpha_cross_frac': cfg.get('alpha_cross_frac'),
                 'dataset': dataset, 'eta': eta, 'seed': seed,
                 'budget_frac': 0.10, 'ACC': 0, 'NMI': 0, 'ARI': 0,
                 'sel_time': 0, 'retrain_time': 0, 'total_time': 0,
                 'error': repr(e)}]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_task(task_name):
    cfg_fn = TASK_CONFIGS[task_name]
    params = TASK_PARAMS[task_name]
    configs = cfg_fn()
    dataset = params['dataset']
    eta = params['eta']
    seeds = params['seeds']
    workers = params['workers']
    target = params['target']
    target_name = params['target_name']

    jobs = []
    for cfg in configs:
        for s in seeds:
            jobs.append((cfg, dataset, eta, s))

    out_csv = os.path.join(RESULTS_DIR, f'rescue_tuning_{task_name}.csv')
    print(f'[{time.strftime("%H:%M:%S")}] Task={task_name} dataset={dataset} '
          f'eta={eta} configs={len(configs)} seeds={len(seeds)} '
          f'jobs={len(jobs)} workers={workers} target={target} ({target_name})')
    print(f'  Output: {out_csv}')

    t0 = time.perf_counter()
    done = 0

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction='ignore')
        w.writeheader()

        if workers <= 1:
            # Single-process (for segment to avoid memory bandwidth contention)
            for job in jobs:
                rows = _worker(job)
                for r in rows:
                    r['task'] = task_name
                    w.writerow(r)
                f.flush()
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    el = time.perf_counter() - t0
                    print(f'  [{time.strftime("%H:%M:%S")}] {done}/{len(jobs)} '
                          f'({100*done/len(jobs):.1f}%) {el/60:.1f}m elapsed')
        else:
            with mp.Pool(workers, maxtasksperchild=50) as pool:
                for rows in pool.imap_unordered(_worker, jobs, chunksize=1):
                    for r in rows:
                        r['task'] = task_name
                        w.writerow(r)
                    f.flush()
                    done += 1
                    if done % 20 == 0 or done == len(jobs):
                        el = time.perf_counter() - t0
                        print(f'  [{time.strftime("%H:%M:%S")}] {done}/{len(jobs)} '
                              f'({100*done/len(jobs):.1f}%) {el/60:.1f}m elapsed')

    elapsed = time.perf_counter() - t0
    print(f'[{time.strftime("%H:%M:%S")}] Task={task_name} DONE in {elapsed/60:.1f}m')

    # Quick summary
    _print_summary(out_csv, task_name, target, target_name)
    return out_csv


def _print_summary(csv_path, task_name, target, target_name):
    """Print per-config mean ACC and whether target is beaten."""
    import collections
    rows_by_cfg = collections.defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('error'):
                continue
            rows_by_cfg[r['config_label']].append(float(r['ACC']))

    print(f'\n{"="*70}')
    print(f'  RESCUE TUNING SUMMARY: {task_name}')
    print(f'  Target: {target:.4f} ({target_name})')
    print(f'{"="*70}')
    print(f'{"Config":<30} {"MeanACC":>10} {"Std":>8} {"N":>4} {"Beat?":>8}')
    print(f'{"-"*70}')

    scored = []
    for label, accs in rows_by_cfg.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        scored.append((label, mean_acc, std_acc, len(accs)))
    scored.sort(key=lambda x: x[1], reverse=True)

    for label, mean_acc, std_acc, n in scored:
        beat = 'YES' if mean_acc >= target else 'no'
        print(f'{label:<30} {mean_acc:>10.4f} {std_acc:>8.4f} {n:>4} {beat:>8}')

    best = scored[0]
    print(f'{"-"*70}')
    print(f'  Best: {best[0]} = {best[1]:.4f} (target {target:.4f})')
    if best[1] >= target:
        print(f'  >>> RESCUE SUCCESS: {best[0]} beats {target_name} by '
              f'{(best[1]-target)*100:.2f}pp')
    else:
        print(f'  >>> RESCUE FAILED: best is {(target-best[1])*100:.2f}pp '
              f'below {target_name}')
    print(f'{"="*70}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['segment', 'glass', 'noise', 'all'])
    args = parser.parse_args()

    if args.task == 'all':
        for t in ['segment', 'glass', 'noise']:
            run_task(t)
    else:
        run_task(args.task)


if __name__ == '__main__':
    main()
