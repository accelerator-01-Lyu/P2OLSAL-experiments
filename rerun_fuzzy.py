"""Rerun only SSFCM/CEFCM/GRFCM on all 6 datasets (fixed return value bug)."""
import sys, os, csv, time, multiprocessing as mp

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.data import load_dataset
    from src.registry import build_registry
    from src.runner_generic import run_one
    from src import config as C

    DATASETS = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment']
    METHODS = ['SSFCM', 'CEFCM', 'GRFCM']
    ETAS = [0.0, 0.10, 0.20, 0.30]
    SEEDS = list(range(30))
    BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]

    FIELDS = ['dataset', 'method', 'ablation', 'study', 'expert', 'eta', 'seed',
              'budget_frac', 'budget_count', 'ACC', 'NMI', 'ARI', 'zeta_t',
              'kappa', 'delta_mis_proxy', 'sel_boundary_margin',
              'argmax_retention', 'sel_time', 'retrain_time', 'total_time', 'error']

    jobs = []
    for ds in DATASETS:
        for m in METHODS:
            for eta in ETAS:
                for s in SEEDS:
                    jobs.append((ds, m, eta, s))

    print(f'Rerunning {len(jobs)} jobs (3 methods x 6 datasets x 30 seeds x 4 etas)')

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
                r['study'] = 'rerun'
                r['error'] = ''
            return rows
        except Exception as e:
            return [dict(error=repr(e), study='rerun', dataset=ds,
                         method=method_name, seed=seed, eta=eta)]

    ts = time.strftime('%Y%m%d_%H%M%S')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', f'rerun_fuzzy_{ts}.csv')
    t0 = time.perf_counter()
    done = 0
    with mp.Pool(10) as pool, open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rows in pool.imap_unordered(_worker, jobs, chunksize=1):
            for r in rows:
                w.writerow({k: r.get(k, '') for k in FIELDS})
            f.flush()
            done += 1
            if done % 50 == 0:
                el = time.perf_counter() - t0
                print(f'  {done}/{len(jobs)} ({100*done/len(jobs):.0f}%) elapsed {el/60:.1f}m')

    print(f'DONE in {(time.perf_counter()-t0)/60:.1f}m -> {out}')
