"""Merge all phase results into one hyperparam_sweep.csv."""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

FIELDS = ['phase', 'config_id', 'kappa', 'rho_expl', 'lam', 'alpha0', 'eps_q',
          'mc_samples', 'alpha_cross_frac', 'dataset', 'seed', 'eta',
          'budget_frac', 'budget_count', 'ACC', 'NMI', 'ARI', 'zeta_t',
          'sel_time', 'retrain_time', 'total_time', 'error']

out_path = os.path.join(RES, 'hyperparam_sweep.csv')
with open(out_path, 'w', newline='') as fout:
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    w.writeheader()
    for phase, fname in [(1, 'hp_phase1.csv'), (2, 'hp_phase2.csv'), (3, 'hp_phase3.csv')]:
        fpath = os.path.join(RES, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as fin:
            for r in csv.DictReader(fin):
                w.writerow({k: r.get(k, '') for k in FIELDS})

print(f'Merged -> {out_path}')
print(f'Total rows: ', end='')
with open(out_path) as f:
    print(sum(1 for _ in f) - 1)
