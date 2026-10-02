"""Detailed per-dataset analysis of phase results."""
import csv, collections, sys
import numpy as np

HP = ['kappa','rho_expl','lam','alpha0','eps_q','mc_samples','alpha_cross_frac']

def analyze_detail(csv_path):
    rows_by_cfg = collections.defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get('error'):
                continue
            rows_by_cfg[r['config_id']].append(r)

    print(f'{"cid":>4s} {"dataset":>10s} {"ACC@10%":>8s} {"ACC@20%":>8s} {"NMI@10":>8s} {"ARI@10":>8s}')
    print('-' * 60)
    for cid in sorted(rows_by_cfg.keys(), key=int):
        rows = rows_by_cfg[cid]
        cfg = {k: float(rows[0][k]) for k in HP}
        datasets = sorted(set(r['dataset'] for r in rows))
        for ds in datasets:
            ds_rows = [r for r in rows if r['dataset'] == ds]
            acc10 = [float(r['ACC']) for r in ds_rows if r['budget_frac'] in ('0.1','0.10')]
            acc20 = [float(r['ACC']) for r in ds_rows if r['budget_frac'] in ('0.2','0.20')]
            nmi10 = [float(r['NMI']) for r in ds_rows if r['budget_frac'] in ('0.1','0.10')]
            ari10 = [float(r['ARI']) for r in ds_rows if r['budget_frac'] in ('0.1','0.10')]
            print(f'{cid:>4s} {ds:>10s} {np.mean(acc10):8.4f} {np.mean(acc20):8.4f} '
                  f'{np.mean(nmi10):8.4f} {np.mean(ari10):8.4f}')
        # summary
        acc10 = [float(r['ACC']) for r in rows if r['budget_frac'] in ('0.1','0.10')]
        acc20 = [float(r['ACC']) for r in rows if r['budget_frac'] in ('0.2','0.20')]
        print(f'{"":>4s} {"AVG":>10s} {np.mean(acc10):8.4f} {np.mean(acc20):8.4f}  '
              f'k={cfg["kappa"]:.2f} rho={cfg["rho_expl"]:.2f} lam={cfg["lam"]:.1f} '
              f'a0={cfg["alpha0"]:.1f} eps={cfg["eps_q"]:.2f} cross={cfg["alpha_cross_frac"]:.2f}')
        print()

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'results/hp_phase2.csv'
    analyze_detail(path)
