"""Analyze phase results and print rankings."""
import csv, collections, sys
import numpy as np

HP = ['kappa','rho_expl','lam','alpha0','eps_q','mc_samples','alpha_cross_frac']

def analyze(csv_path, top_n=15):
    rows_by_cfg = collections.defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get('error'):
                continue
            rows_by_cfg[r['config_id']].append(r)

    scored = []
    for cid, rows in rows_by_cfg.items():
        accs = [float(r['ACC']) for r in rows if r.get('ACC','')]
        acc10 = [float(r['ACC']) for r in rows if r.get('budget_frac') in ('0.1','0.10')]
        acc20 = [float(r['ACC']) for r in rows if r.get('budget_frac') in ('0.2','0.20')]
        if not accs:
            continue
        cfg = {k: float(rows[0][k]) for k in HP}
        scored.append((np.mean(acc10), np.mean(acc20), np.mean(accs), cfg, cid))

    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    print(f'Total configs: {len(scored)}')

    # Find default
    for rank, s in enumerate(scored):
        cfg = s[3]
        if (abs(cfg['kappa']-0.1)<1e-6 and abs(cfg['rho_expl']-0.1)<1e-6
                and abs(cfg['lam']-1.0)<1e-6 and abs(cfg['alpha0']-1.0)<1e-6):
            print(f'Default config rank=#{rank+1}: ACC@10%={s[0]:.4f} '
                  f'ACC@20%={s[1]:.4f} meanACC={s[2]:.4f} cid={s[4]}')
            break

    print(f'\nTop {top_n} by ACC@10%:')
    for i,(a10,a20,macc,cfg,cid) in enumerate(scored[:top_n]):
        print(f'#{i+1:2d} cid={cid:>3s} ACC10={a10:.4f} ACC20={a20:.4f} '
              f'mean={macc:.4f} | k={cfg["kappa"]:.2f} '
              f'rho={cfg["rho_expl"]:.2f} lam={cfg["lam"]:.1f} '
              f'a0={cfg["alpha0"]:.1f} eps={cfg["eps_q"]:.2f} '
              f'cross={cfg["alpha_cross_frac"]:.2f}')
    return scored

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'results/hp_phase1.csv'
    analyze(path)
