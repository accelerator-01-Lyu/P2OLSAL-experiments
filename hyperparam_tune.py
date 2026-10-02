"""P2-OLSAL hyperparameter tuning script.

Does NOT modify any file under src/.  All hyperparameters are injected via
subclassing + per-process config monkey-patching.

Phases:
  1. Quick screen on iris/wine/seeds (3 seeds, budgets 10%/20%, eta=0)
  2. Validation on glass/ecoli (5 seeds)
  3. Confirmation on segment (5 seeds)

Usage:
  python hyperparam_tune.py --phase 1 --workers 4
  python hyperparam_tune.py --phase 2 --workers 4 --top 5
  python hyperparam_tune.py --phase 3 --workers 4
"""
import os, sys, csv, time, argparse, itertools, random
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src import config as C
from src.p2_method import P2OLSAL
from src.robust_selection import robust_gain_dv, compute_redundancy_matrix, compute_alpha, misclassification_proxy
from src.data import load_dataset
from src.runner_generic import run_one

RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Tunable P2 class: adds lam (redundancy weight) and alpha_cross_frac,
# and accepts kappa / rho_expl from constructor.
# alpha0 / eps_q / mc_samples are injected via per-process config patch.
# ---------------------------------------------------------------------------
class TunableP2(P2OLSAL):
    def __init__(self, lam=1.0, alpha_cross_frac=0.5, alpha_within_mult=2.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.alpha_cross_frac = alpha_cross_frac
        self.alpha_within_mult = alpha_within_mult

    def select(self, X, U, V, labeled_mask, budget, c, y_true=None, seed=0):
        self.rng = np.random.RandomState(seed)
        n = X.shape[0]
        unlabeled = np.where(~labeled_mask)[0]
        nu = len(unlabeled)
        if nu <= budget:
            return unlabeled.tolist(), self._diag(U, unlabeled, np.array([]),
                                                  labeled_mask, y_true)

        gains = np.zeros(nu)
        point_gains = np.zeros(nu)
        for k in range(nu):
            i = unlabeled[k]
            if self.robust:
                g_rob, g_pt = robust_gain_dv(U[i], c, kappa=self.kappa,
                                             rng=self.rng)
                gains[k] = g_rob
                point_gains[k] = g_pt
            else:
                g_pt = -np.sum(U[i] * np.log(U[i] + 1e-12))
                gains[k] = g_pt
                point_gains[k] = g_pt

        if self.exploration:
            u_mean = U.mean(axis=0)
            expl = np.linalg.norm(U[unlabeled] - u_mean, axis=1)
            expl = expl / (expl.max() + 1e-12)
            gains = gains + self.rho_expl * expl

        if self.redundancy:
            R = compute_redundancy_matrix(U[unlabeled])
            alpha, _ = compute_alpha(R, nu)
        else:
            R = None
            alpha = 0.0

        if self.block and R is not None:
            block_ids = U[unlabeled].argmax(axis=1)
            alpha_within = alpha * self.alpha_within_mult
            alpha_cross = alpha * self.alpha_cross_frac
            A = np.where(block_ids[:, None] == block_ids[None, :],
                         alpha_within, alpha_cross)
            np.fill_diagonal(A, 0.0)
        else:
            A = np.full((nu, nu), alpha)
            np.fill_diagonal(A, 0.0)

        selected = []
        red_sum = np.zeros(nu)
        available = np.ones(nu, dtype=bool)
        for _ in range(budget):
            if len(selected) > 0:
                scores = gains - self.lam * red_sum / len(selected)
            else:
                scores = gains.copy()
            scores[~available] = -np.inf
            best_k = int(np.argmax(scores))
            if scores[best_k] == -np.inf:
                break
            selected.append(best_k)
            available[best_k] = False
            if R is not None:
                red_sum += A[best_k, :] * R[best_k, :]

        sel_idx = unlabeled[selected]
        info = self._diag(U, sel_idx, point_gains[selected], labeled_mask, y_true)
        return sel_idx.tolist(), info


# ---------------------------------------------------------------------------
# Per-process config patch + worker
# ---------------------------------------------------------------------------
HP_FIELDS = ['kappa', 'rho_expl', 'lam', 'alpha0', 'eps_q', 'mc_samples',
             'alpha_cross_frac']

OUT_FIELDS = (['phase', 'config_id'] + HP_FIELDS +
              ['dataset', 'seed', 'eta', 'budget_frac', 'budget_count',
               'ACC', 'NMI', 'ARI', 'zeta_t', 'sel_time', 'retrain_time',
               'total_time', 'error'])


def _worker(args):
    """Run one (config, dataset, seed) job.  Patches config in this process."""
    cfg, dataset, seed, eta, budgets, phase, config_id = args
    try:
        # Patch config constants for this process only
        C.DIRICHLET_ALPHA0 = cfg['alpha0']
        C.MC_SAMPLES = cfg['mc_samples']
        C.ROBUST_EPS_Q = cfg['eps_q']
        C.ROBUST_KAPPA_BASE = cfg['kappa']

        method = TunableP2(
            robust=True, exploration=True, redundancy=True, fmis=True, block=True,
            kappa=cfg['kappa'], rho_expl=cfg['rho_expl'],
            lam=cfg['lam'], alpha_cross_frac=cfg['alpha_cross_frac'],
        )
        state = {'base_solver': 'plcfcm', 'c': None, 'seed': seed}
        rows = run_one(dataset, method, state, expert='clean', eta=eta,
                       seed=seed, budgets=budgets)
        out = []
        for r in rows:
            row = {'phase': phase, 'config_id': config_id, 'error': ''}
            for k in HP_FIELDS:
                row[k] = cfg[k]
            row.update({
                'dataset': r['dataset'], 'seed': r['seed'], 'eta': r['eta'],
                'budget_frac': r['budget_frac'], 'budget_count': r['budget_count'],
                'ACC': r['ACC'], 'NMI': r['NMI'], 'ARI': r['ARI'],
                'zeta_t': r.get('zeta_t', 0),
                'sel_time': r.get('sel_time', 0),
                'retrain_time': r.get('retrain_time', 0),
                'total_time': r.get('total_time', 0),
            })
            out.append(row)
        return out
    except Exception as e:
        import traceback
        return [{'phase': phase, 'config_id': config_id, 'error': repr(e),
                 'dataset': dataset, 'seed': seed, 'eta': eta,
                 'budget_frac': '', 'ACC': '', 'NMI': '', 'ARI': ''}]


def run_phase(configs, datasets, seeds, eta, budgets, phase, out_csv, workers=4):
    jobs = []
    for cid, cfg in enumerate(configs):
        for ds in datasets:
            for s in seeds:
                jobs.append((cfg, ds, s, eta, budgets, phase, cid))
    print(f'[{time.strftime("%H:%M:%S")}] Phase {phase}: {len(jobs)} jobs '
          f'({len(configs)} configs x {len(datasets)} ds x {len(seeds)} seeds) '
          f'-> {out_csv}')
    t0 = time.perf_counter()
    done = 0
    with mp.Pool(workers, maxtasksperchild=20) as pool, \
         open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for rows in pool.imap_unordered(_worker, jobs, chunksize=1):
            for r in rows:
                w.writerow({k: r.get(k, '') for k in OUT_FIELDS})
            f.flush()
            done += 1
            if done % 20 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                print(f'  [{time.strftime("%H:%M:%S")}] {done}/{len(jobs)} '
                      f'({100*done/len(jobs):.1f}%) '
                      f'{el/60:.1f}m elapsed, '
                      f'ETA {(len(jobs)-done)*el/max(done,1)/60:.1f}m')
    print(f'[{time.strftime("%H:%M:%S")}] Phase {phase} DONE -> {out_csv}')
    return out_csv


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
def generate_random_configs(n, seed=42):
    rng = random.Random(seed)
    kappa_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    rho_vals = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    lam_vals = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    alpha0_vals = [0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
    eps_vals = [0.01, 0.05, 0.1, 0.2]
    cross_vals = [0.1, 0.25, 0.5, 1.0, 2.0]
    configs = []
    seen = set()
    while len(configs) < n:
        cfg = {
            'kappa': rng.choice(kappa_vals),
            'rho_expl': rng.choice(rho_vals),
            'lam': rng.choice(lam_vals),
            'alpha0': rng.choice(alpha0_vals),
            'eps_q': rng.choice(eps_vals),
            'mc_samples': 30,
            'alpha_cross_frac': rng.choice(cross_vals),
        }
        key = tuple(cfg[k] for k in HP_FIELDS)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    # Always include the default config as a reference
    default = {
        'kappa': 0.1, 'rho_expl': 0.1, 'lam': 1.0, 'alpha0': 1.0,
        'eps_q': 0.05, 'mc_samples': 30, 'alpha_cross_frac': 0.5,
    }
    if default not in configs:
        configs.insert(0, default)
    return configs


def load_top_configs(csv_path, top_n=5):
    """Load phase-1 results, rank by mean ACC@10%+20%, return top configs."""
    import collections
    rows_by_cfg = collections.defaultdict(list)
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f):
            if r.get('error'):
                continue
            cid = r['config_id']
            rows_by_cfg[cid].append(r)
    scored = []
    for cid, rows in rows_by_cfg.items():
        accs = [float(r['ACC']) for r in rows if r.get('ACC', '')]
        if not accs:
            continue
        mean_acc = np.mean(accs)
        # Also compute ACC@10% specifically
        acc10 = [float(r['ACC']) for r in rows
                 if r.get('budget_frac') in ('0.1', '0.10')]
        mean10 = np.mean(acc10) if acc10 else 0
        cfg = {k: rows[0][k] for k in HP_FIELDS}
        for k in ['kappa', 'rho_expl', 'lam', 'alpha0', 'eps_q',
                  'mc_samples', 'alpha_cross_frac']:
            cfg[k] = float(cfg[k])
        scored.append((mean10, mean_acc, cfg))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [s[2] for s in scored[:top_n]], scored[:top_n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--n-configs', type=int, default=80)
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--phase1-csv', type=str,
                        default=os.path.join(RESULTS_DIR, 'hp_phase1.csv'))
    parser.add_argument('--phase2-csv', type=str,
                        default=os.path.join(RESULTS_DIR, 'hp_phase2.csv'))
    parser.add_argument('--phase3-csv', type=str,
                        default=os.path.join(RESULTS_DIR, 'hp_phase3.csv'))
    args = parser.parse_args()

    if args.phase == 1:
        configs = generate_random_configs(args.n_configs)
        run_phase(configs, datasets=['iris', 'wine', 'seeds'],
                  seeds=[0, 1, 2], eta=0.0, budgets=[0.10, 0.20],
                  phase=1, out_csv=args.phase1_csv, workers=args.workers)

    elif args.phase == 2:
        top_cfgs, scored = load_top_configs(args.phase1_csv, args.top)
        print(f'Top {args.top} configs from phase 1:')
        for i, (m10, macc, cfg) in enumerate(scored):
            print(f'  #{i+1} ACC@10%={m10:.4f} meanACC={macc:.4f}  {cfg}')
        # Add default config for baseline comparison
        default = {
            'kappa': 0.1, 'rho_expl': 0.1, 'lam': 1.0, 'alpha0': 1.0,
            'eps_q': 0.05, 'mc_samples': 50, 'alpha_cross_frac': 0.5,
        }
        if not any(abs(c['kappa']-0.1)<1e-6 and abs(c['rho_expl']-0.1)<1e-6
                   and abs(c['lam']-1.0)<1e-6 for c in top_cfgs):
            top_cfgs.append(default)
            print('  + default config added for comparison')
        # Bump mc_samples for validation
        for cfg in top_cfgs:
            cfg['mc_samples'] = 50
        run_phase(top_cfgs, datasets=['glass', 'ecoli'],
                  seeds=[0, 1, 2, 3, 4], eta=0.0, budgets=[0.10, 0.20],
                  phase=2, out_csv=args.phase2_csv, workers=args.workers)

    elif args.phase == 3:
        top_cfgs, scored = load_top_configs(args.phase2_csv, 1)
        best = top_cfgs[0]
        best['mc_samples'] = 50
        print(f'Best config from phase 2: {best}')
        print(f'  ACC@10%={scored[0][0]:.4f} meanACC={scored[0][1]:.4f}')
        # Add default for comparison
        default = {
            'kappa': 0.1, 'rho_expl': 0.1, 'lam': 1.0, 'alpha0': 1.0,
            'eps_q': 0.05, 'mc_samples': 50, 'alpha_cross_frac': 0.5,
        }
        configs = [best, default]
        run_phase(configs, datasets=['segment'],
                  seeds=[0, 1, 2, 3, 4], eta=0.0, budgets=[0.10, 0.20],
                  phase=3, out_csv=args.phase3_csv, workers=args.workers)


if __name__ == '__main__':
    main()
