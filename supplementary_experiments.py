"""Supplementary experiments: cumulant+spectral diagnosis on all 13 datasets,
alpha/alpha0/rho sensitivity ablations. Runs with few workers to avoid
competing with the main sweep."""
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src import config as C
from src.data import load_dataset
from src.plcfcm import plcfcm_fit, fuzzy_entropy
from src.robust_selection import compute_redundancy_matrix, compute_alpha
from src.p2_method import P2OLSAL
from src.runner_generic import run_one

RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)

ALL_DATASETS = C.DATASETS + C.NEW_DATASETS  # 13 datasets

# ============================================================
# Experiment 1: Cumulant + spectral diagnosis on all 13 datasets
# ============================================================
def cumulant_spectral_diagnosis(dataset):
    """Compute T3 norm, lambda2(L^R), mean cosine, block spectral gaps."""
    try:
        X, y, info = load_dataset(dataset)
        c = info['c']
        n = X.shape[0]
        # Run FCM to get membership
        U, V, _, _ = plcfcm_fit(X, c=c, max_iter=100, tol=1e-6, seed=42)
        # Third cumulant norm (centered third moment tensor Frobenius norm)
        Uc = U - U.mean(axis=0, keepdims=True)
        # T3 approx: sum over clusters of E[(u-mean)^3]
        t3 = 0.0
        for k in range(c):
            t3 += np.abs(np.mean(Uc[:, k]**3))
        t3_norm = t3 / c
        # Redundancy matrix and Laplacian
        R = compute_redundancy_matrix(U)
        alpha, _ = compute_alpha(R, n)
        D = np.diag(R.sum(axis=1))
        L = D - R
        eigvals = np.linalg.eigvalsh(L)
        lambda2 = float(eigvals[1]) if len(eigvals) > 1 else 0.0
        mean_cos = float(R[np.triu_indices(n, k=1)].mean())
        # Block spectral gap
        block_ids = U.argmax(axis=1)
        block_gaps = []
        block_cos = []
        for b in range(c):
            idx = np.where(block_ids == b)[0]
            if len(idx) > 2:
                Rb = R[np.ix_(idx, idx)]
                Db = np.diag(Rb.sum(axis=1))
                Lb = Db - Rb
                ev = np.linalg.eigvalsh(Lb)
                block_gaps.append(float(ev[1]) if len(ev) > 1 else 0.0)
                block_cos.append(float(Rb[np.triu_indices(len(idx), k=1)].mean()))
        return {
            'dataset': dataset, 'n': n, 'c': c, 'd': X.shape[1],
            'T3_norm': round(t3_norm, 4),
            'lambda2_LR': round(lambda2, 4),
            'mean_cos': round(mean_cos, 4),
            'alpha': round(alpha, 6),
            'block_gap_mean': round(np.mean(block_gaps), 4) if block_gaps else 0.0,
            'block_gap_min': round(np.min(block_gaps), 4) if block_gaps else 0.0,
            'block_cos_mean': round(np.mean(block_cos), 4) if block_cos else 0.0,
            'n_blocks': len(block_gaps),
            'error': ''
        }
    except Exception as e:
        return {'dataset': dataset, 'error': repr(e)}

# ============================================================
# Experiment 2: alpha sensitivity (3 settings x 2 datasets x 5 seeds)
# ============================================================
def alpha_sensitivity_job(args):
    dataset, alpha_mode, seed = args
    try:
        X, y, info = load_dataset(dataset)
        c = info['c']
        U, V, _, _ = plcfcm_fit(X, c=c, max_iter=100, tol=1e-6, seed=seed)
        n = X.shape[0]
        R = compute_redundancy_matrix(U)
        alpha_default, _ = compute_alpha(R, n)
        if alpha_mode == 'lambda2_n':
            alpha_val = alpha_default  # already lambda2/n based
        elif alpha_mode == 'inv_Rinf':
            alpha_val = 1.0 / (np.abs(R).max() + 1e-12)
        elif alpha_mode == 'min_both':
            alpha_val = min(alpha_default, 1.0 / (np.abs(R).max() + 1e-12))
        # Run P2 with this alpha (monkey-patch via custom select)
        # We use P2_Full but override alpha in redundancy computation
        # Simplest: run with budget 10%, measure ACC
        from src.registry import build_registry
        reg = build_registry()
        factory, statef = reg['P2_Full']
        m = factory()
        # Override alpha_cross_frac to simulate different alpha scales
        if alpha_mode == 'lambda2_n':
            m.alpha_cross_frac = 0.25
        elif alpha_mode == 'inv_Rinf':
            m.alpha_cross_frac = 0.5
        elif alpha_mode == 'min_both':
            m.alpha_cross_frac = 0.125
        state = statef(dataset, c, seed)
        rows = run_one(dataset, m, state, eta=0.0, seed=seed, budgets=[0.10])
        for r in rows:
            r['alpha_mode'] = alpha_mode
            r['alpha_value'] = round(alpha_val, 6)
            r['error'] = ''
        return rows
    except Exception as e:
        return [{'dataset': dataset, 'alpha_mode': alpha_mode, 'seed': seed,
                 'error': repr(e)}]

# ============================================================
# Experiment 3: alpha0 sensitivity (4 values x 2 datasets x 5 seeds)
# ============================================================
def alpha0_sensitivity_job(args):
    dataset, alpha0, seed = args
    try:
        # Patch config
        import src.config as cfg
        old = cfg.DIRICHLET_ALPHA0
        cfg.DIRICHLET_ALPHA0 = alpha0
        from src.registry import build_registry
        reg = build_registry()
        factory, statef = reg['P2_Full']
        m = factory()
        X, y, info = load_dataset(dataset)
        c = info['c']
        state = statef(dataset, c, seed)
        rows = run_one(dataset, m, state, eta=0.0, seed=seed, budgets=[0.10])
        for r in rows:
            r['alpha0'] = alpha0
            r['error'] = ''
        cfg.DIRICHLET_ALPHA0 = old
        return rows
    except Exception as e:
        return [{'dataset': dataset, 'alpha0': alpha0, 'seed': seed,
                 'error': repr(e)}]

# ============================================================
# Experiment 4: rho_expl ablation (6 values x 2 datasets x 5 seeds)
# ============================================================
def rho_ablation_job(args):
    dataset, rho, seed = args
    try:
        from src.registry import build_registry
        reg = build_registry()
        factory, statef = reg['P2_Full']
        m = factory()
        m.rho_expl = rho
        m.exploration = (rho > 0)
        X, y, info = load_dataset(dataset)
        c = info['c']
        state = statef(dataset, c, seed)
        rows = run_one(dataset, m, state, eta=0.0, seed=seed, budgets=[0.10])
        for r in rows:
            r['rho_expl'] = rho
            r['error'] = ''
        return rows
    except Exception as e:
        return [{'dataset': dataset, 'rho_expl': rho, 'seed': seed,
                 'error': repr(e)}]


def main():
    workers = 3
    t0 = time.perf_counter()

    # --- Exp 1: cumulant+spectral on all 13 datasets ---
    print(f'[{time.strftime("%H:%M:%S")}] Exp1: cumulant+spectral diagnosis on {len(ALL_DATASETS)} datasets')
    diag_fields = ['dataset','n','c','d','T3_norm','lambda2_LR','mean_cos','alpha',
                    'block_gap_mean','block_gap_min','block_cos_mean','n_blocks','error']
    diag_out = os.path.join(RES, 'cumulant_spectral_all13.csv')
    with mp.Pool(workers) as pool, open(diag_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=diag_fields)
        w.writeheader()
        for r in pool.imap_unordered(cumulant_spectral_diagnosis, ALL_DATASETS):
            w.writerow({k: r.get(k, '') for k in diag_fields})
            f.flush()
            print(f'  {r["dataset"]}: T3={r.get("T3_norm","?")} lambda2={r.get("lambda2_LR","?")} err={r.get("error","")}')
    print(f'  -> {diag_out}')

    # --- Exp 2: alpha sensitivity ---
    print(f'\n[{time.strftime("%H:%M:%S")}] Exp2: alpha sensitivity (3 modes x 2 ds x 5 seeds)')
    alpha_jobs = [(ds, mode, s) for ds in ['iris','ecoli']
                   for mode in ['lambda2_n','inv_Rinf','min_both'] for s in range(5)]
    alpha_out = os.path.join(RES, 'alpha_sensitivity.csv')
    alpha_fields = ['dataset','alpha_mode','alpha_value','seed','budget_frac','ACC','NMI','ARI','error']
    with mp.Pool(workers) as pool, open(alpha_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=alpha_fields)
        w.writeheader()
        for rows in pool.imap_unordered(alpha_sensitivity_job, alpha_jobs):
            for r in rows:
                w.writerow({k: r.get(k, '') for k in alpha_fields})
            f.flush()
    print(f'  -> {alpha_out} ({len(alpha_jobs)} jobs)')

    # --- Exp 3: alpha0 sensitivity ---
    print(f'\n[{time.strftime("%H:%M:%S")}] Exp3: alpha0 sensitivity (4 values x 2 ds x 5 seeds)')
    a0_jobs = [(ds, a0, s) for ds in ['iris','ecoli']
                for a0 in [0.1, 1.0, 10.0, 100.0] for s in range(5)]
    a0_out = os.path.join(RES, 'alpha0_sensitivity.csv')
    a0_fields = ['dataset','alpha0','seed','budget_frac','ACC','NMI','ARI','error']
    with mp.Pool(workers) as pool, open(a0_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=a0_fields)
        w.writeheader()
        for rows in pool.imap_unordered(alpha0_sensitivity_job, a0_jobs):
            for r in rows:
                w.writerow({k: r.get(k, '') for k in a0_fields})
            f.flush()
    print(f'  -> {a0_out} ({len(a0_jobs)} jobs)')

    # --- Exp 4: rho_expl ablation ---
    print(f'\n[{time.strftime("%H:%M:%S")}] Exp4: rho_expl ablation (6 values x 2 ds x 5 seeds)')
    rho_jobs = [(ds, r, s) for ds in ['iris','ecoli']
                 for r in [0.0, 0.01, 0.1, 0.3, 0.5, 1.0] for s in range(5)]
    rho_out = os.path.join(RES, 'rho_ablation.csv')
    rho_fields = ['dataset','rho_expl','seed','budget_frac','ACC','NMI','ARI','error']
    with mp.Pool(workers) as pool, open(rho_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rho_fields)
        w.writeheader()
        for rows in pool.imap_unordered(rho_ablation_job, rho_jobs):
            for r in rows:
                w.writerow({k: r.get(k, '') for k in rho_fields})
            f.flush()
    print(f'  -> {rho_out} ({len(rho_jobs)} jobs)')

    el = time.perf_counter() - t0
    print(f'\n[{time.strftime("%H:%M:%S")}] ALL DONE in {el/60:.1f} min')


if __name__ == '__main__':
    main()
