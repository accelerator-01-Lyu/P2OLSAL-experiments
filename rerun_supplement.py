"""Rerun only Exp1 (cumulant+spectral) and Exp2 (alpha sensitivity)."""
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src import config as C
from src.data import load_dataset
from src.plcfcm import plcfcm_fit
from src.robust_selection import compute_redundancy_matrix, compute_alpha
from src.runner_generic import run_one

RES = os.path.join(HERE, 'results')
ALL_DATASETS = C.DATASETS + C.NEW_DATASETS

def cumulant_spectral_diagnosis(dataset):
    try:
        X, y, info = load_dataset(dataset)
        c = info['c']
        n = X.shape[0]
        U, V, _ = plcfcm_fit(X, c=c, max_iter=100, tol=1e-6, seed=42)
        Uc = U - U.mean(axis=0, keepdims=True)
        t3 = 0.0
        for k in range(c):
            t3 += np.abs(np.mean(Uc[:, k]**3))
        t3_norm = t3 / c
        R = compute_redundancy_matrix(U)
        alpha, _ = compute_alpha(R, n)
        D = np.diag(R.sum(axis=1))
        L = D - R
        eigvals = np.linalg.eigvalsh(L)
        lambda2 = float(eigvals[1]) if len(eigvals) > 1 else 0.0
        mean_cos = float(R[np.triu_indices(n, k=1)].mean())
        block_ids = U.argmax(axis=1)
        block_gaps, block_cos = [], []
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
            'n_blocks': len(block_gaps), 'error': ''
        }
    except Exception as e:
        import traceback
        return {'dataset': dataset, 'error': repr(e) + ' | ' + traceback.format_exc()[:200]}

def alpha_sensitivity_job(args):
    dataset, alpha_mode, seed = args
    try:
        X, y, info = load_dataset(dataset)
        c = info['c']
        U, V, _ = plcfcm_fit(X, c=c, max_iter=100, tol=1e-6, seed=seed)
        n = X.shape[0]
        R = compute_redundancy_matrix(U)
        alpha_default, _ = compute_alpha(R, n)
        if alpha_mode == 'lambda2_n':
            alpha_val = alpha_default
        elif alpha_mode == 'inv_Rinf':
            alpha_val = 1.0 / (np.abs(R).max() + 1e-12)
        elif alpha_mode == 'min_both':
            alpha_val = min(alpha_default, 1.0 / (np.abs(R).max() + 1e-12))
        from src.registry import build_registry
        reg = build_registry()
        factory, statef = reg['P2_Full']
        m = factory()
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
        import traceback
        return [{'dataset': dataset, 'alpha_mode': alpha_mode, 'seed': seed,
                 'error': repr(e) + ' | ' + traceback.format_exc()[:200]}]

def main():
    workers = 3
    # Exp1
    print(f'[{time.strftime("%H:%M:%S")}] Exp1: cumulant+spectral on 13 datasets')
    diag_fields = ['dataset','n','c','d','T3_norm','lambda2_LR','mean_cos','alpha',
                    'block_gap_mean','block_gap_min','block_cos_mean','n_blocks','error']
    diag_out = os.path.join(RES, 'cumulant_spectral_all13.csv')
    with mp.Pool(workers) as pool, open(diag_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=diag_fields)
        w.writeheader()
        for r in pool.imap_unordered(cumulant_spectral_diagnosis, ALL_DATASETS):
            w.writerow({k: r.get(k, '') for k in diag_fields})
            f.flush()
            print(f'  {r["dataset"]}: T3={r.get("T3_norm","?")} err={r.get("error","")[:60]}')
    # Exp2
    print(f'\n[{time.strftime("%H:%M:%S")}] Exp2: alpha sensitivity')
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
    print(f'\n[{time.strftime("%H:%M:%S")}] DONE')

if __name__ == '__main__':
    main()
