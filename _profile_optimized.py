"""Profile optimized P2 on usps and letter."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src.data import load_dataset
from src.plcfcm import plcfcm_fit
from src.robust_selection import robust_gain_dv_batch, compute_redundancy_matrix, compute_alpha

for ds in ['usps', 'letter']:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds}")
    X, y, info = load_dataset(ds)
    n, c = X.shape[0], info['c']
    print(f"n={n}, d={X.shape[1]}, c={c}")

    rng = np.random.RandomState(0)
    init_count = max(c, int(0.02 * n))
    labeled_idx = rng.choice(n, size=init_count, replace=False).tolist()
    labels = y[labeled_idx].copy()

    # PLCFCM retrain (3 runs to get stable timing)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        U, V, it = plcfcm_fit(X, np.array(labeled_idx), labels, c=c, seed=0)
        times.append(time.perf_counter() - t0)
    print(f"PLCFCM: {np.mean(times):.2f}s (iters={it}), best={min(times):.2f}s")

    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[labeled_idx] = True
    unlabeled = np.where(~labeled_mask)[0]
    nu = len(unlabeled)

    # Robust gain
    t0 = time.perf_counter()
    gains, _ = robust_gain_dv_batch(U[unlabeled], c, kappa=1.0, rng=rng)
    print(f"robust_gain_dv_batch: {time.perf_counter()-t0:.3f}s")

    # Full R
    t0 = time.perf_counter()
    R = compute_redundancy_matrix(U[unlabeled])
    print(f"compute_redundancy_matrix ({nu}x{nu}): {time.perf_counter()-t0:.3f}s")

    # Alpha with eigsh (full R)
    t0 = time.perf_counter()
    alpha, lam2 = compute_alpha(R, nu)
    print(f"compute_alpha (eigsh full): {time.perf_counter()-t0:.3f}s, alpha={alpha:.6f}, lambda2={lam2:.4f}")

    # Alpha with subsampled R
    sub_n = min(3000, nu)
    sub_idx = rng.choice(nu, size=sub_n, replace=False)
    t0 = time.perf_counter()
    R_sub = compute_redundancy_matrix(U[unlabeled[sub_idx]])
    alpha_sub, _ = compute_alpha(R_sub, sub_n)
    print(f"compute_alpha (subsampled {sub_n}): {time.perf_counter()-t0:.3f}s, alpha={alpha_sub:.6f}")

    # Estimate full job time
    plcfcm_per = min(times)
    sel_per = 0.5  # rough estimate after optimization
    # 5 budget steps: each has 1 PLCFCM (selection) + 1 PLCFCM (eval) + 1 selection
    est_job = 5 * (2 * plcfcm_per + sel_per)
    print(f"\nEstimated per-job: {est_job:.1f}s")
    n_jobs = 112 if ds == 'usps' else 105
    print(f"Estimated total ({n_jobs} jobs): {est_job * n_jobs / 3600:.2f}h")
