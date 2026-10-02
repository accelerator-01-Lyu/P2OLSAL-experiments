"""Profile P2 selection on usps to find bottlenecks."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src.data import load_dataset
from src.registry import build_registry
from src.plcfcm import plcfcm_fit
from src.robust_selection import robust_gain_dv_batch, compute_redundancy_matrix, compute_alpha

ds = 'usps'
X, y, info = load_dataset(ds)
n, c = X.shape[0], info['c']
print(f"{ds}: n={n}, d={X.shape[1]}, c={c}")

rng = np.random.RandomState(0)
init_count = max(c, int(0.02 * n))
labeled_idx = rng.choice(n, size=init_count, replace=False).tolist()
labels = y[labeled_idx].copy()

# PLCFCM retrain
t0 = time.perf_counter()
U, V, _ = plcfcm_fit(X, np.array(labeled_idx), labels, c=c, seed=0)
t_plcfcm = time.perf_counter() - t0
print(f"PLCFCM retrain: {t_plcfcm:.2f}s")

labeled_mask = np.zeros(n, dtype=bool)
labeled_mask[labeled_idx] = True
unlabeled = np.where(~labeled_mask)[0]
nu = len(unlabeled)
print(f"unlabeled: {nu}")

# Robust gain batch
t0 = time.perf_counter()
gains, point_gains = robust_gain_dv_batch(U[unlabeled], c, kappa=1.0, rng=rng)
t_gain = time.perf_counter() - t0
print(f"robust_gain_dv_batch: {t_gain:.2f}s")

# Redundancy matrix (full dense)
t0 = time.perf_counter()
R = compute_redundancy_matrix(U[unlabeled])
t_R = time.perf_counter() - t0
print(f"compute_redundancy_matrix (full {nu}x{nu}): {t_R:.2f}s")

# Alpha from full R
t0 = time.perf_counter()
alpha, lambda2 = compute_alpha(R, nu)
t_alpha = time.perf_counter() - t0
print(f"compute_alpha (full eigendecomp): {t_alpha:.2f}s")

# Alpha from subsampled R
sub_n = 3000
sub_idx = rng.choice(nu, size=sub_n, replace=False)
t0 = time.perf_counter()
R_sub = compute_redundancy_matrix(U[unlabeled[sub_idx]])
alpha_sub, _ = compute_alpha(R_sub, sub_n)
t_alpha_sub = time.perf_counter() - t0
print(f"compute_alpha (subsampled {sub_n}): {t_alpha_sub:.2f}s, alpha={alpha_sub:.6f} vs full={alpha:.6f}")

# Greedy loop timing
budget = max(1, int(0.05 * n) - len(labeled_idx))
print(f"budget: {budget}")
block_ids = U[unlabeled].argmax(axis=1)
alpha_within = alpha * 2.0
alpha_cross = alpha * 0.25
A = np.where(block_ids[:, None] == block_ids[None, :], alpha_within, alpha_cross)
np.fill_diagonal(A, 0.0)

t0 = time.perf_counter()
selected = []
red_sum = np.zeros(nu)
available = np.ones(nu, dtype=bool)
lam = 0.1
for _ in range(budget):
    if len(selected) > 0:
        scores = gains - lam * red_sum / len(selected)
    else:
        scores = gains.copy()
    scores[~available] = -np.inf
    best_k = int(np.argmax(scores))
    selected.append(best_k)
    available[best_k] = False
    red_sum += A[best_k, :] * R[best_k, :]
t_greedy = time.perf_counter() - t0
print(f"greedy loop ({budget} iters): {t_greedy:.2f}s")

print(f"\nTotal selection estimate: gain={t_gain:.2f} + R={t_R:.2f} + alpha(full)={t_alpha:.2f} + greedy={t_greedy:.2f} = {t_gain+t_R+t_alpha+t_greedy:.2f}s")
print(f"With subsampled alpha: {t_gain+t_R+t_alpha_sub+t_greedy:.2f}s")
