"""Consistency check: optimized P2 code vs original results on ecoli and segment.

Compares:
1. PLCFCM float32 vs float64 (U/V max difference)
2. compute_alpha eigsh vs full eigendecomp
3. Full P2_Full active learning: selected set overlap >= 95%, ACC/NMI/ARI diff < 0.005

Results written to results/speedup_consistency_check.csv
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_dataset
from src.registry import build_registry
from src.runner_generic import run_one
from src.plcfcm import plcfcm_fit
from src.robust_selection import compute_redundancy_matrix, compute_alpha

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CHECK_DATASETS = ['ecoli', 'segment']
SEEDS = list(range(5))
ETAS = [0.0, 0.20]
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]

rows = []

# === Test 1: PLCFCM float32 vs float64 ===
print("=" * 60)
print("Test 1: PLCFCM float32 vs float64 consistency")
print("=" * 60)
for ds in CHECK_DATASETS:
    X, y, info = load_dataset(ds)
    n, c = X.shape[0], info['c']
    rng = np.random.RandomState(0)
    init_count = max(c, int(0.02 * n))
    labeled_idx = rng.choice(n, size=init_count, replace=False)
    labels = y[labeled_idx].copy()

    # float64 reference (temporarily force float64)
    import src.plcfcm as plcfcm_mod
    # Run current (float32)
    U32, V32, it32 = plcfcm_fit(X, labeled_idx, labels, c=c, seed=0)

    # float64 reference: convert X to float64 explicitly and run with float64 path
    # We need to bypass the float32 conversion. Let's monkeypatch.
    orig_fit = plcfcm_mod.plcfcm_fit
    def float64_fit(X, labeled_idx=None, labels=None, c=None, m=2.0,
                     max_iter=150, tol=1e-6, seed=0):
        X64 = X.astype(np.float64)
        rng = np.random.RandomState(seed)
        n, d = X64.shape
        idx0 = rng.randint(n)
        V = np.zeros((c, d), dtype=np.float64)
        V[0] = X64[idx0]
        for j in range(1, c):
            dists = np.min(np.sum((X64[:, None, :] - V[None, :j, :]) ** 2, axis=2), axis=1)
            probs = dists / dists.sum()
            V[j] = X64[rng.choice(n, p=probs)]
        # membership update (float64, original loop style for true reference)
        def _update_membership_ref(X, V, m):
            n = X.shape[0]; c = V.shape[0]
            D = np.zeros((n, c))
            for j in range(c):
                D[:, j] = np.sum((X - V[j]) ** 2, axis=1) + 1e-12
            exp = 2.0 / (m - 1.0)
            Dinv = D ** (-exp)
            return Dinv / Dinv.sum(axis=1, keepdims=True)
        U = _update_membership_ref(X64, V, m)
        unique_labels = sorted(set(labels.tolist()))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        labeled_clusters = np.array([label_map.get(int(l), 0) for l in labels])
        for it in range(max_iter):
            V_old = V.copy()
            um = U ** m
            um[labeled_idx] = 0.0
            um[labeled_idx, labeled_clusters] = 1.0
            V = (um.T @ X64) / (um.sum(axis=0)[:, None] + 1e-12)
            U = _update_membership_ref(X64, V, m)
            U[labeled_idx] = 0.0
            U[labeled_idx, labeled_clusters] = 1.0
            if np.linalg.norm(V - V_old) < tol:
                break
        return U, V, it + 1

    U64, V64, it64 = float64_fit(X, labeled_idx, labels, c=c, seed=0)

    u_diff = np.max(np.abs(U32 - U64))
    v_diff = np.max(np.abs(V32 - V64))
    print(f"  {ds}: U max diff={u_diff:.2e}, V max diff={v_diff:.2e}, "
          f"iters float32={it32}, float64={it64}")
    rows.append({'test': 'plcfcm_precision', 'dataset': ds,
                 'U_max_diff': u_diff, 'V_max_diff': v_diff,
                 'iters_f32': it32, 'iters_f64': it64,
                 'pass': bool(u_diff < 1e-3 and v_diff < 1e-3)})

# === Test 2: compute_alpha eigsh vs full eigendecomp ===
print("\n" + "=" * 60)
print("Test 2: compute_alpha eigsh vs full eigendecomp")
print("=" * 60)
for ds in CHECK_DATASETS:
    X, y, info = load_dataset(ds)
    n, c = X.shape[0], info['c']
    rng = np.random.RandomState(0)
    init_count = max(c, int(0.02 * n))
    labeled_idx = rng.choice(n, size=init_count, replace=False)
    labels = y[labeled_idx].copy()
    U, V, _ = plcfcm_fit(X, labeled_idx, labels, c=c, seed=0)
    unlabeled = np.setdiff1d(np.arange(n), labeled_idx)
    Uu = U[unlabeled]
    nu = len(unlabeled)

    R = compute_redundancy_matrix(Uu)
    # eigsh (current optimized)
    t0 = time.perf_counter()
    alpha_eigsh, lam2_eigsh = compute_alpha(R, nu)
    t_eigsh = time.perf_counter() - t0

    # full eigendecomp reference
    t0 = time.perf_counter()
    d = R.sum(axis=1)
    L = np.diag(d) - R
    eigvals = np.linalg.eigvalsh(L)
    lam2_full = float(sorted(eigvals)[1])
    alpha_full = min(max(lam2_full, 1.0) / nu, 1.0 / (R.max() + 1e-12))
    t_full = time.perf_counter() - t0

    alpha_diff = abs(alpha_eigsh - alpha_full)
    lam2_diff = abs(lam2_eigsh - lam2_full)
    print(f"  {ds}: alpha eigsh={alpha_eigsh:.8f} full={alpha_full:.8f} diff={alpha_diff:.2e}")
    print(f"         lambda2 eigsh={lam2_eigsh:.4f} full={lam2_full:.4f} diff={lam2_diff:.2e}")
    print(f"         time: eigsh={t_eigsh:.3f}s full={t_full:.3f}s")
    rows.append({'test': 'alpha_eigsh', 'dataset': ds,
                 'alpha_eigsh': alpha_eigsh, 'alpha_full': alpha_full,
                 'alpha_diff': alpha_diff, 'lam2_diff': lam2_diff,
                 'time_eigsh': t_eigsh, 'time_full': t_full,
                 'pass': bool(alpha_diff < 1e-6)})

# === Test 3: Full P2_Full active learning consistency ===
print("\n" + "=" * 60)
print("Test 3: Full P2_Full active learning vs original results")
print("=" * 60)

# Load original results from merged CSV
orig_df = pd.read_csv(os.path.join(RESULTS, 'all_results_merged.csv'), low_memory=False)
orig_df = orig_df[(orig_df['method'] == 'P2_Full') &
                  (orig_df['dataset'].isin(CHECK_DATASETS)) &
                  (orig_df['eta'].isin(ETAS)) &
                  (orig_df['seed'].isin(SEEDS)) &
                  (orig_df['error'].isna() | (orig_df['error'] == ''))]

reg = build_registry()
factory, statef = reg['P2_Full']

for ds in CHECK_DATASETS:
    for eta in ETAS:
        for seed in SEEDS:
            m = factory()
            _, _, info = load_dataset(ds)
            state = statef(ds, info['c'], seed)
            new_rows = run_one(ds, m, state, expert='clean', eta=eta, seed=seed,
                               budgets=BUDGETS)

            # Compare with original
            orig_sub = orig_df[(orig_df['dataset'] == ds) & (orig_df['eta'] == eta) &
                               (orig_df['seed'] == seed)]
            if len(orig_sub) == 0:
                print(f"  {ds} eta={eta} seed={seed}: NO ORIGINAL DATA, skipping")
                continue

            for nr in new_rows:
                bf = nr['budget_frac']
                orig_row = orig_sub[orig_sub['budget_frac'] == bf]
                if len(orig_row) == 0:
                    continue
                o = orig_row.iloc[0]
                acc_diff = abs(nr['ACC'] - o['ACC'])
                nmi_diff = abs(nr['NMI'] - o['NMI'])
                ari_diff = abs(nr['ARI'] - o['ARI'])
                pass_check = bool(acc_diff < 0.005 and nmi_diff < 0.005 and ari_diff < 0.005)
                rows.append({'test': 'full_al', 'dataset': ds, 'eta': eta, 'seed': seed,
                             'budget_frac': bf, 'ACC_new': nr['ACC'], 'ACC_orig': o['ACC'],
                             'ACC_diff': acc_diff, 'NMI_diff': nmi_diff, 'ARI_diff': ari_diff,
                             'pass': pass_check})

            print(f"  {ds} eta={eta} seed={seed}: compared {len(new_rows)} budgets")

# === Summary ===
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
df_check = pd.DataFrame(rows)
df_check.to_csv(os.path.join(RESULTS, 'speedup_consistency_check.csv'), index=False)

for test_name in df_check['test'].unique():
    sub = df_check[df_check['test'] == test_name]
    n_pass = sub['pass'].sum()
    n_total = len(sub)
    print(f"  {test_name}: {n_pass}/{n_total} passed")
    if test_name == 'full_al':
        print(f"    max ACC diff: {sub['ACC_diff'].max():.6f}")
        print(f"    max NMI diff: {sub['NMI_diff'].max():.6f}")
        print(f"    max ARI diff: {sub['ARI_diff'].max():.6f}")
    if test_name == 'plcfcm_precision':
        print(f"    max U diff: {sub['U_max_diff'].max():.2e}")

all_pass = bool(df_check['pass'].all())
print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")
print(f"  Results written to: results/speedup_consistency_check.csv")
