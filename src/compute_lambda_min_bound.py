"""Diagnostic: close the Theorem 4.3 / Appendix C cumulant-remainder bound.

For each of the 13 datasets we rebuild the *exact* current membership matrix U
that produced experiments/results/cumulant_spectral_all13.csv (same loader, same
fuzzifier m=2.0, same init seed, same call), and report:

  * lambda_min_UtU  = min eig of the UNCENTERED Gram U^T U  (n x c row-stochastic U)
  * cond_UtU        = lambda_max / lambda_min of U^T U
  * sigma_min       = sqrt( lambda_min(Sigma) ), Sigma = (1/n) U_c^T U_c the
                      membership SAMPLE covariance, evaluated on the simplex TANGENT
                      subspace. Because every row of U sums to 1, every centered
                      row U_c[i] = u_i - mean(u) is orthogonal to the all-ones
                      vector, so U_c^T U_c has an EXACT structural zero eigenvalue
                      along 1_c; we therefore exclude it and take the smallest
                      POSITIVE (tangent) eigenvalue = eigvalsh(U_c^T U_c)[1].
                      This sigma_min is the Appendix C quantity sigma_min^2 =
                      lambda_min(Sigma) used to whiten (Cov(Y)=I), NOT the (R3)
                      data-space within-cluster scatter eigenvalue.

  * residual_bound_nats = T3_norm^2 / (12 * sigma_min^3)   [Eq. C.4 / C.6]

T3_norm is READ from cumulant_spectral_all13.csv (not recomputed). usps has no
T3 there (the original diagnosis OOM'd while building its n x n redundancy
matrix), so its numeric bound is reported NA.

Reproducibility: thread env vars pinned to 1; deterministic seed=42 FCM.
Run:  python experiments/src/compute_lambda_min_bound.py
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, csv
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))   # experiments/src
EXP_DIR = os.path.dirname(SRC_DIR)                      # experiments
sys.path.insert(0, EXP_DIR)

from src.data import load_dataset
from src.plcfcm import plcfcm_fit

RES = os.path.join(EXP_DIR, 'results')
T3_CSV = os.path.join(RES, 'cumulant_spectral_all13.csv')
OUT_CSV = os.path.join(RES, 'cumulant_bound_diagnostics.csv')

# Exact order requested.
DATASETS = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment',
            'balance', 'aggregation', 'compound',
            'letter', 'shuttle', 'usps', 'fashion']

# ---------- read measured T3_norm from the existing diagnostic CSV ----------
t3_lookup = {}
with open(T3_CSV, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        ds = row['dataset'].strip()
        v = (row.get('T3_norm') or '').strip()
        try:
            t3_lookup[ds] = float(v)
        except ValueError:
            t3_lookup[ds] = np.nan


def sig(x, nd=6):
    """Format x with nd significant digits; NA if missing/non-finite."""
    if x is None or not np.isfinite(x):
        return 'NA'
    return float(f'{x:.{nd}g}')


rows = []
for ds in DATASETS:
    note_parts = []
    try:
        X, y, info = load_dataset(ds)
    except Exception as e:
        rows.append(dict(dataset=ds, n='NA', c='NA', lambda_min_UtU='NA',
                         cond_UtU='NA', sigma_min='NA', T3_norm='NA',
                         residual_bound_nats='NA',
                         note=f'load_dataset failed: {e!r}'))
        continue

    n, c = X.shape[0], int(info['c'])

    # ---- EXACT same U pipeline as rerun_supplement.cumulant_spectral_diagnosis ----
    try:
        U, V, _ = plcfcm_fit(X, c=c, max_iter=100, tol=1e-6, seed=42)
        u_ok = True
    except Exception as e:
        U = None
        u_ok = False
        note_parts.append(f'plcfcm_fit failed: {e!r}')

    t3 = t3_lookup.get(ds, np.nan)

    lam_UtU = cond_UtU = sigma_min = np.nan
    if u_ok:
        row_sum_dev = float(np.abs(U.sum(axis=1) - 1.0).max())
        # uncentered Gram
        evU = np.linalg.eigvalsh(U.T @ U)
        lam_UtU = max(float(evU[0]), 0.0)
        lam_UtU_raw = float(evU[0])
        lam_maxU = float(evU[-1])
        cond_UtU = lam_maxU / lam_UtU if lam_UtU > 0 else np.inf
        if lam_UtU_raw < 0:
            note_parts.append('lambda_min(UtU) negative->0 (roundoff noise)')

        # centered membership covariance Sigma = (1/n) U_c^T U_c
        mu = U.mean(axis=0, keepdims=True)
        Uc = U - mu
        evC = np.linalg.eigvalsh(Uc.T @ Uc)   # ascending
        struct_null = float(evC[0])
        # simplex redundancy: centered rows are orthogonal to 1_c -> one exact 0.
        # take smallest POSITIVE (tangent) eigenvalue = 2nd smallest
        if len(evC) >= 2:
            lam_tan = float(evC[1])
        else:
            lam_tan = float(evC[0])
        lam_tan = max(lam_tan, 0.0)
        lam_min_Sigma = lam_tan / n
        sigma_min = np.sqrt(lam_min_Sigma)
        note_parts.append(
            f'centered-Gram ev[0]={struct_null:.2e}(simplex null);'
            f' tangent min eig(UtUcent)={lam_tan:.4g}; row-sum max dev={row_sum_dev:.1e}')

    # ---- residual bound ----
    resid = np.nan
    if u_ok and np.isfinite(t3):
        if sigma_min <= 0:
            note_parts.append('sigma_min==0: bound divergent/infinite')
        else:
            resid = t3 ** 2 / (12.0 * sigma_min ** 3)
            if resid > 1.0:
                note_parts.append(
                    f'bound {resid:.3g} nats >1: not informative, report form bound only')
            if not np.isfinite(resid):
                note_parts.append('non-finite residual')
    else:
        if not np.isfinite(t3):
            note_parts.append(
                'T3_norm NA in cumulant_spectral_all13.csv (original run OOM building '
                'n x n redundancy matrix); numeric bound skipped per protocol')

    rows.append(dict(
        dataset=ds, n=n, c=c,
        lambda_min_UtU=sig(lam_UtU),
        cond_UtU=sig(cond_UtU),
        sigma_min=sig(sigma_min),
        T3_norm=(f'{t3:.4f}' if np.isfinite(t3) else 'NA'),
        residual_bound_nats=sig(resid),
        note='; '.join(note_parts),
    ))
    print(f'{ds:>12} n={n:5d} c={c:3d} lamUtU={lam_UtU:.4g} '
          f'cond={cond_UtU:.4g} sig={sigma_min:.4g} T3={t3:.4f} resid={resid:.4g}')

fields = ['dataset', 'n', 'c', 'lambda_min_UtU', 'cond_UtU', 'sigma_min',
          'T3_norm', 'residual_bound_nats', 'note']
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f'\nwritten {OUT_CSV}')
