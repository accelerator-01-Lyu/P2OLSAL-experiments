"""Record per-round block-contrast gamma trajectory for P2_Full active learning.

PURE OBSERVATION HOOK (does not change selection, RNG, or algorithm semantics):
  - Replicates run_one's exact outer loop (init rng, plcfcm_fit(seed=seed),
    P2OLSAL.select(seed=seed)) but drives it in small batches (~1% n) so that a
    PLCFCM-refit happens at every round.
  - After every plcfcm_fit, a read-only callback computes structural stats from
    the freshly fitted U using only LOCAL variables; it never touches the
    method's rng, never writes into U, and takes no part in scoring/branching.
  - block contrast (r_within, r_cross, gamma) is computed on the UNLABELED pool
    Uu via an O(m*c) block-sum vectorization (mathematically identical to
    block_contrast_stats / dense R, but builds NO m x m redundancy matrix).
  - PC and H_norm use the FULL U, exactly via p2_method.partition_diagnostics
    (same normalization as alpha_structure).

Writes: results/gamma_trajectory.csv  (long table)
Columns: scheme,dataset,seed,round,n_labeled,gamma,r_within,r_cross,PC,H_norm

Self-checks (printed, not written to the CSV):
  1. ecoli: hook ON vs hook OFF -> selected idx sequence must be identical.
  2. ecoli: O(m*c) block-sum vs O(m^2) dense block_contrast_stats -> must match.
"""
import os
# --- single-thread BLAS (mandatory) BEFORE importing numpy ---
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import csv
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # .../experiments/src
ROOT = os.path.dirname(HERE)                                # .../experiments
sys.path.insert(0, ROOT)

from src.data import load_dataset
from src.plcfcm import plcfcm_fit
from src.p2_method import P2OLSAL, block_contrast_stats, partition_diagnostics
from src.robust_selection import compute_redundancy_matrix

OUT_CSV = os.path.join(ROOT, 'results', 'gamma_trajectory.csv')

FIELDS = ['scheme', 'dataset', 'seed', 'round', 'n_labeled',
          'gamma', 'r_within', 'r_cross', 'PC', 'H_norm']

DATASETS = ['fashion', 'letter', 'shuttle', 'usps']
SEEDS = [0, 1, 2]
TARGET_BUDGET = 0.20
BATCH_FRAC = 0.01          # ~1% of n per round -> ~18 rounds from 2% to 20%


def block_contrast_exact(Uu, c):
    """Exact block-contrast on unlabeled membership Uu (m, c), O(m*c).

    Equivalent to block_contrast_stats(Uu, R, c) with R = cosine(Uu), but no
    m x m matrix is ever formed. Returns (r_within, r_cross, gamma).

      Un_i = Uu_i / ||Uu_i|| ; S_k = sum_{i: block_i=k} Un_i ; n_k = block size
      r_within = sum_{i<j, same block} cos(i,j) / #within pairs
               = [sum_k (||S_k||^2 - n_k)/2] / [sum_k n_k(n_k-1)/2]
      r_cross  = sum_{i<j, diff block} cos(i,j) / #cross pairs
               = [ (||sum_k S_k||^2 - sum_k ||S_k||^2)/2 ] / [ (m^2 - sum n_k^2)/2 ]
      gamma    = clip((r_within - r_cross)/(1 - r_cross), 0, 1)
    """
    m = Uu.shape[0]
    if m < 2:
        return 0.0, 0.0, 0.0
    norms = np.linalg.norm(Uu, axis=1, keepdims=True) + 1e-12
    Un = Uu / norms                       # unit-length membership rows (copy)
    blocks = Uu.argmax(axis=1)
    S = np.zeros((c, Un.shape[1]), dtype=np.float64)
    np.add.at(S, blocks, Un)             # block sums
    counts = np.bincount(blocks, minlength=c).astype(np.float64)

    s2 = np.sum(S ** 2, axis=1)          # ||S_k||^2
    within_pairs = 0.5 * np.sum(counts * (counts - 1.0))
    within_sum = 0.5 * np.sum(s2 - counts)
    total = S.sum(axis=0)
    cross_pairs = 0.5 * (m * m - np.sum(counts ** 2))
    cross_sum = 0.5 * (np.dot(total, total) - np.sum(s2))

    r_within = float(within_sum / within_pairs) if within_pairs > 0 else 0.0
    r_cross = float(cross_sum / cross_pairs) if cross_pairs > 0 else 0.0
    denom = 1.0 - r_cross
    gamma = float(np.clip((r_within - r_cross) / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
    return r_within, r_cross, gamma


def observe(U, labeled_idx, n, c, dataset, seed, round_no):
    """Read-only observation. Returns one CSV row dict; mutates nothing."""
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[list(labeled_idx)] = True
    unlabeled = np.where(~labeled_mask)[0]
    Uu = U[unlabeled]
    r_within, r_cross, gamma = block_contrast_exact(Uu, c)
    pc, h_norm = partition_diagnostics(U)     # FULL U, alpha_structure normalization
    return {
        'scheme': 'P2_Full',
        'dataset': dataset,
        'seed': int(seed),
        'round': int(round_no),
        'n_labeled': int(len(labeled_idx)),
        'gamma': float(gamma),
        'r_within': float(r_within),
        'r_cross': float(r_cross),
        'PC': float(pc),
        'H_norm': float(h_norm),
    }


def run_one_seed(dataset, seed, target_budget=TARGET_BUDGET,
                 batch_frac=BATCH_FRAC, hook=True):
    """Drive P2_Full in small batches; optionally record U stats each round.

    Returns (rows, selected_idx_list). rows is [] when hook=False.
    """
    X, y, info = load_dataset(dataset)
    n, c = X.shape[0], info['c']
    rng = np.random.RandomState(seed)
    init_count = max(c, int(0.02 * n))
    labeled_idx = rng.choice(n, size=init_count, replace=False).tolist()
    labels = y[labeled_idx].copy()

    method = P2OLSAL(name='P2_Full', scheme='A')
    target = int(round(target_budget * n))
    batch = max(c, int(round(batch_frac * n)))

    rows = []
    selected_all = []
    round_no = 0
    while True:
        # PLCFCM refit (own internal RandomState(seed); deterministic)
        U, V, _ = plcfcm_fit(X, np.array(labeled_idx), labels, c=c, seed=seed)

        # ---- pure observation hook (local vars only; no rng, no mutation) ----
        if hook:
            rows.append(observe(U, labeled_idx, n, c, dataset, seed, round_no))

        if len(labeled_idx) >= target:
            break

        b = min(batch, target - len(labeled_idx))
        labeled_mask = np.zeros(n, dtype=bool)
        labeled_mask[labeled_idx] = True
        new_idx, _diag = method.select(X, U, V, labeled_mask, b, c,
                                       y_true=y, seed=seed)
        selected_all.extend(list(new_idx))
        labeled_idx.extend(list(new_idx))
        labels = np.concatenate([labels, y[np.array(new_idx)]])
        round_no += 1

    return rows, selected_all


def selfcheck():
    print('=' * 70)
    print('SELF-CHECK 1: hook on/off -> selected idx sequence on ecoli')
    rows_h, sel_h = run_one_seed('ecoli', seed=0, hook=True)
    _, sel_n = run_one_seed('ecoli', seed=0, hook=False)
    sel_h = np.array(sel_h); sel_n = np.array(sel_n)
    print('  hook ON  selected=%d  first=%s' % (len(sel_h), sel_h[:8].tolist()))
    print('  hook OFF selected=%d  first=%s' % (len(sel_n), sel_n[:8].tolist()))
    assert len(sel_h) == len(sel_n), 'length mismatch: %d vs %d' % (len(sel_h), len(sel_n))
    assert np.array_equal(sel_h, sel_n), 'HOOK CHANGED THE SELECTION SEQUENCE!'
    print('  >>> ASSERT PASSED: selected idx sequences identical (hook on/off).')
    print('  ecoli hook rows recorded (rounds=%d, n_labeled span %d..%d)'
          % (len(rows_h), rows_h[0]['n_labeled'], rows_h[-1]['n_labeled']))

    print('SELF-CHECK 2: O(m*c) block-sum == O(m^2) dense block_contrast_stats')
    X, y, info = load_dataset('ecoli')
    c = info['c']
    rng = np.random.RandomState(0)
    li = rng.choice(X.shape[0], size=max(c, int(0.02 * X.shape[0])), replace=False)
    U, _, _ = plcfcm_fit(X, li, y[li], c=c, seed=0)
    um = U[np.setdiff1d(np.arange(X.shape[0]), li)]
    rw, rc, g = block_contrast_exact(um, c)
    st = block_contrast_stats(um, compute_redundancy_matrix(um), c)
    print('  exact: r_within=%.6f r_cross=%.6f gamma=%.6f' % (rw, rc, g))
    print('  dense: r_within=%.6f r_cross=%.6f gamma=%.6f'
          % (st['r_within'], st['r_cross'], st['gamma']))
    assert abs(rw - st['r_within']) < 1e-9
    assert abs(rc - st['r_cross']) < 1e-9
    assert abs(g - st['gamma']) < 1e-9
    print('  >>> ASSERT PASSED: block-sum matches dense R block_contrast_stats.')
    print('=' * 70)


def main():
    selfcheck()

    print('RECORDING trajectory for %s @ seeds %s' % (DATASETS, SEEDS))
    t0 = time.perf_counter()
    all_rows = []
    for ds in DATASETS:
        for sd in SEEDS:
            ts = time.perf_counter()
            rows, _sel = run_one_seed(ds, seed=sd, hook=True)
            all_rows.extend(rows)
            nrounds = len(rows)
            nlab = (rows[0]['n_labeled'], rows[-1]['n_labeled'])
            print('  %-8s seed=%d  rounds=%d  n_labeled %d->%d  (%.1fs)'
                  % (ds, sd, nrounds, nlab[0], nlab[1], time.perf_counter() - ts),
                  flush=True)

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print('WROTE %s  (%d data rows) in %.1fs'
          % (OUT_CSV, len(all_rows), time.perf_counter() - t0))


if __name__ == '__main__':
    main()
