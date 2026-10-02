"""Generic active learning runner: one (dataset, method, eta, seed) job.

Runs the full active learning loop across budget fractions, returns list of
result dicts (one per budget).
"""
import time
import numpy as np
from sklearn.metrics import (accuracy_score, normalized_mutual_info_score,
                             adjusted_rand_score)
from .data import load_dataset
from .plcfcm import plcfcm_fit
from .baselines_fuzzy import SSFCM, CEFCM, GRFCM


def _get_solver(base_solver):
    if base_solver == 'plcfcm' or base_solver == 'plcfcm_passive':
        return plcfcm_fit
    elif base_solver == 'ssfcm':
        return lambda X, li, lab, c, seed: SSFCM().fit(X, li, lab, c, seed)
    elif base_solver == 'ceffcm':
        return lambda X, li, lab, c, seed: CEFCM().fit(X, li, lab, c, seed)
    elif base_solver == 'grfcm':
        return lambda X, li, lab, c, seed: GRFCM().fit(X, li, lab, c, seed)
    return plcfcm_fit


def _map_clusters(y_pred, y_true, c):
    """Hungarian-style best mapping from predicted clusters to true labels."""
    from scipy.optimize import linear_sum_assignment
    n = len(y_true)
    cost = np.zeros((c, c))
    for i in range(c):
        for j in range(c):
            cost[i, j] = -np.sum((y_pred == i) & (y_true == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {row_ind[k]: col_ind[k] for k in range(len(row_ind))}
    return np.array([mapping.get(p, p) for p in y_pred])


def run_one(dataset, method, state, expert='clean', eta=0.0, seed=0,
            budgets=None):
    """Run one active learning experiment.

    Returns list of dicts with keys: dataset, method, ablation, study, expert,
    eta, seed, budget_frac, budget_count, ACC, NMI, ARI, zeta_t, kappa,
    delta_mis_proxy, sel_boundary_margin, argmax_retention,
    sel_time, retrain_time, total_time, error
    """
    from . import config as C
    if budgets is None:
        budgets = C.BUDGETS

    t_total_start = time.perf_counter()
    X, y_true, info = load_dataset(dataset)
    n, c = X.shape[0], info['c']
    rng = np.random.RandomState(seed)

    # Initial labeled set: 2% random (minimum to bootstrap)
    init_count = max(c, int(0.02 * n))
    labeled_idx = rng.choice(n, size=init_count, replace=False).tolist()
    labels = y_true[labeled_idx].copy()

    # Add label noise
    if eta > 0:
        n_noise = int(eta * len(labels))
        noise_idx = rng.choice(len(labels), size=n_noise, replace=False)
        for ni in noise_idx:
            wrong = [k for k in range(c) if k != labels[ni]]
            labels[ni] = rng.choice(wrong)

    base_solver = _get_solver(state['base_solver'])
    results = []

    for budget_frac in budgets:
        budget_count = max(1, int(budget_frac * n) - len(labeled_idx))
        if budget_count <= 0:
            # Already have enough labels, just evaluate
            U, V, _ = base_solver(X, np.array(labeled_idx), labels, c=c,
                                  seed=seed)
            y_pred = U.argmax(axis=1)
            y_mapped = _map_clusters(y_pred, y_true, c)
            results.append({
                'dataset': dataset, 'method': getattr(method, 'name', method.__class__.__name__),
                'ablation': '', 'study': 'main', 'expert': expert,
                'eta': eta, 'seed': seed, 'budget_frac': budget_frac,
                'budget_count': len(labeled_idx),
                'ACC': accuracy_score(y_true, y_mapped),
                'NMI': normalized_mutual_info_score(y_true, y_pred),
                'ARI': adjusted_rand_score(y_true, y_pred),
                'zeta_t': 0.0, 'kappa': 0.0, 'delta_mis_proxy': 0.0,
                'sel_boundary_margin': 0.0, 'argmax_retention': 0.0,
                'sel_time': 0.0, 'retrain_time': 0.0,
                'total_time': time.perf_counter() - t_total_start,
                'error': ''
            })
            continue

        # Retrain with current labels
        t_retrain = time.perf_counter()
        U, V, _ = base_solver(X, np.array(labeled_idx), labels, c=c,
                              seed=seed)
        retrain_time = time.perf_counter() - t_retrain

        # Select new points
        labeled_mask = np.zeros(n, dtype=bool)
        labeled_mask[labeled_idx] = True
        t_sel = time.perf_counter()
        sel_result = method.select(X, U, V, labeled_mask, budget_count, c,
                                   y_true=y_true, seed=seed)
        if isinstance(sel_result, tuple):
            new_idx, diag = sel_result
        else:
            new_idx = sel_result
            diag = {}
        sel_time = time.perf_counter() - t_sel

        # Label new points (with noise)
        new_labels = y_true[new_idx].copy()
        if eta > 0:
            n_noise = max(1, int(eta * len(new_idx)))
            noise_idx = rng.choice(len(new_idx), size=n_noise, replace=False)
            for ni in noise_idx:
                wrong = [k for k in range(c) if k != new_labels[ni]]
                new_labels[ni] = rng.choice(wrong)

        labeled_idx.extend(new_idx)
        labels = np.concatenate([labels, new_labels])

        # Evaluate
        U_final, V_final, _ = base_solver(X, np.array(labeled_idx), labels,
                                          c=c, seed=seed)
        y_pred = U_final.argmax(axis=1)
        y_mapped = _map_clusters(y_pred, y_true, c)

        results.append({
            'dataset': dataset,
            'method': getattr(method, 'name', method.__class__.__name__),
            'ablation': '',
            'study': 'main',
            'expert': expert,
            'eta': eta,
            'seed': seed,
            'budget_frac': budget_frac,
            'budget_count': len(labeled_idx),
            'ACC': accuracy_score(y_true, y_mapped),
            'NMI': normalized_mutual_info_score(y_true, y_pred),
            'ARI': adjusted_rand_score(y_true, y_pred),
            'zeta_t': diag.get('zeta_t', 0.0),
            'kappa': diag.get('kappa', 0.0),
            'delta_mis_proxy': diag.get('delta_mis_proxy', 0.0),
            'sel_boundary_margin': diag.get('sel_boundary_margin', 0.0),
            'argmax_retention': diag.get('argmax_retention', 0.0),
            'sel_time': sel_time,
            'retrain_time': retrain_time,
            'total_time': time.perf_counter() - t_total_start,
            'error': ''
        })

    return results
