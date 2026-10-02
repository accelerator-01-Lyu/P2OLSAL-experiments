"""Adversarial oracle experiment (Figure 3).

At each round, the oracle flips labels to maximize disagreement with model.
Tracks cumulative dynamic regret for P2_Full vs BADGE vs QBC.
"""
import os, csv, time
import numpy as np
from .data import load_dataset
from .registry import build_registry
from .plcfcm import plcfcm_fit
from .runner_generic import _map_clusters
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')


def adversarial_experiment(dataset, methods, n_rounds=10, seed=0):
    """Run adversarial oracle experiment. Returns list of per-round regret."""
    X, y_true, info = load_dataset(dataset)
    n, c = X.shape[0], info['c']
    rng = np.random.RandomState(seed)
    reg = build_registry()

    results = {m: [] for m in methods}

    for method_name in methods:
        factory, statef = reg[method_name]
        method = factory()
        state = statef(dataset, c, seed)

        init_count = max(c, int(0.02 * n))
        labeled_idx = rng.choice(n, size=init_count, replace=False).tolist()
        labels = y_true[labeled_idx].copy()

        cumulative_regret = 0.0

        for t in range(n_rounds):
            # Retrain
            U, V, _ = plcfcm_fit(X, np.array(labeled_idx), labels, c=c, seed=seed)

            # Select
            labeled_mask = np.zeros(n, dtype=bool)
            labeled_mask[labeled_idx] = True
            budget = max(1, int(0.02 * n))
            sel_result = method.select(X, U, V, labeled_mask, budget, c,
                                       y_true=y_true, seed=seed + t)
            if isinstance(sel_result, tuple):
                new_idx, diag = sel_result
            else:
                new_idx = sel_result

            # Adversarial label: flip to least likely class per model
            adv_labels = []
            for i in new_idx:
                least = U[i].argmin()
                adv_labels.append(least)
            adv_labels = np.array(adv_labels)

            # Compute regret: difference between oracle-best gain and actual
            # Oracle best: label that maximizes entropy reduction
            oracle_gain = 0.0
            actual_gain = 0.0
            for i in new_idx:
                H_before = -np.sum(U[i] * np.log(U[i] + 1e-12))
                # Oracle best: true label
                u_true = np.zeros(c); u_true[y_true[i]] = 1.0
                u_star_true = (U[i] + 0.5 * u_true) / 1.5
                H_after_true = -np.sum(u_star_true * np.log(u_star_true + 1e-12))
                oracle_gain += H_before - H_after_true
                # Actual (adversarial) label
                u_adv = np.zeros(c); u_adv[adv_labels[list(new_idx).index(i)]] = 1.0
                u_star_adv = (U[i] + 0.5 * u_adv) / 1.5
                H_after_adv = -np.sum(u_star_adv * np.log(u_star_adv + 1e-12))
                actual_gain += H_before - H_after_adv

            regret = max(0, oracle_gain - actual_gain)
            cumulative_regret += regret
            results[method_name].append({
                'round': t, 'dataset': dataset, 'method': method_name,
                'seed': seed, 'regret': cumulative_regret,
                'zeta': float(np.mean([np.linalg.norm(U[i] - (np.eye(c)[adv_labels[j]]) ,1)
                                       for j, i in enumerate(new_idx)])),
            })

            labeled_idx.extend(new_idx)
            labels = np.concatenate([labels, adv_labels])

    return results


def main():
    datasets = ['iris', 'ecoli', 'segment']
    methods = ['P2_Full', 'BADGE', 'QBC']
    all_rows = []
    for ds in datasets:
        for seed in range(5):
            print(f'  adversarial {ds} seed={seed}', flush=True)
            res = adversarial_experiment(ds, methods, n_rounds=10, seed=seed)
            for m, rows in res.items():
                all_rows.extend(rows)

    out = os.path.join(RES, 'adversarial_experiment.csv')
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
    print(f'saved {out} ({len(all_rows)} rows)')


if __name__ == '__main__':
    main()
