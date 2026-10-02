"""SOTA active learning baselines: Random, Entropy, BADGE, CoreSet, BALD, QBC."""
import numpy as np
from .plcfcm import plcfcm_fit, fuzzy_entropy


class RandomSampler:
    def select(self, X, U, V, labeled_mask, budget, c, y_true=None, seed=0):
        rng = np.random.RandomState(seed)
        unlabeled = np.where(~labeled_mask)[0]
        sel = rng.choice(unlabeled, size=min(budget, len(unlabeled)),
                         replace=False).tolist()
        return sel, {'zeta_t': 0.0, 'kappa': 0.0, 'delta_mis_proxy': 0.0,
                     'sel_boundary_margin': 0.0, 'argmax_retention': 0.0}


class EntropySampler:
    def select(self, X, U, V, labeled_mask, budget, c, y_true=None, seed=0):
        unlabeled = np.where(~labeled_mask)[0]
        H = fuzzy_entropy(U[unlabeled])
        order = np.argsort(-H)[:budget]
        sel = unlabeled[order].tolist()
        return sel, {'zeta_t': float(H[order].mean()), 'kappa': 0.0,
                     'delta_mis_proxy': 0.0,
                     'sel_boundary_margin': float(
                         (1.0 - U[sel].max(axis=1)).mean()),
                     'argmax_retention': float((U[sel].max(axis=1) > 0.5).mean())}


class BADGESampler:
    """BADGE: gradient embedding k-means++ (Ash et al., ICLR 2020)."""

    def select(self, X, U, V, labeled_mask, budget, c, y_true=None, seed=0):
        rng = np.random.RandomState(seed)
        unlabeled = np.where(~labeled_mask)[0]
        nu = len(unlabeled)
        # Vectorized gradient embedding: (p_i - e_{argmax}) \otimes x_i
        Uu = U[unlabeled]
        Xu = X[unlabeled]
        preds = Uu.argmax(axis=1)
        g = np.zeros((nu, c))
        g[np.arange(nu), preds] = 1.0
        # grad: (nu, c, d) -> flatten to (nu, c*d)
        grad = (Uu - g)[:, :, None] * Xu[:, None, :]
        embeddings = grad.reshape(nu, -1)
        # k-means++ initialization with incremental distance maintenance
        n_emb = nu
        emb_sq = np.sum(embeddings ** 2, axis=1)
        centers = [int(rng.randint(n_emb))]
        k_max = min(budget, n_emb)
        # Incremental: maintain min distance to all selected centers
        first_center = embeddings[centers[0]]
        min_dists = emb_sq + np.sum(first_center ** 2) - 2.0 * (embeddings @ first_center)
        min_dists = np.maximum(min_dists, 0.0)
        for _ in range(k_max - 1):
            probs = min_dists / (min_dists.sum() + 1e-12)
            new_idx = int(rng.choice(n_emb, p=probs))
            centers.append(new_idx)
            # Update min distances with new center only
            new_center = embeddings[new_idx]
            new_dist = emb_sq + np.sum(new_center ** 2) - 2.0 * (embeddings @ new_center)
            new_dist = np.maximum(new_dist, 0.0)
            min_dists = np.minimum(min_dists, new_dist)
        sel = unlabeled[centers[:budget]].tolist()
        return sel, {'zeta_t': 0.0, 'kappa': 0.0, 'delta_mis_proxy': 0.0,
                     'sel_boundary_margin': float(
                         (1.0 - U[sel].max(axis=1)).mean()),
                     'argmax_retention': float((U[sel].max(axis=1) > 0.5).mean())}


class CoreSetSampler:
    """Core-Set: k-center greedy (Sener & Savarese, ICLR 2018)."""

    def select(self, X, U, V, labeled_mask, budget, c, y_true=None, seed=0):
        unlabeled = np.where(~labeled_mask)[0]
        Xu = X[unlabeled]
        n = len(Xu)
        if n <= budget:
            return unlabeled.tolist(), self._diag(U, unlabeled)
        # Start from point farthest from labeled centroid
        labeled_idx = np.where(labeled_mask)[0]
        if len(labeled_idx) > 0:
            center = X[labeled_idx].mean(axis=0)
        else:
            center = Xu.mean(axis=0)
        Xu_sq = np.sum(Xu ** 2, axis=1)
        dists = Xu_sq - 2.0 * (Xu @ center) + np.sum(center ** 2)
        selected = [int(np.argmax(dists))]
        # Incremental k-center: maintain min distance to all selected
        first = Xu[selected[0]]
        min_dists = Xu_sq + np.sum(first ** 2) - 2.0 * (Xu @ first)
        for _ in range(budget - 1):
            new_idx = int(np.argmax(min_dists))
            selected.append(new_idx)
            new_pt = Xu[new_idx]
            new_dist = Xu_sq + np.sum(new_pt ** 2) - 2.0 * (Xu @ new_pt)
            min_dists = np.minimum(min_dists, new_dist)
        sel = unlabeled[selected].tolist()
        return sel, self._diag(U, sel)

    def _diag(self, U, sel):
        return {'zeta_t': 0.0, 'kappa': 0.0, 'delta_mis_proxy': 0.0,
                'sel_boundary_margin': float((1.0 - U[sel].max(axis=1)).mean()),
                'argmax_retention': float((U[sel].max(axis=1) > 0.5).mean())}


class BALDSampler:
    """BALD: Bayesian active learning by disagreement (Houlsby et al., 2011).

    Approximated via MC dropout-style ensemble of FCM with perturbed centroids.
    """

    def select(self, X, U, V, labeled_mask, budget, c, y_true=None, seed=0):
        rng = np.random.RandomState(seed)
        unlabeled = np.where(~labeled_mask)[0]
        n_ens = 8
        preds = np.zeros((n_ens, len(unlabeled), c))
        for t in range(n_ens):
            V_pert = V + rng.randn(*V.shape) * 0.05
            # Recompute membership
            D = np.sum((X[unlabeled, None, :] - V_pert[None, :, :]) ** 2, axis=2)
            Dinv = (D + 1e-12) ** (-1.0)
            preds[t] = Dinv / Dinv.sum(axis=1, keepdims=True)
        # BALD = H(mean pred) - mean H(pred)
        mean_pred = preds.mean(axis=0)
        H_mean = -np.sum(mean_pred * np.log(mean_pred + 1e-12), axis=1)
        H_cond = -np.mean(np.sum(preds * np.log(preds + 1e-12), axis=2), axis=0)
        bald = H_mean - H_cond
        order = np.argsort(-bald)[:budget]
        sel = unlabeled[order].tolist()
        return sel, {'zeta_t': float(bald[order].mean()), 'kappa': 0.0,
                     'delta_mis_proxy': 0.0,
                     'sel_boundary_margin': float(
                         (1.0 - U[sel].max(axis=1)).mean()),
                     'argmax_retention': float((U[sel].max(axis=1) > 0.5).mean())}


class QBCSampler:
    """Query-by-Committee: disagreement among committee members."""

    def select(self, X, U, V, labeled_mask, budget, c, y_true=None, seed=0):
        rng = np.random.RandomState(seed)
        unlabeled = np.where(~labeled_mask)[0]
        n_comm = 5
        preds = np.zeros((n_comm, len(unlabeled), c))
        for t in range(n_comm):
            V_t = V + rng.randn(*V.shape) * 0.1
            D = np.sum((X[unlabeled, None, :] - V_t[None, :, :]) ** 2, axis=2)
            Dinv = (D + 1e-12) ** (-1.0)
            preds[t] = Dinv / Dinv.sum(axis=1, keepdims=True)
        # Disagreement = entropy of average prediction
        mean_p = preds.mean(axis=0)
        disagreement = -np.sum(mean_p * np.log(mean_p + 1e-12), axis=1)
        order = np.argsort(-disagreement)[:budget]
        sel = unlabeled[order].tolist()
        return sel, {'zeta_t': float(disagreement[order].mean()), 'kappa': 0.0,
                     'delta_mis_proxy': 0.0,
                     'sel_boundary_margin': float(
                         (1.0 - U[sel].max(axis=1)).mean()),
                     'argmax_retention': float((U[sel].max(axis=1) > 0.5).mean())}


# Module-level cache for the representative queue: keyed by (n, d, c).
# The queue depends ONLY on the standardized X (and c), and k-means uses a
# fixed random_state, so it is identical across seeds/etas for a dataset.
_REPR_QUEUE_CACHE = {}


class RepresentativePassiveSampler:
    """Deterministic, label-free, non-active, one-time fixed representative set.

    Contrast with Random (per-round stochastic active selection): here the
    ordering is computed ONCE from X only (no y_true), via unsupervised
    k-means(K=c, fixed random_state). Within each unsupervised cluster, points
    are sorted by Euclidean distance to that cluster centroid (nearest first);
    clusters are then interleaved round-robin so that the budget grows with a
    balanced quota per cluster. The sampler walks this queue with a monotonic
    pointer, skipping already-labeled points, so a larger budget is always a
    pure-prefix superset of a smaller budget (nested). It never consumes the
    active-selection RNG and ignores U/V / y_true.
    """

    def __init__(self):
        self.name = 'PLCFCMPassive'
        self._queue = None
        self._ptr = 0

    def _build_queue(self, X, c):
        from sklearn.cluster import KMeans
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        key = (n, d, c)
        if key in _REPR_QUEUE_CACHE:
            self._queue = _REPR_QUEUE_CACHE[key]
            return
        km = KMeans(n_clusters=c, random_state=0, n_init=1)
        lab = km.fit_predict(X)
        centers = km.cluster_centers_
        cluster_lists = []
        for k in range(c):
            idx = np.where(lab == k)[0]
            if len(idx) == 0:
                continue
            dist = np.sum((X[idx] - centers[k]) ** 2, axis=1)
            order = np.argsort(dist, kind='stable')  # nearest centroid first
            cluster_lists.append(idx[order])
        # Round-robin merge -> balanced quota, nested prefix growth
        queue = []
        i = 0
        while any(i < len(lst) for lst in cluster_lists):
            for lst in cluster_lists:
                if i < len(lst):
                    queue.append(int(lst[i]))
            i += 1
        queue = np.array(queue, dtype=int)
        _REPR_QUEUE_CACHE[key] = queue
        self._queue = queue

    def select(self, X, U, V, labeled_mask, budget, c, y_true=None, seed=0):
        if self._queue is None:
            self._build_queue(X, c)
        labeled = set(np.where(labeled_mask)[0].tolist())
        sel = []
        nq = len(self._queue)
        while len(sel) < budget and self._ptr < nq:
            pt = int(self._queue[self._ptr])
            self._ptr += 1
            if pt not in labeled:
                sel.append(pt)
                labeled.add(pt)
        return sel, {'zeta_t': 0.0, 'kappa': 0.0, 'delta_mis_proxy': 0.0,
                     'sel_boundary_margin': 0.0, 'argmax_retention': 0.0}
