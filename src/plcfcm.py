"""PLCFCM: Passive-Label Constrained Fuzzy C-Means solver.

Semi-supervised FCM that incorporates labeled points as hard constraints
(centroid attraction) while optimizing the standard FCM objective on
unlabeled points.  Used as the base clusterer for all active learning methods.
"""
import numpy as np
from . import config as C


def _update_centroids(X, U, m):
    um = U ** m
    V = (um.T @ X) / (um.sum(axis=0)[:, None] + 1e-12)
    return V


def _update_membership(X, V, m):
    """Vectorized membership update using ||x-v||^2 = ||x||^2 + ||v||^2 - 2*x.v.

    Avoids the per-cluster Python loop and large (n,c,d) temporary arrays.
    """
    X_sq = np.sum(X ** 2, axis=1)          # (n,)
    V_sq = np.sum(V ** 2, axis=1)          # (c,)
    XV = X @ V.T                             # (n, c)
    D = X_sq[:, None] + V_sq[None, :] - 2.0 * XV + 1e-12  # (n, c)
    exp = 2.0 / (m - 1.0)
    Dinv = D ** (-exp)
    U = Dinv / Dinv.sum(axis=1, keepdims=True)
    return U


def plcfcm_fit(X, labeled_idx=None, labels=None, c=None, m=C.FCM_M,
               max_iter=C.FCM_MAX_ITER, tol=C.FCM_TOL, seed=0):
    """Fit PLCFCM.

    labeled_idx: array of indices that are labeled
    labels: integer labels for those indices (will be mapped to cluster ids)
    Returns (U, V, n_iter)

    Internal computation uses float32 for speed; outputs are cast back to float64.
    """
    n, d = X.shape
    if c is None:
        c = int(labels.max()) + 1 if labels is not None else 2
    # Use float32 internally for BLAS speedup (output cast back to float64)
    X32 = X.astype(np.float32, copy=False)
    rng = np.random.RandomState(seed)
    # Initialize centroids via k-means++ style (vectorized distance)
    idx0 = rng.randint(n)
    V = np.zeros((c, d), dtype=np.float32)
    V[0] = X32[idx0]
    X_sq = np.sum(X32 ** 2, axis=1)  # (n,) precompute for distance trick
    for j in range(1, c):
        V_sq = np.sum(V[:j] ** 2, axis=1)  # (j,)
        XV = X32 @ V[:j].T                   # (n, j)
        dists = np.min(X_sq[:, None] + V_sq[None, :] - 2.0 * XV, axis=1)
        dists = np.maximum(dists, 0.0)
        probs = dists / dists.sum()
        V[j] = X32[rng.choice(n, p=probs)]
    U = _update_membership(X32, V, m)

    # Build label-to-cluster mapping if labels provided
    labeled_clusters = None
    if labeled_idx is not None and len(labeled_idx) > 0 and labels is not None:
        unique_labels = sorted(set(labels.tolist()))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        labeled_clusters = np.array([label_map.get(int(l), 0) for l in labels],
                                    dtype=int)

    m_float = np.float32(m)
    for it in range(max_iter):
        V_old = V.copy()
        # Centroid update with label constraints
        if labeled_clusters is not None:
            um = U ** m_float
            # Labeled points: override membership to one-hot (vectorized)
            um[labeled_idx] = 0.0
            um[labeled_idx, labeled_clusters] = 1.0
            V = (um.T @ X32) / (um.sum(axis=0)[:, None] + 1e-12)
        else:
            um = U ** m_float
            V = (um.T @ X32) / (um.sum(axis=0)[:, None] + 1e-12)
        U = _update_membership(X32, V, m)
        # Re-apply hard labels (vectorized)
        if labeled_clusters is not None:
            U[labeled_idx] = 0.0
            U[labeled_idx, labeled_clusters] = 1.0
        if np.linalg.norm(V - V_old) < tol:
            break
    return U.astype(np.float64), V.astype(np.float64), it + 1


def fuzzy_entropy(U):
    """Row-wise entropy of membership matrix (nats)."""
    eps = 1e-12
    return -np.sum(U * np.log(U + eps), axis=1)
