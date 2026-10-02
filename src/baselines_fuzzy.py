"""Semi-supervised fuzzy clustering baselines: SSFCM, CEFCM, GRFCM, PLCFCMPassive.

These are alternative base clusterers used with the same active sampling
(entropy-based) to isolate the effect of the base solver.
"""
import numpy as np
from .plcfcm import plcfcm_fit, _update_membership, _update_centroids
from . import config as C


class SSFCM:
    """Self-training semi-supervised FCM: pseudo-labels high-confidence points."""

    def fit(self, X, labeled_idx, labels, c, seed=0):
        U, V, _ = plcfcm_fit(X, labeled_idx, labels, c=c, seed=seed)
        # Self-training: add pseudo-labels for high-confidence unlabeled
        unlabeled = np.ones(X.shape[0], dtype=bool)
        unlabeled[labeled_idx] = False
        conf = U.max(axis=1)
        pseudo_idx = np.where(unlabeled & (conf > 0.8))[0]
        if len(pseudo_idx) > 0:
            pseudo_labels = U[pseudo_idx].argmax(axis=1)
            all_idx = np.concatenate([labeled_idx, pseudo_idx])
            all_lab = np.concatenate([labels, pseudo_labels])
            U, V, _ = plcfcm_fit(X, all_idx, all_lab, c=c, seed=seed)
        return U, V, 0



class CEFCM:
    """Cross-entropy FCM: aligns labeled membership with one-hot targets."""

    def fit(self, X, labeled_idx, labels, c, seed=0):
        n, d = X.shape
        rng = np.random.RandomState(seed)
        V = X[rng.choice(n, c, replace=False)]
        U = _update_membership(X, V, C.FCM_M)
        lam_ce = 0.3
        for _ in range(C.FCM_MAX_ITER):
            V_old = V.copy()
            um = U ** C.FCM_M
            V = (um.T @ X) / (um.sum(axis=0)[:, None] + 1e-12)
            U = _update_membership(X, V, C.FCM_M)
            # Cross-entropy pull for labeled points
            if len(labeled_idx) > 0:
                target = np.zeros((len(labeled_idx), c))
                for i, li in enumerate(labeled_idx):
                    target[i, int(labels[i])] = 1.0
                U[labeled_idx] = (1 - lam_ce) * U[labeled_idx] + lam_ce * target
                U[labeled_idx] = U[labeled_idx] / U[labeled_idx].sum(axis=1, keepdims=True)
            if np.linalg.norm(V - V_old) < C.FCM_TOL:
                break
        return U, V, 0



class GRFCM:
    """Graph-regularized FCM: preserves local neighborhood structure."""

    def fit(self, X, labeled_idx, labels, c, seed=0):
        n, d = X.shape
        # Build kNN graph (safe fallback for degenerate datasets)
        try:
            from sklearn.neighbors import kneighbors_graph
            k = min(10, n - 1)
            W = kneighbors_graph(X, k, mode='connectivity', include_self=False).toarray()
            W = (W + W.T) / 2.0
        except Exception:
            W = np.eye(n)  # fallback: no graph regularization
        D = np.diag(W.sum(axis=1))
        L = D - W
        U, V, _ = plcfcm_fit(X, labeled_idx, labels, c=c, seed=seed)
        lam_graph = 0.1
        for _ in range(50):
            V_old = V.copy()
            um = U ** C.FCM_M
            V = (um.T @ X) / (um.sum(axis=0)[:, None] + 1e-12)
            # Graph-regularized membership update
            U_new = _update_membership(X, V, C.FCM_M)
            U = U_new - lam_graph * (L @ U_new)
            U = np.clip(U, 1e-6, 1.0)
            U = U / U.sum(axis=1, keepdims=True)
            if len(labeled_idx) > 0:
                for i, li in enumerate(labeled_idx):
                    U[li] = 0.0
                    U[li, int(labels[i])] = 1.0
            if np.linalg.norm(V - V_old) < C.FCM_TOL:
                break
        return U, V, 0



class PLCFCMPassive:
    """Passive PLCFCM: uses a fixed random labeled set (no active selection)."""

    def fit(self, X, labeled_idx, labels, c, seed=0):
        return plcfcm_fit(X, labeled_idx, labels, c=c, seed=seed)

