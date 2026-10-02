"""P2-OLSAL: Posterior-Predictive Online Local Submodular Active Learning.

Full method with robust gain, redundancy penalty, exploration reward,
and block-adaptive spectral regularization.

Alpha schemes (redundancy magnitude / block weights):
  scheme='A'  (current P2_Full): magnitude alpha = min(lambda2/n, 1/||R||inf),
              block weights (within, cross) = (2 alpha, 0.25 alpha). Unchanged.
  scheme='B'  (block-relative only): same magnitude as A; cross weight 0 and
              within weight w_B = 2 + 0.25(1-p)/p (pair-mass conserving),
              p = fraction of unordered pairs that are within-block.
  scheme='C'  (theory-adaptive magnitude): alpha_C = c0 * gamma_f * max(lambda2,1)/m
              with block contrast gamma recomputed each round from the CURRENT
              membership U; gamma_f = max(gamma, gamma_floor). Same B weights.
              gamma = clip[(r_within - r_cross)/(1 - r_cross), 0, 1],
              r0 = r_cross (scale-consistent plug-in for the Welch floor).
"""
import numpy as np
from . import config as C
from .plcfcm import plcfcm_fit, fuzzy_entropy
from .robust_selection import (robust_gain_dv, robust_gain_dv_batch,
                               compute_redundancy_matrix,
                               compute_alpha, misclassification_proxy)


def _participation_ratio_from_membership(Um):
    """d_eff = PR(R) using ONLY positive eigenvalues of R = Un Un^T.

    Nonzero eigenvalues of Un Un^T equal eigenvalues of the c x c Gram Un^T Un,
    so this is exact while avoiding an m x m eigendecomposition.
    """
    norms = np.linalg.norm(Um, axis=1, keepdims=True) + 1e-12
    Un = Um / norms
    G = Un.T @ Un                      # (c, c)
    ev = np.linalg.eigvalsh(G)
    evp = ev[ev > 1e-10]
    if len(evp) == 0:
        return 1.0, Un
    d_eff = float((evp.sum() ** 2) / np.sum(evp ** 2))
    return d_eff, Un


def block_contrast_stats(Um, R, c):
    """Structural redundancy stats for the m points whose memberships are Um.

    Returns dict with lambda2(L_R), d_eff=PR(R), r_within, r_cross, gamma,
    Welch floor r_W=(m-d_eff)/(d_eff(m-1)), within-pair fraction p.
    """
    m = R.shape[0]
    blocks = Um.argmax(axis=1)
    iu = np.triu_indices(m, k=1)
    same = blocks[:, None] == blocks[None, :]
    same_iu = same[iu]
    rv = R[iu]
    r_within = float(rv[same_iu].mean()) if np.any(same_iu) else 0.0
    r_cross = float(rv[~same_iu].mean()) if np.any(~same_iu) else 0.0
    denom = 1.0 - r_cross
    gamma = float(np.clip((r_within - r_cross) / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0

    d_eff, _ = _participation_ratio_from_membership(Um)
    r_W = float((m - d_eff) / (d_eff * (m - 1))) if (d_eff > 0 and m > 1) else 1.0

    d = R.sum(axis=1)
    L = -R.copy()
    np.fill_diagonal(L, d)
    evL = np.linalg.eigvalsh(L)
    lam2 = float(evL[1]) if len(evL) > 1 else 1.0

    counts = np.bincount(blocks, minlength=c).astype(float)
    w_pairs = float(np.sum(counts * (counts - 1)) / 2.0)
    t_pairs = m * (m - 1) / 2.0
    p = float(w_pairs / t_pairs) if t_pairs > 0 else 1.0
    return {'m': m, 'blocks': blocks, 'r_within': r_within, 'r_cross': r_cross,
            'gamma': gamma, 'd_eff': d_eff, 'lambda2': lam2, 'r_W': r_W, 'p': p}


def partition_diagnostics(U):
    """Partition coefficient PC and normalized mean membership entropy."""
    c = U.shape[1]
    pc = float(np.mean(np.sum(U ** 2, axis=1)))
    H = -np.sum(U * np.log(U + 1e-12), axis=1) / np.log(c)
    return pc, float(np.mean(H))


class P2OLSAL:
    """Full P2-OLSAL active sampler."""

    def __init__(self, robust=True, exploration=True, redundancy=True,
                 fmis=True, block=True, kappa=C.ROBUST_KAPPA_BASE,
                 rho_expl=C.EXPL_RHO, lam=C.P2_LAM,
                 alpha_cross_frac=C.P2_ALPHA_CROSS_FRAC,
                 alpha_within_mult=2.0, name='P2_Full',
                 scheme='A', c0=1.0, gamma_floor=0.01, collect_stats=False):
        self.robust = robust
        self.exploration = exploration
        self.redundancy = redundancy
        self.fmis = fmis
        self.block = block
        self.kappa = kappa
        self.rho_expl = rho_expl
        self.lam = lam
        self.alpha_cross_frac = alpha_cross_frac
        self.alpha_within_mult = alpha_within_mult
        self.name = name
        self.scheme = scheme
        self.c0 = float(c0)
        self.gamma_floor = float(gamma_floor)
        self.collect_stats = collect_stats or (scheme in ('B', 'C', 'D', 'E'))
        self.alpha_trace = []
        self.rng = np.random.RandomState(0)

    def _resolve_weights(self, alpha_A, st):
        """Return (alpha_within, alpha_cross, trace_record) per scheme."""
        rec = {'scheme': self.scheme, 'lambda2': st['lambda2'], 'd_eff': st['d_eff'],
               'r_within': st['r_within'], 'r_cross': st['r_cross'],
               'gamma_raw': st['gamma'], 'r_W': st['r_W'], 'p': st['p'],
               'm': st['m']}
        if self.scheme == 'C':
            g_use = max(st['gamma'], self.gamma_floor)
            alpha_mag = self.c0 * g_use * max(st['lambda2'], 1.0) / st['m']
            p = max(st['p'], 1e-12)
            w_B = 2.0 + 0.25 * (1.0 - p) / p
            aw, ac = w_B * alpha_mag, 0.0
            rec.update(gamma_used=g_use, w_B=w_B, alpha=alpha_mag)
        elif self.scheme == 'B':
            alpha_mag = alpha_A
            p = max(st['p'], 1e-12)
            w_B = 2.0 + 0.25 * (1.0 - p) / p
            aw, ac = w_B * alpha_mag, 0.0
            rec.update(gamma_used=st['gamma'], w_B=w_B, alpha=alpha_mag)
        elif self.scheme == 'D':
            # 2x2 cell: cross removed, within kept at baseline mult 2
            alpha_mag = alpha_A
            aw, ac = 2.0 * alpha_mag, 0.0
            rec.update(gamma_used=st['gamma'], w_B=2.0, alpha=alpha_mag)
        elif self.scheme == 'E':
            # 2x2 cell: cross kept at 0.25, within mass-conservation raised
            alpha_mag = alpha_A
            p = max(st['p'], 1e-12)
            w_B = 2.0 + 0.25 * (1.0 - p) / p
            aw, ac = w_B * alpha_mag, 0.25 * alpha_mag
            rec.update(gamma_used=st['gamma'], w_B=w_B, alpha=alpha_mag)
        else:  # A
            aw, ac = self.alpha_within_mult * alpha_A, self.alpha_cross_frac * alpha_A
            rec.update(gamma_used=st['gamma'], w_B=self.alpha_within_mult,
                       alpha=alpha_A)
        rec['alpha_within'] = aw
        rec['alpha_cross'] = ac
        return aw, ac, rec

    def select(self, X, U, V, labeled_mask, budget, c, y_true=None,
               seed=0):
        """Greedy submodular selection with incremental redundancy update."""
        self.rng = np.random.RandomState(seed)
        n = X.shape[0]
        unlabeled = np.where(~labeled_mask)[0]
        nu = len(unlabeled)
        if nu <= budget:
            return unlabeled.tolist(), self._diag(U, unlabeled, np.array([]),
                                                  labeled_mask, y_true)

        # Precompute gains for all unlabeled (BATCH vectorized)
        if self.robust:
            gains, point_gains = robust_gain_dv_batch(
                U[unlabeled], c, kappa=self.kappa, rng=self.rng)
        else:
            Uu = U[unlabeled]
            point_gains = -np.sum(Uu * np.log(Uu + 1e-12), axis=1)
            gains = point_gains.copy()

        if self.exploration:
            u_mean = U.mean(axis=0)
            expl = np.linalg.norm(U[unlabeled] - u_mean, axis=1)
            expl = expl / (expl.max() + 1e-12)
            gains = gains + self.rho_expl * expl

        # Redundancy matrix (nu x nu) and alpha
        if self.redundancy:
            if nu <= C.MAX_REDUNDANCY_N:
                R = compute_redundancy_matrix(U[unlabeled])
                alpha, _ = compute_alpha(R, nu)
                R_full = R
                if self.collect_stats:
                    st = block_contrast_stats(U[unlabeled], R, c)
                else:
                    st = None
            else:
                # Large dataset: subsample for alpha, compute rows on-the-fly
                sub_idx = self.rng.choice(nu, size=min(C.REDUNDANCY_SUBSAMPLE_N, nu),
                                           replace=False)
                R_sub = compute_redundancy_matrix(U[unlabeled[sub_idx]])
                alpha, _ = compute_alpha(R_sub, len(sub_idx))
                R_full = None  # signal on-the-fly mode
                # Precompute normalized U for fast cosine rows
                _Uu = U[unlabeled]
                _norms = np.linalg.norm(_Uu, axis=1, keepdims=True) + 1e-12
                _Un = _Uu / _norms
                if self.collect_stats:
                    st = block_contrast_stats(U[unlabeled[sub_idx]], R_sub, c)
                else:
                    st = None
        else:
            R = None
            R_full = None
            alpha = 0.0
            st = None

        # Resolve block weights per scheme (scheme A keeps the exact original path)
        trace_rec = None
        if self.redundancy and st is not None:
            alpha_within, alpha_cross, trace_rec = self._resolve_weights(alpha, st)
            if trace_rec is not None:
                pc, h_norm = partition_diagnostics(U)
                trace_rec['PC'] = pc
                trace_rec['H_norm'] = h_norm
                trace_rec['nu'] = nu
                self.alpha_trace.append(trace_rec)
        elif self.redundancy:
            # scheme A without stats collection: identical to original
            alpha_within = alpha * self.alpha_within_mult
            alpha_cross = alpha * self.alpha_cross_frac
        else:
            alpha_within = alpha_cross = 0.0

        if self.block and (R_full is not None or self.redundancy):
            block_ids = U[unlabeled].argmax(axis=1)
            if R_full is not None:
                A = np.where(block_ids[:, None] == block_ids[None, :],
                             alpha_within, alpha_cross)
                np.fill_diagonal(A, 0.0)
            else:
                A = None  # compute per-row on-the-fly
        else:
            if R_full is not None:
                A = np.full((nu, nu), alpha)
                np.fill_diagonal(A, 0.0)
            else:
                A = None
                alpha_within = alpha
                alpha_cross = alpha

        # Greedy with incremental redundancy: red_sum[k] = sum_j selected A[k,j]*R[k,j]
        selected = []
        red_sum = np.zeros(nu)
        available = np.ones(nu, dtype=bool)
        for _ in range(budget):
            if len(selected) > 0:
                scores = gains - self.lam * red_sum / len(selected)
            else:
                scores = gains.copy()
            scores[~available] = -np.inf
            best_k = int(np.argmax(scores))
            if scores[best_k] == -np.inf:
                break
            selected.append(best_k)
            available[best_k] = False
            if R_full is not None:
                red_sum += A[best_k, :] * R_full[best_k, :]
            elif self.redundancy:
                # On-the-fly: cosine row + block-aware alpha
                r_row = _Un @ _Un[best_k]
                r_row[best_k] = 0.0
                if A is None and self.block:
                    a_row = np.where(block_ids == block_ids[best_k],
                                     alpha_within, alpha_cross)
                    a_row[best_k] = 0.0
                elif A is None:
                    a_row = np.full(nu, alpha)
                    a_row[best_k] = 0.0
                else:
                    a_row = A[best_k, :]
                red_sum += a_row * r_row

        sel_idx = unlabeled[selected]
        info = self._diag(U, sel_idx, point_gains[selected], labeled_mask, y_true)
        return sel_idx.tolist(), info

    def _diag(self, U, sel_idx, point_gains, labeled_mask, y_true):
        if len(sel_idx) == 0:
            return {'zeta_t': 0.0, 'kappa': float(self.kappa),
                    'delta_mis_proxy': 0.0, 'sel_boundary_margin': 0.0,
                    'argmax_retention': 0.0}
        return {
            'zeta_t': float(np.mean(point_gains)) if len(point_gains) > 0 else 0.0,
            'kappa': float(self.kappa),
            'delta_mis_proxy': float(np.mean(misclassification_proxy(
                U, y_true, np.where(labeled_mask)[0]))),
            'sel_boundary_margin': float(np.mean(1.0 - U[sel_idx].max(axis=1))),
            'argmax_retention': float(np.mean(U[sel_idx].max(axis=1) > 0.5)),
        }
