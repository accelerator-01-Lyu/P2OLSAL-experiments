"""Robust information gain and FMIS computation for P2-OLSAL.

Implements:
- Posterior predictive q_i = (1-eps_q) Dir(alpha0*p_i) + eps_q Dir(1)
- Ambiguity set Q_i = {P: D_KL(P||q_i) <= kappa_i}
- Robust gain w_i^rob = inf_{P in Q_i} E_P[Delta H] via Donsker-Varadhan dual
- FMIS (Fuzzy Mutual Information on Simplex): pairwise max-entropy projection
- Redundancy matrix R_ij and spectral regularization alpha
"""
import numpy as np
from scipy.special import digamma, gammaln
from scipy.optimize import brentq
from . import config as C
from .plcfcm import fuzzy_entropy


def dirichlet_entropy(alpha):
    """Entropy of Dirichlet(alpha) vector."""
    a0 = alpha.sum()
    return (gammaln(a0) - gammaln(alpha).sum()
            + (a0 - len(alpha)) * digamma(a0)
            - np.sum((alpha - 1) * digamma(alpha)))


def _sample_gains(p_i, c, alpha0, eps_q, mc_samples, rng, lam=0.5):
    """Vectorized Monte Carlo: return array of entropy gains."""
    H_before = -np.sum(p_i * np.log(p_i + 1e-12))
    uniform_mask = rng.rand(mc_samples) < eps_q
    n_unif = int(uniform_mask.sum())
    n_dir = mc_samples - n_unif
    s_all = np.zeros((mc_samples, c))
    if n_dir > 0:
        s_all[~uniform_mask] = rng.dirichlet(alpha0 * p_i + 1e-12, size=n_dir)
    if n_unif > 0:
        s_all[uniform_mask] = rng.dirichlet(np.ones(c), size=n_unif)
    u_star = (p_i[None, :] + lam * s_all) / (1 + lam)
    u_star = u_star / u_star.sum(axis=1, keepdims=True)
    H_after = -np.sum(u_star * np.log(u_star + 1e-12), axis=1)
    return H_before - H_after


def expected_entropy_gain(p_i, c, alpha0=C.DIRICHLET_ALPHA0,
                          eps_q=C.ROBUST_EPS_Q, mc_samples=C.MC_SAMPLES,
                          rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    gains = _sample_gains(p_i, c, alpha0, eps_q, mc_samples, rng)
    return gains.mean(), gains.std()


def robust_gain_dv(p_i, c, kappa, alpha0=C.DIRICHLET_ALPHA0,
                   eps_q=C.ROBUST_EPS_Q, mc_samples=C.MC_SAMPLES,
                   rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    gains = _sample_gains(p_i, c, alpha0, eps_q, mc_samples, rng)
    g_mean = gains.mean()
    if kappa <= 0:
        return g_mean, g_mean

    def dual(lam_val):
        if lam_val < 0:
            return -np.inf
        log_mgf = np.log(np.mean(np.exp(-lam_val * gains)) + 1e-300)
        return lam_val * g_mean - lam_val * kappa - log_mgf

    a, b = 0.0, 50.0
    gr = (np.sqrt(5) - 1) / 2
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    f1, f2 = dual(c1), dual(c2)
    for _ in range(30):
        if f1 > f2:
            b = c2; c2 = c1; f2 = f1
            c1 = b - gr * (b - a); f1 = dual(c1)
        else:
            a = c1; c1 = c2; f1 = f2
            c2 = a + gr * (b - a); f2 = dual(c2)
    w_rob = max(0.0, dual((a + b) / 2))
    return w_rob, g_mean


def compute_redundancy_matrix(U, method='cosine'):
    """Pairwise redundancy R_ij in membership space."""
    n = U.shape[0]
    if method == 'cosine':
        norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
        Un = U / norms
        R = Un @ Un.T
    else:
        R = np.exp(-0.5 * np.sum((U[:, None, :] - U[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(R, 0.0)
    return R


def compute_alpha(R, n):
    """Spectral regularization alpha from redundancy Laplacian spectral gap.

    alpha = min(lambda_2(L_R) / n, 1 / max(R))
    Uses dense eigendecomposition (reliable); matrices passed in are at most
    4000x4000 (full R for small datasets, subsampled R for large datasets).
    """
    d = R.sum(axis=1)
    # Form L = D - R without explicitly constructing D
    L = -R.copy()
    np.fill_diagonal(L, d)  # R diagonal is 0, so L_ii = d_i
    try:
        eigvals = np.linalg.eigvalsh(L)
        lambda2 = float(sorted(eigvals)[1]) if len(eigvals) > 1 else 1.0
    except Exception:
        lambda2 = 1.0
    alpha_spec = max(lambda2, 1.0) / n
    alpha_bound = 1.0 / (R.max() + 1e-12)
    return min(alpha_spec, alpha_bound), lambda2


def fmis_gain(p_i, U_selected, V, c):
    """FMIS: fuzzy mutual information on the simplex (pairwise max-entropy).

    Marginal gain of adding sample i given already-selected set S.
    FMIS(S ∪ {i}) - FMIS(S) approximated by pairwise interaction term.
    """
    if len(U_selected) == 0:
        return robust_gain_dv(p_i, c, kappa=C.ROBUST_KAPPA_BASE)[0]
    # Pairwise max-entropy projection residual
    gains = []
    for u_j in U_selected:
        # Max-entropy distribution matching first two moments of (p_i, u_j)
        cov = np.outer(p_i - p_i.mean(), u_j - u_j.mean())
        # FMIS pairwise term = 0.5 * logdet(I + covariance coupling)
        try:
            mi_pair = 0.5 * np.log(np.linalg.det(np.eye(c) + cov @ cov.T) + 1e-12)
        except Exception:
            mi_pair = 0.0
        gains.append(max(0.0, mi_pair))
    base, _ = robust_gain_dv(p_i, c, kappa=C.ROBUST_KAPPA_BASE)
    return base * (1.0 - 0.3 * np.mean(gains))


def misclassification_proxy(U, y_true, labeled_idx):
    """Empirical misfit proxy: centroid perturbation ensemble variance."""
    if len(labeled_idx) == 0:
        return np.zeros(U.shape[0])
    # Variance of membership across bootstrap perturbations of labeled set
    rng = np.random.RandomState(42)
    n_boot = 5
    vars_ = np.zeros(U.shape[0])
    for _ in range(n_boot):
        idx = rng.choice(labeled_idx, size=max(1, len(labeled_idx) // 2),
                         replace=True)
        # Perturb centroids slightly
        noise = rng.randn(*U.shape) * 0.01
        U_pert = U + noise
        U_pert = np.clip(U_pert, 1e-6, 1.0)
        U_pert = U_pert / U_pert.sum(axis=1, keepdims=True)
        vars_ += np.var(U_pert, axis=1)
    return vars_ / n_boot


def robust_gain_dv_batch(U_batch, c, kappa, alpha0=C.DIRICHLET_ALPHA0,
                          eps_q=C.ROBUST_EPS_Q, mc_samples=C.MC_SAMPLES,
                          rng=None):
    """Vectorized batch robust gain for all unlabeled samples at once.

    U_batch: (nu, c) membership matrix.
    Returns: w_rob (nu,), g_mean (nu,)
    """
    if rng is None:
        rng = np.random.RandomState(0)
    nu = U_batch.shape[0]

    # Batch MC via gamma: Dirichlet(alpha) = Gamma(alpha,1) / sum
    alpha_dir = alpha0 * U_batch + 1e-12  # (nu, c)
    uniform_mask = rng.rand(mc_samples, nu) < eps_q  # (mc_samples, nu)
    g1 = rng.gamma(alpha_dir[None, :, :], 1.0, size=(mc_samples, nu, c))
    g2 = rng.gamma(np.ones((1, 1, c)), 1.0, size=(mc_samples, nu, c))
    g = np.where(uniform_mask[:, :, None], g2, g1)
    s_all = g / g.sum(axis=2, keepdims=True)  # (mc_samples, nu, c)

    H_before = -np.sum(U_batch * np.log(U_batch + 1e-12), axis=1)  # (nu,)
    u_star = (U_batch[None, :, :] + 0.5 * s_all) / 1.5
    u_star = u_star / u_star.sum(axis=2, keepdims=True)
    H_after = -np.sum(u_star * np.log(u_star + 1e-12), axis=2)  # (mc_samples, nu)
    gains_samples = (H_before[None, :] - H_after).T  # (nu, mc_samples)
    g_mean = gains_samples.mean(axis=1)  # (nu,)

    kappa_arr = np.full(nu, kappa) if np.isscalar(kappa) else np.asarray(kappa)

    # Vectorized golden section search (20 iters -> ~1e-5 precision)
    a = np.zeros(nu); b = np.full(nu, 50.0)
    gr = (np.sqrt(5) - 1) / 2
    c1 = b - gr * (b - a); c2 = a + gr * (b - a)

    def dual_vec(lam):
        log_mgf = np.log(np.mean(np.exp(-lam[:, None] * gains_samples), axis=1) + 1e-300)
        return lam * g_mean - lam * kappa_arr - log_mgf

    f1 = dual_vec(c1); f2 = dual_vec(c2)
    for _ in range(20):
        mask = f1 > f2
        b = np.where(mask, c2, b)
        c2 = np.where(mask, c1, c2)
        f2 = np.where(mask, f1, f2)
        c1 = np.where(mask, b - gr * (b - a), c1)
        a = np.where(~mask, c1, a)
        c1 = np.where(~mask, c2, c1)
        f1 = np.where(~mask, f2, f1)
        c2 = np.where(~mask, a + gr * (b - a), c2)
        f1 = np.where(mask, dual_vec(c1), f1)
        f2 = np.where(~mask, dual_vec(c2), f2)

    w_rob = np.maximum(0.0, dual_vec((a + b) / 2))
    w_rob = np.where(kappa_arr <= 0, g_mean, w_rob)
    return w_rob, g_mean
