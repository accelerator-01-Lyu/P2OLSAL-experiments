"""Cumulant diagnosis and block spectral gap analysis (Figures 4 & 6).

Computes third-order cumulant norm of membership vectors for each dataset,
and within-block Laplacian spectral gaps.
"""
import os, csv
import numpy as np
from .data import load_dataset
from .plcfcm import plcfcm_fit

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')


def third_cumulant_norm(U):
    """Compute Frobenius norm of the third-order cumulant tensor of memberships.

    For a matrix U (n x c), the third cumulant is a 3-tensor T_{ijk} =
    E[(u_i - mu_i)(u_j - mu_j)(u_k - mu_k)].  We compute the symmetric
    tensor and return its Frobenius norm.
    """
    n, c = U.shape
    mu = U.mean(axis=0)
    Uc = U - mu
    # Third cumulant: average of outer products u⊗u⊗u
    T = np.zeros((c, c, c))
    for i in range(n):
        u = Uc[i]
        T += np.einsum('i,j,k->ijk', u, u, u)
    T /= n
    # Frobenius norm
    return float(np.sqrt(np.sum(T ** 2)))


def block_spectral_gap(U, R):
    """Compute within-block spectral gap for each argmax-block."""
    block_ids = U.argmax(axis=1)
    blocks = sorted(set(block_ids.tolist()))
    gaps = []
    cosines = []
    for b in blocks:
        idx = np.where(block_ids == b)[0]
        if len(idx) < 3:
            continue
        R_b = R[np.ix_(idx, idx)]
        D = np.diag(R_b.sum(axis=1))
        L = D - R_b
        try:
            eigvals = np.linalg.eigvalsh(L)
            lambda2 = sorted(eigvals)[1] if len(eigvals) > 1 else 0.0
        except Exception:
            lambda2 = 0.0
        gaps.append(lambda2)
        # Average within-block cosine similarity (excluding diagonal)
        mask = ~np.eye(len(idx), dtype=bool)
        if mask.sum() > 0:
            cosines.append(float(R_b[mask].mean()))
    return gaps, cosines


def main():
    datasets = ['iris', 'wine', 'seeds', 'glass', 'ecoli', 'segment',
                'balance', 'aggregation', 'compound']
    rows = []
    for ds in datasets:
        try:
            X, y, info = load_dataset(ds)
        except Exception as e:
            print(f'  skip {ds}: {e}')
            continue
        U, V, _ = plcfcm_fit(X, c=info['c'], seed=0)
        t3 = third_cumulant_norm(U)
        # Redundancy matrix
        norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
        Un = U / norms
        R = Un @ Un.T
        np.fill_diagonal(R, 0.0)
        gaps, cosines = block_spectral_gap(U, R)
        rows.append({
            'dataset': ds, 'n': info['n'], 'd': info['d'], 'c': info['c'],
            'third_cumulant_norm': t3,
            'mean_block_spectral_gap': float(np.mean(gaps)) if gaps else 0.0,
            'mean_block_cosine': float(np.mean(cosines)) if cosines else 0.0,
            'n_blocks': len(gaps),
        })
        print(f'  {ds}: T3={t3:.4f}, gap={np.mean(gaps) if gaps else 0:.4f}, '
              f'cos={np.mean(cosines) if cosines else 0:.4f}')

    out = os.path.join(RES, 'cumulant_spectral_diagnosis.csv')
    if rows:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
