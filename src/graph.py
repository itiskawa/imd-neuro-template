"""A radius graph, density correction, and coordinate drift/CDC estimates."""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def proximity_graph(x, k=40, radius=None):
    """Return (W, radius), with binary symmetric affinities and self-loops.

    If radius is omitted, use the median distance to the kth other point.
    Thus k calibrates one global radius, rather than specifying graph degree.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite (n, d) array with n >= 2")
    tree = cKDTree(x)
    if radius is None:
        if not isinstance(k, (int, np.integer)) or not 1 <= k < len(x):
            raise ValueError("k must be an integer between 1 and n-1")
        distances, _ = tree.query(x, k=k + 1)
        radius = float(np.median(distances[:, k]))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    pairs = tree.query_pairs(radius, output_type='ndarray')
    diagonal = np.arange(len(x))
    row = np.r_[pairs[:, 0], pairs[:, 1], diagonal]
    col = np.r_[pairs[:, 1], pairs[:, 0], diagonal]
    w = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(x), len(x)))
    return w, float(radius)


def graph_generator(w, radius, alpha=1.0, intrinsic_dim=2):
    """Return A = (m+2)/radius² * (P-I), targeting (1/2) Delta at alpha=1.

    The calibration is for a binary ball kernel in intrinsic dimension m.
    alpha=1 removes observation-density drift asymptotically; on a uniform
    cloud alpha=0 also targets Brownian motion. Row sums of A are zero.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not np.isfinite(alpha) or intrinsic_dim < 1:
        raise ValueError("alpha must be finite and intrinsic_dim positive")
    q = np.asarray(w.sum(axis=1)).ravel()
    if np.any(q <= 0):
        raise ValueError("each vertex must have positive degree")
    correction = sparse.diags(q ** (-alpha))
    corrected = correction @ w @ correction
    degree = np.asarray(corrected.sum(axis=1)).ravel()
    p = sparse.diags(1 / degree) @ corrected
    # A ball in R^m has coordinate variance radius²/(m+2).
    # A has second-order coefficient scale/[2*(m+2)], so scale=m+2
    # gives generator (1/2) Delta.
    scale = intrinsic_dim + 2
    return (scale / radius**2 * (p - sparse.eye(w.shape[0]))).tocsr()


def drift_and_cdc(x, generator):
    """Return drift b (n,d) and noise covariance Gamma (n,d,d).

    b^a = A x^a
    Gamma^{ab} = A(x^a x^b) - x^a A x^b - x^b A x^a.

    This is the full product-rule defect (twice the convention that inserts
    a 1/2 in the definition of carré du champ), so Gamma itself is the SDE
    noise covariance. A is the negative-semidefinite generator, not I-P.
    """
    x = np.asarray(x, dtype=float)
    b = generator @ x
    n, d = x.shape
    gamma = np.empty((n, d, d))
    for a in range(d):
        for c in range(a, d):
            value = generator @ (x[:, a] * x[:, c]) - x[:, a] * b[:, c] - x[:, c] * b[:, a]
            gamma[:, a, c] = gamma[:, c, a] = value
    return b, gamma
