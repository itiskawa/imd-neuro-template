"""Reproducible point clouds on the unit two-sphere S² in R³."""

import numpy as np


def sample_uniform_sphere(n=3000, seed=0):
    """Sample uniformly in surface area by normalizing isotropic Gaussians."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    x = np.random.default_rng(seed).normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def sample_vmf_sphere(n=3000, mu=(0.0, 0.0, 1.0), kappa=2.0, seed=0):
    """Sample density proportional to exp(kappa * mu·x) on the unit sphere.

    mu is normalized internally. kappa=0 gives uniform surface measure;
    larger kappa concentrates samples around mu. In three dimensions the
    axial coordinate has an exact inverse-CDF sampler (no rejection loop).
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    mu = np.asarray(mu, dtype=float)
    if mu.shape != (3,) or not np.isfinite(mu).all() or np.linalg.norm(mu) == 0:
        raise ValueError("mu must be a finite nonzero vector of length 3")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    mu = mu / np.linalg.norm(mu)
    if kappa < 1e-8:
        return sample_uniform_sphere(n, seed)

    rng = np.random.default_rng(seed)
    u = np.clip(rng.random(n), np.finfo(float).tiny, 1 - np.finfo(float).eps)
    # log(u + (1-u) exp(-2*kappa)) is stable for concentrated distributions.
    w = 1 + np.logaddexp(np.log(u), np.log1p(-u) - 2 * kappa) / kappa
    w = np.clip(w, -1, 1)
    angle = rng.uniform(0, 2 * np.pi, n)
    helper = np.eye(3)[np.argmin(np.abs(mu))]
    e1 = np.cross(mu, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(mu, e1)
    tangent = np.cos(angle)[:, None] * e1 + np.sin(angle)[:, None] * e2
    return w[:, None] * mu + np.sqrt(1 - w**2)[:, None] * tangent
