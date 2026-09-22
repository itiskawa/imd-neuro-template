"""Kernel interpolation and an ambient Euler--Maruyama IMD simulator."""

import numpy as np
from scipy.spatial import cKDTree


def make_coefficient_lookup(x, drift, cdc, k=20):
    """Interpolate b and Gamma from k nearest observations at any ambient state."""
    x = np.asarray(x, dtype=float)
    drift, cdc = np.asarray(drift), np.asarray(cdc)
    if drift.shape != x.shape or cdc.shape != (len(x), x.shape[1], x.shape[1]):
        raise ValueError("drift and cdc shapes must match x")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(x):
        raise ValueError("k must be an integer between 1 and n")
    tree = cKDTree(x)

    def lookup(z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        distances, indices = tree.query(z, k=k)
        distances, indices = distances.reshape(len(z), k), indices.reshape(len(z), k)
        d2 = distances**2
        bandwidth = np.maximum(np.median(d2, axis=1, keepdims=True), 1e-12)
        # Subtract the closest distance to avoid underflow far from the cloud.
        weights = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / bandwidth)
        weights /= weights.sum(axis=1, keepdims=True)
        return (np.einsum('nk,nkd->nd', weights, drift[indices]),
                np.einsum('nk,nkij->nij', weights, cdc[indices]))

    return lookup


def euler_maruyama(z0, lookup, dt=0.001, n_steps=500, seed=0, eigenvalue_cutoff=0.01):
    """Simulate dZ = b(Z)dt + sqrt(Gamma(Z))dW without surface projection.

    Returns shape (n_paths, n_steps+1, ambient_dim), including initial states.
    Eigenvalues below eigenvalue_cutoff * largest_eigenvalue are discarded
    to suppress estimated normal noise. Set the cutoff to 0 to keep all
    nonnegative covariance eigenvalues. This is a numerical regularization.
    """
    if not np.isfinite(dt) or dt <= 0 or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("dt must be positive and n_steps a positive integer")
    if not 0 <= eigenvalue_cutoff < 1:
        raise ValueError("eigenvalue_cutoff must be in [0, 1)")
    z = np.atleast_2d(np.asarray(z0, dtype=float)).copy()
    if not np.isfinite(z).all():
        raise ValueError("initial states must be finite")
    rng = np.random.default_rng(seed)
    paths = np.empty((len(z), n_steps + 1, z.shape[1]))
    paths[:, 0] = z
    for step in range(n_steps):
        b, gamma = lookup(z)
        eigenvalues, vectors = np.linalg.eigh((gamma + gamma.swapaxes(-1, -2)) / 2)
        eigenvalues = np.maximum(eigenvalues, 0)
        eigenvalues[eigenvalues < eigenvalue_cutoff * eigenvalues.max(axis=1, keepdims=True)] = 0
        # An eigenvector factor gives covariance Gamma without forming sqrt(Gamma).
        noise = np.einsum('nij,nj->ni', vectors, np.sqrt(eigenvalues) * rng.normal(size=z.shape))
        z = z + dt * b + np.sqrt(dt) * noise
        paths[:, step + 1] = z
    return paths
