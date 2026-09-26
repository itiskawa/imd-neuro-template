"""
Bins hidden states by true (wrapped) heading angle and averages within each
bin, across all time steps and trajectories -- removes off-manifold variance
(recent velocity, transients, elapsed time since last visiting that heading)
that has nothing to do with heading itself. Standard technique from the
ring-attractor literature (e.g. Chaudhuri et al. 2019).
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.append(str(THIS_DIR))

FIG_DIR = REPO_ROOT / "results" / "figures"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


data = np.load(REPO_ROOT / "data" / "simulated" / "heavy_head_direction_trajectories.npz")
hidden_states = data["hidden_states"]
theta = data["theta"]
h_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
theta_flat = theta.reshape(-1)
log(f"loaded {len(h_flat)} (trajectory, timestep) points")

# wrap to true physical heading in [-pi, pi) -- this is the key step: average
# across DIFFERENT winding counts that share the same physical direction
theta_wrapped = np.mod(theta_flat + np.pi, 2 * np.pi) - np.pi

N_BINS = 72  # 5-degree bins
bin_edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
bin_idx = np.clip(np.digitize(theta_wrapped, bin_edges) - 1, 0, N_BINS - 1)

bin_avg_h = []
bin_centers = []
bin_counts = []
for b in range(N_BINS):
    mask = bin_idx == b
    count = mask.sum()
    if count == 0:
        continue
    bin_avg_h.append(h_flat[mask].mean(axis=0))
    bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
    bin_counts.append(count)

bin_avg_h = np.array(bin_avg_h)
bin_centers = np.array(bin_centers)
bin_counts = np.array(bin_counts)
log(f"averaged into {len(bin_avg_h)} bins, {bin_counts.min()}-{bin_counts.max()} points per bin "
    f"(mean {bin_counts.mean():.0f})")

pca = PCA(n_components=3)
h_pca_binned = pca.fit_transform(bin_avg_h)
evr = pca.explained_variance_ratio_
log(f"explained variance on BINNED data (top 3 PCs): {evr} (cumulative {evr.sum():.3f})")

# also PCA the raw (unbinned) points for a side-by-side comparison
pca_raw = PCA(n_components=2)
n_raw_plot = 5000
idx_raw = np.random.default_rng(0).choice(len(h_flat), size=n_raw_plot, replace=False)
h_pca_raw = pca_raw.fit_transform(h_flat[idx_raw])
theta_raw_wrapped = theta_wrapped[idx_raw]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

sc0 = axes[0].scatter(h_pca_raw[:, 0], h_pca_raw[:, 1], c=theta_raw_wrapped, cmap="hsv", s=4)
axes[0].set_title(f"Points bruts (n={n_raw_plot})")
axes[0].set_xlabel("PC1 (raw)")
axes[0].set_ylabel("PC2 (raw)")

sc1 = axes[1].scatter(h_pca_binned[:, 0], h_pca_binned[:, 1], c=bin_centers, cmap="hsv", s=30)
axes[1].plot(h_pca_binned[:, 0], h_pca_binned[:, 1], '-', color='gray', alpha=0.3, linewidth=1, zorder=0)
axes[1].set_title(f"Moyenné par bin d'angle vrai ({len(bin_avg_h)} bins)")
axes[1].set_xlabel("PC1 (binned)")
axes[1].set_ylabel("PC2 (binned)")

plt.colorbar(sc1, ax=axes[1], label="theta (rad, wrapped)")
fig.tight_layout()
fig.savefig(FIG_DIR / "binned_vs_raw_pca.png", dpi=150)
plt.close(fig)
log("saved comparison plot -> results/figures/binned_vs_raw_pca.png")

# 3D version of the binned result, since 2D might still not capture it fully
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")
sc = ax.scatter(h_pca_binned[:, 0], h_pca_binned[:, 1], h_pca_binned[:, 2], c=bin_centers, cmap="hsv", s=30)
ax.plot(h_pca_binned[:, 0], h_pca_binned[:, 1], h_pca_binned[:, 2], '-', color='gray', alpha=0.4, linewidth=1)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.colorbar(sc, label="theta (rad, wrapped)")
ax.set_title("Ring binned (3D PCA)")
fig.savefig(FIG_DIR / "binned_pca_3d.png", dpi=150)
plt.close(fig)
log("saved 3D binned plot -> results/figures/binned_pca_3d.png")

log("DONE")
