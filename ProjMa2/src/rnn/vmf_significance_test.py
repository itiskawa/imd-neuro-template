"""
Tests whether the persistent-homology loop finding is significant against a
principled null: fit a single von Mises-Fisher distribution (a "just a blob,
no loop" model) to the hidden states projected onto a unit hypersphere, then
check whether the real data's H1 persistence is a genuine outlier relative to
many synthetic samples drawn from that null.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import vonmises_fisher
from ripser import ripser

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.append(str(THIS_DIR))

import heavy_run as hr

N_NULL_SAMPLES = 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_hypersphere(X):
    X_centered = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
    return X_centered / norms


data = np.load(REPO_ROOT / "data" / "simulated" / "heavy_head_direction_trajectories.npz")
hidden_states = data["hidden_states"]
h_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
log(f"loaded hidden states: {h_flat.shape}")

# same sample the real persistence diagram was computed on
rng = np.random.default_rng(0)
idx_main = rng.choice(len(h_flat), size=hr.DIAG_SAMPLE_SIZE, replace=False)
real_sample = h_flat[idx_main]
real_sphere = to_hypersphere(real_sample)

log(f"projected {len(real_sphere)} points onto unit hypersphere in R^{real_sphere.shape[1]}")

log("fitting von Mises-Fisher null model...")
mu, kappa = vonmises_fisher.fit(real_sphere)
log(f"fitted vMF: kappa={kappa:.3f}")

log("computing real data's persistence diagram (for comparison)...")
real_result = ripser(real_sample, maxdim=1)
real_h1 = real_result["dgms"][1]
real_pers = np.sort(real_h1[:, 1] - real_h1[:, 0])[::-1]
real_top1, real_top2 = real_pers[0], real_pers[1]
log(f"real data: top1 H1 persistence = {real_top1:.4f}, top2 = {real_top2:.4f}")

log(f"generating {N_NULL_SAMPLES} null samples from fitted vMF and running persistent homology on each...")
vmf_dist = vonmises_fisher(mu, kappa)
null_top1 = []
null_top2 = []
for i in range(N_NULL_SAMPLES):
    null_sphere = vmf_dist.rvs(size=len(real_sphere), random_state=i)
    result = ripser(null_sphere, maxdim=1)
    h1 = result["dgms"][1]
    pers = np.sort(h1[:, 1] - h1[:, 0])[::-1] if len(h1) else np.array([0.0])
    top1 = pers[0] if len(pers) >= 1 else 0.0
    top2 = pers[1] if len(pers) >= 2 else 0.0
    null_top1.append(top1)
    null_top2.append(top2)
    log(f"  null {i+1}/{N_NULL_SAMPLES}: top1={top1:.4f}, top2={top2:.4f}")

null_top1 = np.array(null_top1)
null_top2 = np.array(null_top2)

p_top1 = (null_top1 >= real_top1).mean()
p_top2 = (null_top2 >= real_top2).mean()

summary = f"""
=== von Mises-Fisher significance test ===
Fitted null: single vMF blob, kappa={kappa:.3f}, on {len(real_sphere)} points in R^{real_sphere.shape[1]}

Real data top1 H1 persistence: {real_top1:.4f}
Null (vMF) top1 H1 persistence: mean={null_top1.mean():.4f}, std={null_top1.std():.4f}, max={null_top1.max():.4f}
p-value (top1): {p_top1:.4f}  (fraction of {N_NULL_SAMPLES} null samples with top1 >= real top1)

Real data top2 H1 persistence: {real_top2:.4f}
Null (vMF) top2 H1 persistence: mean={null_top2.mean():.4f}, std={null_top2.std():.4f}, max={null_top2.max():.4f}
p-value (top2): {p_top2:.4f}  (fraction of {N_NULL_SAMPLES} null samples with top2 >= real top2)
"""
print(summary)

with open(REPO_ROOT / "results" / "vmf_significance_test.txt", "w") as f:
    f.write(summary)
log("saved -> results/vmf_significance_test.txt")
