"""
Unattended, heavier end-to-end run: trains a longer/more-converged head-direction
RNN, simulates a larger set of trajectories, and runs the full diagnostic suite
(loss curve, sanity check, 3D PCA, persistent homology with a stability check,
and aggregate circular coordinates), saving every plot and a text summary to
disk so results can be reviewed without needing to babysit the run.

Usage:
    python heavy_run.py
"""

import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.append(str(THIS_DIR))

from data import generate_batch
from model import HeadDirectionRNN

FIG_DIR = REPO_ROOT / "results" / "figures"
MODEL_DIR = REPO_ROOT / "results" / "models"
DATA_DIR = REPO_ROOT / "data" / "simulated"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = REPO_ROOT / "results" / "heavy_run_summary.txt"
CKPT_PATH = MODEL_DIR / "head_direction_rnn_heavy.pt"

HIDDEN_SIZE = 64
SEQ_LEN = 600
BATCH_SIZE = 128
LR = 1e-3
N_STEPS = 20000
CHECKPOINT_EVERY = 2000
LOG_EVERY = 200

N_SEQUENCES_SIM = 500
DIAG_SAMPLE_SIZE = 1500
RIPSER_MAXDIM = 1
N_LANDMARKS = 400
N_BOOTSTRAP = 8
BOOTSTRAP_SAMPLE_SIZE = 1000


def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)


def train_with_checkpoints():
    log(f"Starting training: n_steps={N_STEPS}, seq_len={SEQ_LEN}, lr={LR}, hidden_size={HIDDEN_SIZE}")
    torch.manual_seed(0)
    model = HeadDirectionRNN(hidden_size=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    losses = []
    for step in range(N_STEPS):
        velocity, target, _ = generate_batch(BATCH_SIZE, SEQ_LEN, dt=0.02)
        pred, _ = model(velocity)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            log(f"step {step:6d} | loss {loss.item():.5f}")

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(), CKPT_PATH)
            np.save(FIG_DIR / "heavy_losses.npy", np.array(losses))
            log(f"checkpoint saved at step {step} -> {CKPT_PATH}")

    torch.save(model.state_dict(), CKPT_PATH)
    np.save(FIG_DIR / "heavy_losses.npy", np.array(losses))
    log(f"training done, final checkpoint saved -> {CKPT_PATH}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(losses)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"Training loss (n_steps={N_STEPS}, lr={LR})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "heavy_loss_curve.png", dpi=150)
    plt.close(fig)
    log("saved loss curve -> results/figures/heavy_loss_curve.png")

    return model, losses


def sanity_check(model):
    log("Running sanity check on a single long trajectory...")
    velocity, target, theta_check = generate_batch(batch_size=1, seq_len=SEQ_LEN, seed=7)
    with torch.no_grad():
        pred, _ = model(velocity)

    t = range(SEQ_LEN)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(t, velocity[0, :, 0].numpy())
    axes[0].set_title("vitesse angulaire (input)")
    axes[0].set_xlabel("time step")

    axes[1].plot(t, theta_check[0].numpy(), label="theta vrai")
    theta_pred = torch.atan2(pred[0, :, 1], pred[0, :, 0]).numpy()
    axes[1].plot(t, np.unwrap(theta_pred), "--", label="theta prédit")
    axes[1].set_title("angle vrai vs prédit")
    axes[1].set_xlabel("time step")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "heavy_sanity_check.png", dpi=150)
    plt.close(fig)

    max_err = np.max(np.abs(np.angle(np.exp(1j * (np.unwrap(theta_pred) - theta_check[0].numpy())))))
    log(f"saved sanity check -> results/figures/heavy_sanity_check.png (max angular error: {max_err:.3f} rad)")
    return max_err


def simulate_trajectories(model):
    log(f"Simulating {N_SEQUENCES_SIM} trajectories of length {SEQ_LEN}...")
    velocity, target, theta = generate_batch(N_SEQUENCES_SIM, SEQ_LEN, dt=0.02, seed=123)
    with torch.no_grad():
        pred, h = model(velocity)

    save_path = DATA_DIR / "heavy_head_direction_trajectories.npz"
    np.savez(
        save_path,
        hidden_states=h.numpy(),
        velocity=velocity.numpy(),
        theta=theta.numpy(),
        pred=pred.numpy(),
    )
    log(f"saved simulated trajectories -> {save_path} (hidden_states shape: {h.shape})")
    return h.numpy(), theta.numpy()


def pca_plots(hidden_states, theta):
    from sklearn.decomposition import PCA

    log("Computing PCA and saving 3D scatter...")
    h_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    theta_flat = theta.reshape(-1)

    pca = PCA(n_components=3)
    h_pca = pca.fit_transform(h_flat)
    evr = pca.explained_variance_ratio_
    log(f"explained variance (top 3 PCs): {evr} (cumulative: {evr.sum():.3f})")

    n_plot = min(5000, len(h_flat))
    idx_plot = np.random.default_rng(0).choice(len(h_flat), size=n_plot, replace=False)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(
        h_pca[idx_plot, 0], h_pca[idx_plot, 1], h_pca[idx_plot, 2],
        c=theta_flat[idx_plot], cmap="hsv", s=3,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.colorbar(sc, label="theta (rad)")
    fig.savefig(FIG_DIR / "heavy_pca3d.png", dpi=150)
    plt.close(fig)
    log("saved 3D PCA plot -> results/figures/heavy_pca3d.png")

    return h_flat, theta_flat, evr


def persistent_homology_with_stability(h_flat, theta_flat):
    from ripser import ripser
    from persim import plot_diagrams

    log(f"Running persistent homology (maxdim={RIPSER_MAXDIM}) on {DIAG_SAMPLE_SIZE} points...")
    rng = np.random.default_rng(0)
    idx_main = rng.choice(len(h_flat), size=DIAG_SAMPLE_SIZE, replace=False)
    t0 = time.time()
    result = ripser(h_flat[idx_main], maxdim=RIPSER_MAXDIM)
    log(f"ripser done in {time.time() - t0:.1f}s")

    fig = plt.figure(figsize=(5, 5))
    plot_diagrams(result["dgms"])
    fig.savefig(FIG_DIR / "heavy_persistence_diagram.png", dpi=150)
    plt.close(fig)
    log("saved persistence diagram -> results/figures/heavy_persistence_diagram.png")

    h1 = result["dgms"][1]
    h1_pers = h1[:, 1] - h1[:, 0]
    top_h1 = np.sort(h1_pers)[::-1][:5]
    log(f"top 5 H1 persistence values: {top_h1}")

    h0 = result["dgms"][0]
    n_inf_h0 = np.sum(np.isinf(h0[:, 1]))
    log(f"H0 features with infinite persistence (connected components): {n_inf_h0}")

    if RIPSER_MAXDIM >= 2:
        h2 = result["dgms"][2]
        h2_pers = h2[:, 1] - h2[:, 0] if len(h2) else np.array([])
        top_h2 = np.sort(h2_pers)[::-1][:3] if len(h2_pers) else np.array([])
        log(f"top H2 persistence values (should be small/absent if not sphere-like): {top_h2}")
    else:
        top_h2 = np.array([])

    # stability check: does the same strong H1 feature show up across independent subsamples?
    log(f"Running stability check: {N_BOOTSTRAP} independent subsamples of size {BOOTSTRAP_SAMPLE_SIZE}...")
    boot_top_h1 = []
    for i in range(N_BOOTSTRAP):
        idx_boot = rng.choice(len(h_flat), size=BOOTSTRAP_SAMPLE_SIZE, replace=False)
        res_boot = ripser(h_flat[idx_boot], maxdim=1)
        h1_boot = res_boot["dgms"][1]
        pers_boot = h1_boot[:, 1] - h1_boot[:, 0]
        top = np.max(pers_boot) if len(pers_boot) else 0.0
        boot_top_h1.append(top)
        log(f"  bootstrap {i+1}/{N_BOOTSTRAP}: max H1 persistence = {top:.4f}")

    boot_top_h1 = np.array(boot_top_h1)
    log(f"stability check summary: mean={boot_top_h1.mean():.4f}, std={boot_top_h1.std():.4f}, "
        f"min={boot_top_h1.min():.4f}, max={boot_top_h1.max():.4f}")

    return {
        "top_h1": top_h1,
        "n_inf_h0": int(n_inf_h0),
        "top_h2": top_h2,
        "bootstrap_top_h1": boot_top_h1,
    }


def circular_coords_aggregate(h_flat, theta_flat):
    try:
        from dreimac import CircularCoords
    except ImportError:
        log("dreimac not available, skipping circular coordinates plot")
        return None

    log(f"Fitting aggregate circular coordinates (n_landmarks={N_LANDMARKS}) on {DIAG_SAMPLE_SIZE} points...")
    rng = np.random.default_rng(1)
    idx = rng.choice(len(h_flat), size=DIAG_SAMPLE_SIZE, replace=False)
    h_sample = h_flat[idx]
    theta_sample = theta_flat[idx]

    t0 = time.time()
    try:
        cc = CircularCoords(h_sample, n_landmarks=N_LANDMARKS)
        circ_coord = cc.get_coordinates()
    except Exception as e:
        log(f"dreimac circular coordinates failed ({e}); skipping this plot, rest of run unaffected")
        return None
    log(f"dreimac done in {time.time() - t0:.1f}s")

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(np.cos(circ_coord), np.sin(circ_coord), c=theta_sample, cmap="hsv", s=4)
    ax.set_title("Coordonnée circulaire agrégée (dreimac) vs angle réel")
    plt.colorbar(sc, label="theta (rad)")
    fig.savefig(FIG_DIR / "heavy_circular_coords.png", dpi=150)
    plt.close(fig)
    log("saved circular coordinates plot -> results/figures/heavy_circular_coords.png")
    return circ_coord


def main():
    t_start = time.time()
    summary_lines = ["Heavy run summary", "==================", ""]

    # training is the expensive, critical stage -- let it fail loudly if it breaks
    model, losses = train_with_checkpoints()
    summary_lines.append(f"Config: n_steps={N_STEPS}, seq_len={SEQ_LEN}, lr={LR}, hidden_size={HIDDEN_SIZE}")
    summary_lines.append(f"Final training loss: {losses[-1]:.6f}")

    # everything past this point is a diagnostic -- failures are logged and
    # skipped individually so one broken plot doesn't lose everything else
    try:
        max_err = sanity_check(model)
        summary_lines.append(f"Sanity check max angular error: {max_err:.4f} rad")
    except Exception:
        log("sanity_check failed:\n" + traceback.format_exc())
        summary_lines.append("Sanity check: FAILED, see log")

    hidden_states = theta = h_flat = theta_flat = None
    try:
        hidden_states, theta = simulate_trajectories(model)
        summary_lines.append(f"Simulated {N_SEQUENCES_SIM} trajectories of length {SEQ_LEN}")
    except Exception:
        log("simulate_trajectories failed:\n" + traceback.format_exc())
        summary_lines.append("Simulation: FAILED, see log")

    if hidden_states is not None:
        try:
            h_flat, theta_flat, evr = pca_plots(hidden_states, theta)
            summary_lines.append(f"PCA explained variance (top 3): {evr} (cumulative {evr.sum():.3f})")
        except Exception:
            log("pca_plots failed:\n" + traceback.format_exc())
            summary_lines.append("PCA: FAILED, see log")

    if h_flat is not None:
        try:
            homology_results = persistent_homology_with_stability(h_flat, theta_flat)
            summary_lines.append(f"Top 5 H1 persistence values: {homology_results['top_h1']}")
            summary_lines.append(f"H0 infinite-persistence components (should be 1): {homology_results['n_inf_h0']}")
            summary_lines.append(f"Top H2 persistence values (should be near 0): {homology_results['top_h2']}")
            summary_lines.append(
                f"Bootstrap stability ({N_BOOTSTRAP} resamples of {BOOTSTRAP_SAMPLE_SIZE} points): "
                f"max H1 per resample={homology_results['bootstrap_top_h1']}, "
                f"mean={homology_results['bootstrap_top_h1'].mean():.4f}, "
                f"std={homology_results['bootstrap_top_h1'].std():.4f}"
            )
        except Exception:
            log("persistent_homology_with_stability failed:\n" + traceback.format_exc())
            summary_lines.append("Persistent homology: FAILED, see log")

        try:
            circ_coord = circular_coords_aggregate(h_flat, theta_flat)
            if circ_coord is not None:
                summary_lines.append("Circular coordinates: saved (see heavy_circular_coords.png)")
            else:
                summary_lines.append("Circular coordinates: FAILED (see log), skipped")
        except Exception:
            log("circular_coords_aggregate failed:\n" + traceback.format_exc())
            summary_lines.append("Circular coordinates: FAILED, see log")

    elapsed = time.time() - t_start
    summary_lines.append(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    summary_lines.append("\nFigures saved in results/figures/ (whichever succeeded):")
    summary_lines.append("  heavy_loss_curve.png, heavy_sanity_check.png, heavy_pca3d.png,")
    summary_lines.append("  heavy_persistence_diagram.png, heavy_circular_coords.png")

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    log(f"DONE. Total runtime: {elapsed/60:.1f} min. Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
