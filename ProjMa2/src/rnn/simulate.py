"""
Runs a trained RNN on many sequences and saves the hidden-state
trajectories -- this is the data cloud that feeds the manifold-estimation
pipeline (notebooks/02_imd_on_rnn.ipynb).

Usage:
    python simulate.py
"""

from pathlib import Path

import numpy as np
import torch

from data import generate_batch
from model import HeadDirectionRNN

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent


def simulate(
    model_path=None,
    n_sequences=200,
    seq_len=200,
    hidden_size=64,
    dt=0.02,
    seed=123,
    save_path=None,
    random_theta0=False,
):
    """random_theta0=True: use the ring model (see train.py) with random initial headings."""
    suffix = "_ring" if random_theta0 else ""
    if model_path is None:
        model_path = REPO_ROOT / "results" / "models" / f"head_direction_rnn{suffix}.pt"
    if save_path is None:
        save_path = REPO_ROOT / "data" / "simulated" / f"head_direction_trajectories{suffix}.npz"
    model_path = Path(model_path)
    save_path = Path(save_path)

    model = HeadDirectionRNN(hidden_size=hidden_size, init_from_heading=random_theta0)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    theta0 = (np.random.default_rng(seed + 1).uniform(-np.pi, np.pi, n_sequences).astype(np.float32)
              if random_theta0 else None)
    velocity, target, theta = generate_batch(n_sequences, seq_len, dt=dt, seed=seed, theta0=theta0)
    with torch.no_grad():
        pred, h = model(velocity, theta0=None if theta0 is None else torch.from_numpy(theta0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        hidden_states=h.numpy(),  # (n_sequences, seq_len, hidden_size) -- the point cloud
        velocity=velocity.numpy(),
        theta=theta.numpy(),      # true heading angle, for coloring/validating the manifold
        pred=pred.numpy(),
    )
    print(f"saved {n_sequences} trajectories of length {seq_len} to {save_path}")
    print(f"hidden_states shape: {h.shape}")

    return h.numpy(), theta.numpy()


if __name__ == "__main__":
    simulate()
