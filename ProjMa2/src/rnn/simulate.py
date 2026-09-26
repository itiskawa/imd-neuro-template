"""
Runs a trained RNN on many sequences and saves the hidden-state
trajectories -- this is the data cloud that feeds the manifold-estimation
pipeline in src/manifold.

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
):
    if model_path is None:
        model_path = REPO_ROOT / "results" / "models" / "head_direction_rnn.pt"
    if save_path is None:
        save_path = REPO_ROOT / "data" / "simulated" / "head_direction_trajectories.npz"
    model_path = Path(model_path)
    save_path = Path(save_path)

    model = HeadDirectionRNN(hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    velocity, target, theta = generate_batch(n_sequences, seq_len, dt=dt, seed=seed)
    with torch.no_grad():
        pred, h = model(velocity)

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
