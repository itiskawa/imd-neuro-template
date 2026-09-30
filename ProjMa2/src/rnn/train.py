"""
Trains the head-direction RNN via Adam + BPTT.

Usage:
    python train.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import generate_batch
from model import HeadDirectionRNN

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent


def train(
    n_steps=3000,
    batch_size=128,
    seq_len=100,
    hidden_size=64,
    dt=0.02,
    lr=3e-3,
    seed=0,
    device="cpu",
    save_path=None,
    log_every=100,
    random_theta0=False,
    lr_schedule="cosine",
):
    """random_theta0=True: each sequence starts at a random heading, given to the
    model through its initial hidden state -- forces a ring-shaped code.

    lr_schedule="cosine" decays the learning rate to 0 over n_steps. With a constant
    rate the loss keeps spiking, and the saved model depends on where training
    stops (test error at 12 s: ~11 deg vs ~0.6 deg with the decay). None = constant.
    Batches are drawn from `seed`, so a run is reproducible."""
    if save_path is None:
        name = "head_direction_rnn_ring.pt" if random_theta0 else "head_direction_rnn.pt"
        save_path = REPO_ROOT / "results" / "models" / name
    save_path = Path(save_path)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = HeadDirectionRNN(hidden_size=hidden_size, init_from_heading=random_theta0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps) if lr_schedule == "cosine" else None
    loss_fn = nn.MSELoss()

    losses = []
    for step in range(n_steps):
        theta0 = rng.uniform(-np.pi, np.pi, batch_size) if random_theta0 else None
        velocity, target, _ = generate_batch(batch_size, seq_len, dt=dt, theta0=theta0,
                                             seed=int(rng.integers(1e9)))
        velocity, target = velocity.to(device), target.to(device)

        pred, _ = model(velocity, theta0=None if theta0 is None else torch.as_tensor(theta0, device=device))
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        if step % log_every == 0 or step == n_steps - 1:
            print(f"step {step:5d} | loss {loss.item():.5f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"saved trained model to {save_path}")

    return model, losses


if __name__ == "__main__":
    train()
