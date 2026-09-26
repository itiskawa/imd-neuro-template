"""
Trains the head-direction RNN via Adam + BPTT.

Usage:
    python train.py
"""

from pathlib import Path

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
):
    if save_path is None:
        save_path = REPO_ROOT / "results" / "models" / "head_direction_rnn.pt"
    save_path = Path(save_path)

    torch.manual_seed(seed)
    model = HeadDirectionRNN(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    for step in range(n_steps):
        velocity, target, _ = generate_batch(batch_size, seq_len, dt=dt)
        velocity, target = velocity.to(device), target.to(device)

        pred, _ = model(velocity)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        if step % log_every == 0 or step == n_steps - 1:
            print(f"step {step:5d} | loss {loss.item():.5f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"saved trained model to {save_path}")

    return model, losses


if __name__ == "__main__":
    train()
