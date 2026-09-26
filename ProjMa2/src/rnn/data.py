"""
Synthetic data generation (RNN) for the head-direction integration task.

input: angular velocity (rad/s)
output: corresponding heading angle (cos theta, sin theta)

Angular velocity is generated as a smoothed
(Ornstein-Uhlenbeck-like) random process so that turning is temporally
correlated, similar to how an animal's turning speed doesn't change
instantly. CHECK THIS WITH RESEARCH 
"""

import numpy as np
import torch


def generate_batch(batch_size, seq_len, dt=0.02, max_speed=3.0, tau=4, seed=None): ## is 3 reasonable?, how about tau=0.5? and dt=0.02?
    """
    Parameters
    ----------
    batch_size : number of independent sequences of angular velocities/trajectories
    seq_len    : number of time steps per sequence/trajectory
    dt         : time step (s)
    max_speed  : angular velocity is clipped to [-max_speed, max_speed] (rad/s) HOW IS THIS DONE IN RESEARCH 
    tau        : correlation time of the angular velocity process (s) (controls the speed at which the head turns, has to be plausible)
    seed       : optional int for reproducibility

    Returns
    -------
    velocity : (batch_size, seq_len, 1) float32 tensor -- input to the RNN
    target   : (batch_size, seq_len, 2) float32 tensor -- (cos theta, sin theta)
    theta    : (batch_size, seq_len) float32 tensor    -- true heading angle (rad),
               kept for plotting/validation, not fed to the model
    """
    rng = np.random.default_rng(seed) 

    noise_std = max_speed * np.sqrt(2 * dt / tau) #scaled so stationary spread of v ~ max_speed regardless of tau

    velocity = np.zeros((batch_size, seq_len), dtype=np.float32) #preallocated output, filled in one time step at a time below
    v = rng.uniform(-max_speed, max_speed, size=batch_size).astype(np.float32) #initial velocity for each sequence in the batch
    for t in range(seq_len): #each time step of 1 batch, update the velocity with a deterministic decay and noise
        v = v + (-v / tau) * dt + noise_std * rng.standard_normal(batch_size).astype(np.float32) #deterministic decay + noise
        v = np.clip(v, -max_speed, max_speed) #clipped to plausible head turning range
        velocity[:, t] = v #store this step's velocity for every trajectory in the batch

    # theta0 = 0 for every sequence 
    theta = np.cumsum(velocity, axis=1) * dt #integrate velocity over time to get heading angle
    target = np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32) #(cos, sin) avoids angle wraparound discontinuity

    velocity_t = torch.from_numpy(velocity[:, :, None]) #add trailing feature dim: (batch, seq_len, 1) for RNN input
    target_t = torch.from_numpy(target)
    theta_t = torch.from_numpy(theta) #raw angle, kept aside for plotting/validation only, not fed to the model
    return velocity_t, target_t, theta_t


if __name__ == "__main__":
    v, y, theta = generate_batch(batch_size=4, seq_len=10, seed=0)
    print("velocity shape:", v.shape)
    print("target shape:", y.shape)
    print("theta shape:", theta.shape)
