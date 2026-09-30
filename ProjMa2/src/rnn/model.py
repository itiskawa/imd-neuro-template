"""
RNN model for head-direction integration task
"""

import torch
import torch.nn as nn


class HeadDirectionRNN(nn.Module):
    """
    Recurrent network trained to integrate angular velocity into a heading
    angle, read out as (cos theta, sin theta)

    Uses a GRU rather than a vanilla tanh RNN: integrating velocity over
    many time steps requires preserving information over a long horizon,
    and a plain RNN's gradients vanish too quickly through that many BPTT
    steps for it to learn this reliably (verified experimentally -- a
    vanilla RNN on this task barely moves off the "predict nothing"
    baseline loss even after thousands of steps). The GRU's gating
    mitigates this.

    The hidden-state trajectory h_t returned by forward() is what later
    feeds the IMD pipeline (notebooks/02_imd_on_rnn.ipynb): each h_t is one
    point in hidden_size-dimensional space, and a trajectory of h_t over
    time is one path through that space.
    """

    def __init__(self, input_size=1, hidden_size=64, output_size=2, init_from_heading=False):
        super().__init__()  #required: sets up nn.Module's internal bookkeeping before attaching layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)  #recurrent core: velocity in, hidden state out each step
        self.readout = nn.Linear(hidden_size, output_size)  #linear map from hidden state to (cos, sin) prediction
        if init_from_heading:
            self.encoder = nn.Linear(2, hidden_size)  #initial heading (cos, sin) -> initial hidden state

    def forward(self, x, h0=None, theta0=None):
        """
        Parameters
        ----------
        x      : (batch, seq_len, input_size) -- angular velocity
        h0     : optional (1, batch, hidden_size) initial hidden state
        theta0 : optional (batch,) initial heading; needs init_from_heading=True.
                 Sets h0 = tanh(encoder(cos theta0, sin theta0)).

        Returns
        -------
        y : (batch, seq_len, output_size) -- predicted (cos, sin)
        h : (batch, seq_len, hidden_size) -- full hidden-state trajectory
        """
        if theta0 is not None:
            theta0 = torch.as_tensor(theta0, dtype=x.dtype)
            h0 = torch.tanh(self.encoder(torch.stack([torch.cos(theta0), torch.sin(theta0)], dim=-1)))[None]
        h, _ = self.rnn(x, h0)  #run GRU over the whole sequence; discard final hidden state (redundant with h[:, -1])
        y = self.readout(h)  #apply same linear readout at every time step (broadcasts over seq_len)
        return y, h
