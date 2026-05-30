"""The end-to-end predator net: raw_obs -> steering force (2).

Mirrors the production output contract (predator_nn.js): standardize inputs by a
stored mean/std, run an MLP, then clamp the force magnitude to PREDATOR_MAX_FORCE.
Kept deliberately tiny + introspectable so the architecture search can shrink it and
a winner can be ported to JS.
"""
import torch
import torch.nn as nn

PREDATOR_MAX_FORCE = 0.05


def clip_mag(f, m=PREDATOR_MAX_FORCE):
    n = torch.sqrt(f[:, 0] ** 2 + f[:, 1] ** 2) + 1e-12
    s = torch.clamp(m / n, max=1.0)
    return f * s.unsqueeze(1)


class E2ENet(nn.Module):
    def __init__(self, in_dim, hidden=(16,), act='relu'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = tuple(hidden)
        self.register_buffer('mean', torch.zeros(in_dim))
        self.register_buffer('std', torch.ones(in_dim))
        A = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}[act]
        layers, d = [], in_dim
        for h in self.hidden:
            layers += [nn.Linear(d, h), A()]
            d = h
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)

    def set_standardizer(self, obs):
        self.mean.copy_(obs.mean(0))
        self.std.copy_(obs.std(0).clamp_min(1e-6))

    def forward(self, obs):
        x = (obs - self.mean) / self.std
        return clip_mag(self.net(x))

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())
