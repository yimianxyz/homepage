"""The end-to-end predator net: raw_obs -> steering force (2).

Mirrors the production output contract (predator_nn.js): standardize inputs by a
stored mean/std, run an MLP, then clamp the force magnitude to PREDATOR_MAX_FORCE.
Kept deliberately tiny + introspectable so the architecture search can shrink it and
a winner can be ported to JS.
"""
import torch
import torch.nn as nn

PREDATOR_MAX_FORCE = 0.05
PREDATOR_MAX_SPEED = 2.5


def clip_mag(f, m=PREDATOR_MAX_FORCE):
    n = torch.sqrt(f[:, 0] ** 2 + f[:, 1] ** 2) + 1e-12
    s = torch.clamp(m / n, max=1.0)
    return f * s.unsqueeze(1)


class E2ENet(nn.Module):
    """head='force'    -> MLP outputs the steering force directly (clip 0.05).
       head='reynolds' -> MLP outputs a DESIRED VELOCITY (clip to max speed); the
                          force is the fixed Reynolds step clip(desired - pred_vel, 0.05).
                          pred_vel is read back from obs[:, :2] (= vel / max_speed).
                          This moves the ill-conditioned subtraction out of the net so
                          it only has to learn the well-conditioned 'where to go'."""
    def __init__(self, in_dim, hidden=(16,), act='relu', head='reynolds'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = tuple(hidden)
        self.head = head
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
        out = self.net(x)
        if self.head == 'reynolds':
            desired = clip_mag(out, PREDATOR_MAX_SPEED)
            pred_vel = obs[:, :2] * PREDATOR_MAX_SPEED
            return clip_mag(desired - pred_vel, PREDATOR_MAX_FORCE)
        return clip_mag(out)

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())
