"""Permutation-invariant set network: boid SET -> steering force (2).

Two modes (Occam-graded, cheapest first):
  mode='deepsets' : phi(boid_i) -> masked mean-pool -> concat pred_vel -> rho -> 2
                    Cannot see boid-boid structure (per-boid local density is pairwise),
                    so this is the cluster centroid WITHOUT density weighting.
  mode='attn'     : phi -> ONE masked multi-head self-attention block -> masked mean-pool
                    -> concat pred_vel -> rho -> 2. Self-attention lets each boid weight by
                    how many boids sit near it = the density weighting production uses.

Reynolds output head (same contract as e2e_net.E2ENet): the net emits a DESIRED VELOCITY,
the fixed step force = clip(desired - pred_vel, PREDATOR_MAX_FORCE) is applied outside, so
the net only learns the well-conditioned 'where to go'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PREDATOR_MAX_FORCE = 0.05
PREDATOR_MAX_SPEED = 2.5


def clip_mag(v, m):
    n = torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2) + 1e-12
    return v * torch.clamp(m / n, max=1.0).unsqueeze(1)


class MaskedSelfAttn(nn.Module):
    def __init__(self, d, heads=2):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, mask):                      # x (B,N,d), mask (B,N) 1/0
        B, N, d = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]             # (B,h,N,dk)
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)        # (B,h,N,N)
        keep = mask[:, None, None, :] > 0.5
        att = att.masked_fill(~keep, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = torch.nan_to_num(att)                  # rows with no valid key -> 0
        o = (att @ v).transpose(1, 2).reshape(B, N, d)
        return self.proj(o)


class AttnPool(nn.Module):
    """Cross-attention pool: a learned query (modulated by pred_vel) attends over boids ->
    one weighted-sum vector. This IS the density-weighted centroid mechanism: the boid
    embeddings (post self-attention) carry local density, the softmax turns it into the
    weighting w_i, the weighted sum of values is the centroid production computes."""
    def __init__(self, d, heads=2):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(2, d)                     # query from pred_vel
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, mask, pvel):                # x (B,N,d)
        B, N, d = x.shape
        q = self.q(pvel).reshape(B, self.h, 1, self.dk)
        kv = self.kv(x).reshape(B, N, 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                          # (B,h,N,dk)
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)       # (B,h,1,N)
        att = att.masked_fill(mask[:, None, None, :] <= 0.5, float('-inf'))
        att = torch.nan_to_num(torch.softmax(att, dim=-1))
        o = (att @ v).reshape(B, d)
        return self.proj(o)


class SetNet(nn.Module):
    def __init__(self, in_dim=5, d=32, rho=(64,), mode='attn', heads=2, act='relu',
                 nblocks=1, pool='mean'):
        super().__init__()
        self.mode = mode
        self.pool = pool
        self.register_buffer('fmean', torch.zeros(in_dim))
        self.register_buffer('fstd', torch.ones(in_dim))
        A = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}[act]
        self.phi = nn.Sequential(nn.Linear(in_dim, d), A(), nn.Linear(d, d))
        if mode == 'attn':
            self.blocks = nn.ModuleList([MaskedSelfAttn(d, heads) for _ in range(nblocks)])
            self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(nblocks)])
        if pool == 'attn':
            self.attnpool = AttnPool(d, heads)
        layers, din = [], d + 2
        for h in rho:
            layers += [nn.Linear(din, h), A()]
            din = h
        layers += [nn.Linear(din, 2)]
        self.rho = nn.Sequential(*layers)

    def set_standardizer(self, feats, mask):
        m = mask.reshape(-1) > 0.5
        x = feats.reshape(-1, feats.shape[-1])[m]
        self.fmean.copy_(x.mean(0))
        self.fstd.copy_(x.std(0).clamp_min(1e-6))

    def forward(self, feats, mask, pvel):            # (B,N,5),(B,N),(B,2)
        x = (feats - self.fmean) / self.fstd
        x = x * mask.unsqueeze(2)
        h = self.phi(x)
        if self.mode == 'attn':
            for blk, nrm in zip(self.blocks, self.norms):
                h = nrm(h + blk(h, mask))
        h = h * mask.unsqueeze(2)
        if self.pool == 'attn':
            pooled = self.attnpool(h, mask, pvel)
        else:
            pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)   # masked mean
        out = self.rho(torch.cat([pooled, pvel], dim=1))
        desired = clip_mag(out, PREDATOR_MAX_SPEED)
        cur = pvel * PREDATOR_MAX_SPEED
        return clip_mag(desired - cur, PREDATOR_MAX_FORCE)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
