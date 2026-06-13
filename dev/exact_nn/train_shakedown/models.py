#!/usr/bin/env python3
"""Three arch families for the EXACT-NN L1 students (SPEC section 4), at
small/medium sizes, each emitting one raw value per candidate (B,16):

  deepset     -- per-boid MLP, masked mean+max pool -> global; per-candidate
                 MLP conditioned on the global. Cheapest; the L1r/L1s default.
  transformer -- 2-4 self-attention layers over [global token + 16 candidate
                 tokens]; boids enter via a deep-set pooled summary.
  pointer     -- set encoder over boids (+predator/cfg in every token) with
                 self-attention, candidate queries cross-attend over the boid
                 memory; per-candidate scalar = pointer logit. The L1p shape.

Output interpretation is the task's business (train.py): scores for L1r/L1s,
logits for L1p.

float64 head variant (f64_head=True): float32 trunk, final Linear in float64
-- the label scale is JS float64 and L1r/L1s regression precision near argmax
margins is the whole game, so we benchmark both.
"""
import torch
import torch.nn as nn

from ds import GLOB_DIM, CAND_DIM


def mlp(dims, act=nn.GELU, last_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


def masked_pool(h, mask):
    """h (B,N,d), mask (B,N) bool -> (B,2d) mean+max over valid tokens."""
    m = mask.unsqueeze(2).to(h.dtype)
    cnt = m.sum(dim=1).clamp(min=1.0)
    mean = (h * m).sum(dim=1) / cnt
    mx = h.masked_fill(~mask.unsqueeze(2), float('-inf')).amax(dim=1)
    return torch.cat([mean, mx], dim=1)


class Head(nn.Module):
    """Per-candidate head; out=1 → (B,K) scalar (l1r/l1s/l1p); out>1 → (B,K,out)
    (l1r-split: 14 catches logits + 1 boot). Optionally float64."""
    def __init__(self, d, f64, out=1):
        super().__init__()
        self.f64 = f64
        self.out = out
        self.lin = nn.Linear(d, out)
        if f64:
            self.lin.double()

    def forward(self, h):                      # (B,K,d) -> (B,K) or (B,K,out)
        if self.f64:
            h = h.double()
        y = self.lin(h)
        return y.squeeze(-1) if self.out == 1 else y


class BoidEncoder(nn.Module):
    """Per-boid MLP + masked mean/max pool + glob -> global embedding (d)."""
    def __init__(self, d):
        super().__init__()
        self.phi = mlp([4, d, d], last_act=True)
        self.post = mlp([2 * d + GLOB_DIM, d, d])

    def forward(self, boid_tok, bmask, glob):
        h = self.phi(boid_tok)
        return self.post(torch.cat([masked_pool(h, bmask), glob], dim=1))


class DeepSet(nn.Module):
    def __init__(self, d=64, f64_head=False, head_out=1):
        super().__init__()
        self.enc = BoidEncoder(d)
        self.cand = mlp([CAND_DIM + d, d, d, d])
        self.head = Head(d, f64_head, head_out)

    def forward(self, b):
        g = self.enc(b['boid_tok'], b['bmask'], b['glob'])     # (B,d)
        K = b['cand_tok'].shape[1]
        gk = g.unsqueeze(1).expand(-1, K, -1)
        h = self.cand(torch.cat([b['cand_tok'], gk], dim=2))   # (B,K,d)
        return self.head(h)


class CandTransformer(nn.Module):
    def __init__(self, d=64, nlayers=2, nheads=4, f64_head=False, head_out=1):
        super().__init__()
        self.enc = BoidEncoder(d)
        self.cproj = mlp([CAND_DIM, d, d])
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nheads, dim_feedforward=4 * d, dropout=0.0,
            batch_first=True, norm_first=True, activation='gelu')
        self.tf = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = Head(d, f64_head, head_out)

    def forward(self, b):
        g = self.enc(b['boid_tok'], b['bmask'], b['glob']).unsqueeze(1)  # (B,1,d)
        c = self.cproj(b['cand_tok'])                                    # (B,K,d)
        h = self.tf(torch.cat([g, c], dim=1))[:, 1:, :]
        return self.head(h)


class PointerNet(nn.Module):
    """Set encoder over boids+predator -> candidate queries point over the
    boid memory (cross-attention); per-candidate scalar output."""
    def __init__(self, d=64, nlayers=1, nheads=4, f64_head=False, head_out=1):
        super().__init__()
        self.bproj = mlp([4 + GLOB_DIM, d, d])
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nheads, dim_feedforward=4 * d, dropout=0.0,
            batch_first=True, norm_first=True, activation='gelu')
        self.self_enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.cproj = mlp([CAND_DIM + GLOB_DIM, d, d])
        self.cross = nn.ModuleList(
            [nn.MultiheadAttention(d, nheads, dropout=0.0, batch_first=True)
             for _ in range(nlayers)])
        self.ln_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(nlayers)])
        self.ff = nn.ModuleList([mlp([d, 2 * d, d]) for _ in range(nlayers)])
        self.head = Head(d, f64_head, head_out)

    def forward(self, b):
        B, N, _ = b['boid_tok'].shape
        K = b['cand_tok'].shape[1]
        gN = b['glob'].unsqueeze(1)
        mem = self.bproj(torch.cat([b['boid_tok'], gN.expand(-1, N, -1)], dim=2))
        pad = ~b['bmask']                                  # True = ignore
        mem = self.self_enc(mem, src_key_padding_mask=pad)
        q = self.cproj(torch.cat([b['cand_tok'], gN.expand(-1, K, -1)], dim=2))
        for attn, ln, ff in zip(self.cross, self.ln_q, self.ff):
            a, _ = attn(ln(q), mem, mem, key_padding_mask=pad, need_weights=False)
            q = q + a
            q = q + ff(q)
        return self.head(q)


SIZES = {
    'deepset':     {'small': dict(d=64),                'medium': dict(d=256)},
    'transformer': {'small': dict(d=64, nlayers=2),     'medium': dict(d=128, nlayers=4)},
    'pointer':     {'small': dict(d=64, nlayers=1),     'medium': dict(d=128, nlayers=2)},
}


def build_model(arch, size, f64_head=False, head_out=1):
    kw = dict(SIZES[arch][size], f64_head=f64_head, head_out=head_out)
    if arch == 'deepset':
        return DeepSet(**kw)
    if arch == 'transformer':
        return CandTransformer(**kw)
    if arch == 'pointer':
        return PointerNet(**kw)
    raise ValueError(arch)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
