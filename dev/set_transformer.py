"""Set Transformer (Lee et al 2019) policy: permutation-invariant attention
over the boid set. Predator state is appended as an extra "token".

Reference:
  Lee et al, "Set Transformer: A Framework for Attention-based
  Permutation-Invariant Neural Networks", ICML 2019.

Architecture:
  Boid features (N, 5):       per-boid {dx, dy, vx, vy, alive_mask}
    + predator state (1, 5):  {0, 0, pred_vx, pred_vy, predator-marker}
  → ISAB(D, m=16) → ISAB(D, m=16) → PMA(D, k=1, num_heads=4)
  → MLP(D → 64 → 2) → tanh * MAX_FORCE

ISAB (Induced Set Attention Block): two cross-attentions through m
learned inducing points. Allows O(N·m) instead of O(N²).
PMA (Pooling by Multihead Attention): pools the set to k vectors via
attention with k learned queries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


PREDATOR_MAX_FORCE = 0.05


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        # q: (B, Lq, D); k,v: (B, Lk, D); mask: (B, Lk) bool (True=valid)
        B, Lq, D = q.shape
        Lk = k.shape[1]
        Q = self.q(q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(k).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(v).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V                                # (B, H, Lq, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out(out)


class ISAB(nn.Module):
    """Induced Set Attention Block: pools through m inducing points."""
    def __init__(self, dim, num_inducing=16, num_heads=4):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.1)
        self.mab1 = MultiHeadAttention(dim, num_heads)
        self.mab2 = MultiHeadAttention(dim, num_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff1 = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ff2 = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, X, mask=None):
        # X: (B, N, D), mask: (B, N)
        B = X.shape[0]
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X, X, mask)
        H = self.ln1(H + self.ff1(H))
        out = self.mab2(X, H, H)              # H has no mask: I is always valid
        out = self.ln2(out + self.ff2(out))
        return out


class PMA(nn.Module):
    """Pooling by Multihead Attention: pool a set to k vectors."""
    def __init__(self, dim, num_pool=1, num_heads=4):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_pool, dim) * 0.1)
        self.mab = MultiHeadAttention(dim, num_heads)
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, X, mask=None):
        # X: (B, N, D), mask: (B, N)
        B = X.shape[0]
        S = self.S.expand(B, -1, -1)
        H = self.mab(S, X, X, mask)
        H = self.ln(H + self.ff(H))
        return H                                  # (B, num_pool, D)


class SetTransformerPolicy(nn.Module):
    """Per-boid features → set transformer → predator steering."""
    def __init__(self, per_boid_dim=5, pred_state_dim=4, d_model=64,
                 num_inducing=16, num_heads=4, num_pool=1, out_hidden=64):
        super().__init__()
        self.per_boid_dim = per_boid_dim
        self.pred_state_dim = pred_state_dim
        self.boid_embed = nn.Linear(per_boid_dim, d_model)
        self.pred_embed = nn.Linear(pred_state_dim, d_model)
        self.isab1 = ISAB(d_model, num_inducing, num_heads)
        self.isab2 = ISAB(d_model, num_inducing, num_heads)
        self.pma = PMA(d_model, num_pool, num_heads)
        self.head = nn.Sequential(
            nn.Linear(d_model * num_pool + d_model, out_hidden),
            nn.GELU(),
            nn.Linear(out_hidden, 2),
        )

    def forward(self, boid_feats, boid_mask, pred_state):
        """
        boid_feats: (B, N, per_boid_dim)
        boid_mask:  (B, N) bool — True = alive
        pred_state: (B, pred_state_dim) — predator's own state
        Returns: (B, 2) steering, clipped to MAX_FORCE
        """
        B = boid_feats.shape[0]
        x = self.boid_embed(boid_feats)            # (B, N, D)
        # Mask dead boids by replacing their embeddings with zeros (ISAB will
        # still see them via mask=False to nullify attention).
        x = x * boid_mask.unsqueeze(-1).float()
        x = self.isab1(x, boid_mask)
        x = self.isab2(x, boid_mask)
        pooled = self.pma(x, boid_mask).view(B, -1)     # (B, num_pool*D)
        p = self.pred_embed(pred_state)            # (B, D)
        h = torch.cat([pooled, p], dim=-1)
        out = self.head(h)
        return torch.tanh(out) * PREDATOR_MAX_FORCE


def num_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    # Smoke test
    B, N = 4, 120
    model = SetTransformerPolicy()
    print(f"SetTransformerPolicy: {num_params(model)} params")
    boid_feats = torch.randn(B, N, 5)
    boid_mask = torch.ones(B, N, dtype=torch.bool)
    boid_mask[0, 100:] = False                     # some boids dead
    pred_state = torch.randn(B, 4)
    out = model(boid_feats, boid_mask, pred_state)
    print(f"Output shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]")
