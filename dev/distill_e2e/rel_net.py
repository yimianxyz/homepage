"""Counting-capable relational nets: boid SET -> steering force (2).

WHY: prior set_net.py maxed out at patrol cos_med ~0.982. Its 'attn' mode used
SOFTMAX self-attention, which computes a NORMALISED weighted average and structurally
discards the softmax denominator (the attention mass) — i.e. it cannot COUNT neighbours.
But E3D's core is cnt_i = #boids within cluster_r, then a_i = (cnt+1)^dens_pow ·
exp(-dpred/reach). The count is an UNNORMALISED pairwise sum. So we test architectures
that can count:

  mode='edge'   : EdgeConv / continuous-kernel GNN. Per boid i, edge MLP over
                  [rel_pos, dist, boid_j feats] aggregated by SUM over alive j (sum = count).
                  Fully generic pairwise; learns its own radial kernel. nblocks layers.
  mode='radial' : E3D-structured but learnable. K soft radial gates sigmoid((R_k^2-d2)*t_k)
                  summed over j -> per-boid soft neighbour COUNTS at learnable radii R_k;
                  plus the gated neighbourhood centroid/velocity (the nbhd two-hop term).
                  A per-boid MLP score -> SHARP softmax selection -> weighted pool. This is
                  a learnable superset of computeEvolvedTarget.
  mode='attn_denom' : standard multi-head self-attention BUT the per-boid output is
                  concatenated with log(sum exp(logits)) = the attention MASS (a soft count).
                  Minimal fix that makes a vanilla transformer counting-capable.
  mode='attn'   : plain softmax self-attention (control; expected to plateau ~0.982).

Output: Reynolds head — net emits a DESIRED VELOCITY, force = clip(desired - cur_vel,
PREDATOR_MAX_FORCE) applied outside (same contract as set_net.SetNet), so the net only
learns 'where to go' (the analytic seek is fixed).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

HALF = 840.0
BOID_MAX = 6.0
PREDATOR_MAX_FORCE = 0.05
PREDATOR_MAX_SPEED = 2.5


def clip_mag(v, m):
    n = torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2) + 1e-12
    return v * torch.clamp(m / n, max=1.0).unsqueeze(1)


def mlp(sizes, act):
    A = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}[act]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(A())
    return nn.Sequential(*layers)


class EdgeBlock(nn.Module):
    """Generic counting-capable message passing: SUM aggregation over neighbours.
    msg_i = Σ_j mask_j · edgeMLP([rel_pos_ij, dist_ij, h_j]); node_i = nodeMLP([h_i, msg_i])."""
    def __init__(self, d, edge_hidden, act):
        super().__init__()
        self.edge = mlp([3 + d, edge_hidden, d], act)        # [relx,rely,dist] + h_j -> d
        self.node = mlp([2 * d, d, d], act)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, pos, mask):                          # h (B,N,d), pos (B,N,2)
        B, N, d = h.shape
        rel = pos[:, :, None, :] - pos[:, None, :, :]         # (B,N,N,2) = p_j - p_i view
        dist = torch.sqrt((rel ** 2).sum(-1, keepdim=True) + 1e-9)  # (B,N,N,1)
        hj = h[:, None, :, :].expand(B, N, N, d)
        ein = torch.cat([rel / HALF, dist / HALF, hj], dim=-1)
        e = self.edge(ein)                                    # (B,N,N,d)
        e = e * mask[:, None, :, None]                        # mask dead j
        msg = e.sum(dim=2)                                    # SUM over j = counting
        out = self.node(torch.cat([h, msg], dim=-1))
        return self.norm(h + out)


class RadialPool(nn.Module):
    """E3D-structured, learnable. Soft counts at K learnable radii + gated neighbourhood
    centroid/velocity; per-boid MLP score -> sharp softmax selection -> weighted pool of
    [pos, vel, nbhd_pos, nbhd_vel]."""
    def __init__(self, in_feat, K=4, score_hidden=32, act='relu', logt_init=0.0):
        super().__init__()
        self.K = K
        self.logR = nn.Parameter(torch.log(torch.linspace(80, 360, K)))   # learnable radii
        self.logt = nn.Parameter(torch.full((K,), float(logt_init)))      # gate sharpness (learnable)
        # per-boid score from [soft_counts(K), own feats(in_feat), nbhd_off(2), nbhd_vel(2)]
        self.score = mlp([K + in_feat + 4, score_hidden, score_hidden, 1], act)
        self.log_tau = nn.Parameter(torch.tensor(-2.0))                   # sharp selection temp

    def forward(self, feat, pos, vel, mask):                  # feat (B,N,F) own per-boid feats
        B, N, _ = pos.shape
        rel = pos[:, :, None, :] - pos[:, None, :, :]
        d2 = (rel ** 2).sum(-1)                               # (B,N,N)
        R2 = (self.logR.exp() ** 2).view(1, 1, 1, self.K)
        t = self.logt.exp().clamp(1e-3, 1e3).view(1, 1, 1, self.K)
        gates = torch.sigmoid((R2 - d2[..., None]) * (t / (R2 + 1.0)))    # (B,N,N,K) soft within-radius
        gates = gates * mask[:, None, :, None]
        counts = gates.sum(dim=2)                             # (B,N,K) soft neighbour counts
        g0 = gates[..., 0]                                    # (B,N,N) primary-radius gate
        gsum = g0.sum(dim=2, keepdim=True).clamp_min(1e-6)    # (B,N,1)
        nbpos = (g0[..., None] * pos[:, None, :, :]).sum(2) / gsum   # (B,N,2) neighbourhood centroid
        nbvel = (g0[..., None] * vel[:, None, :, :]).sum(2) / gsum
        nboff = nbpos - pos
        sin = torch.cat([counts, feat, nboff / HALF, nbvel / BOID_MAX], dim=-1)
        s = self.score(sin).squeeze(-1)                       # (B,N)
        s = s / self.log_tau.exp().clamp(1e-2, 1e2)
        s = s.masked_fill(mask <= 0.5, float('-inf'))
        w = torch.nan_to_num(torch.softmax(s, dim=-1))        # (B,N) sharp selection
        feats_pool = torch.cat([pos / HALF, vel / BOID_MAX, nbpos / HALF, nbvel / BOID_MAX], dim=-1)
        pooled = (w[..., None] * feats_pool).sum(1)           # (B,8)
        return pooled


class AttnDenom(nn.Module):
    """Multi-head self-attention whose per-boid output is augmented with the log
    attention-mass log Σ_j exp(logit_ij) — the soft neighbour count a vanilla
    transformer normalises away."""
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d + heads, d)                  # +heads = per-head log mass
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        B, N, d = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logit = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)            # (B,h,N,N)
        keep = mask[:, None, None, :] > 0.5
        logit = logit.masked_fill(~keep, float('-inf'))
        mass = torch.logsumexp(logit, dim=-1)                          # (B,h,N) soft log-count
        mass = torch.nan_to_num(mass, neginf=0.0)
        att = torch.nan_to_num(torch.softmax(logit, dim=-1))
        o = (att @ v).transpose(1, 2).reshape(B, N, d)
        o = torch.cat([o, mass.transpose(1, 2)], dim=-1)               # (B,N,d+heads)
        return self.norm(x + self.proj(o))


class PlainAttn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        B, N, d = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)
        att = att.masked_fill(mask[:, None, None, :] <= 0.5, float('-inf'))
        att = torch.nan_to_num(torch.softmax(att, dim=-1))
        o = (att @ v).transpose(1, 2).reshape(B, N, d)
        return self.norm(x + self.proj(o))


class CountEncoder(nn.Module):
    """Graphormer-style centrality/DEGREE encoding for continuous geometry: per-boid soft
    neighbour COUNTS at K learnable radii (sum aggregation = counting; PNA proves softmax/
    mean cannot do this), embedded to d and ADDED to the token residual stream. This is the
    'count as a separate structure passed to deep layers like a residual' — info the plain
    transformer structurally cannot compute. Also returns pairwise d2 for the spatial bias.

    With nbhd=True it additionally carries the SOFT-NEIGHBOURHOOD centroid offset and mean
    velocity (the E3D two-hop nbhd term) into the residual — so each token gets explicit
    'where/how-fast is my cluster' geometry, not just its scalar degree."""
    def __init__(self, d, K=6, hidden=64, act='relu', nbhd=False, logt_init=1.0):
        super().__init__()
        self.K = K
        self.nbhd = nbhd
        self.logR = nn.Parameter(torch.log(torch.linspace(60, 420, K)))
        self.logt = nn.Parameter(torch.full((K,), float(logt_init)))      # gate sharpness (ReZero-protected)
        in_feat = 2 * K + (4 if nbhd else 0)
        self.emb = mlp([in_feat, hidden, d], act)
        self.ln = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.zeros(1))   # ReZero gate: start as plain transformer, grow count use

    def forward(self, pos, mask, vel=None):
        d2 = ((pos[:, :, None, :] - pos[:, None, :, :]) ** 2).sum(-1)       # (B,N,N)
        R2 = (self.logR.exp() ** 2).view(1, 1, 1, self.K)
        t = self.logt.exp().clamp(1e-3, 1e3).view(1, 1, 1, self.K)
        gates = torch.sigmoid((R2 - d2[..., None]) * (t / (R2 + 1.0))) * mask[:, None, :, None]
        cnt = gates.sum(2)                                                  # (B,N,K) soft counts
        feats = [cnt, torch.log1p(cnt)]                                     # raw + degree-like
        if self.nbhd:
            g0 = gates[..., 0]                                              # (B,N,N) primary radius
            gsum = g0.sum(2, keepdim=True).clamp_min(1e-6)
            nbpos = (g0[..., None] * pos[:, None, :, :]).sum(2) / gsum      # (B,N,2) centroid
            nbvel = (g0[..., None] * vel[:, None, :, :]).sum(2) / gsum
            feats += [(nbpos - pos) / HALF, nbvel / BOID_MAX]
        return self.alpha * self.ln(self.emb(torch.cat(feats, -1))), d2


class SpatialBias(nn.Module):
    """Graphormer spatial encoding for continuous distance: per-head additive attention bias
    = linear(RBF(dist)). Lets the softmax attention gate by radius (focus within cluster_r)."""
    def __init__(self, heads, nrbf=16, dmax=900.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0, dmax, nrbf))
        self.logw = nn.Parameter(torch.full((nrbf,), -4.0))                # rbf inverse-width
        self.lin = nn.Linear(nrbf, heads)

    def forward(self, d2):
        d = torch.sqrt(d2 + 1e-9)[..., None]                               # (B,N,N,1)
        w = self.logw.exp().clamp(1e-6, 1e2)
        rbf = torch.exp(-((d - self.centers) ** 2) * w)                    # (B,N,N,nrbf)
        return self.lin(rbf).permute(0, 3, 1, 2)                           # (B,heads,N,N)


class TBlock(nn.Module):
    """Standard pre-LN transformer block + additive spatial attention bias."""
    def __init__(self, d, heads, ffn_mult=4, act='relu'):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.ffn = mlp([d, d * ffn_mult, d], act)

    def forward(self, x, bias, mask):
        B, N, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5) + bias
        att = att.masked_fill(mask[:, None, None, :] <= 0.5, float('-inf'))
        att = torch.nan_to_num(torch.softmax(att, dim=-1))
        o = (att @ v).transpose(1, 2).reshape(B, N, d)
        x = x + self.proj(o)
        x = x + self.ffn(self.ln2(x))
        return x


class PMA(nn.Module):
    """Set Transformer pooling-by-multihead-attention: n_seeds learned queries attend over
    tokens with a learnable SHARP temperature (the (a/amax)^sharp selection) -> n_seeds*d."""
    def __init__(self, d, heads, n_seeds=2):
        super().__init__()
        self.h, self.dk, self.n_seeds = heads, d // heads, n_seeds
        self.S = nn.Parameter(torch.randn(n_seeds, d) * 0.02)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d)
        self.log_tau = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, mask):
        B, N, d = x.shape
        S = self.q(self.S).unsqueeze(0).expand(B, -1, -1)
        q = S.reshape(B, self.n_seeds, self.h, self.dk).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)
        att = att / self.log_tau.exp().clamp(1e-2, 1e2)
        att = att.masked_fill(mask[:, None, None, :] <= 0.5, float('-inf'))
        att = torch.nan_to_num(torch.softmax(att, dim=-1))
        o = (att @ v).transpose(1, 2).reshape(B, self.n_seeds * d)
        return o


class RelNet(nn.Module):
    def __init__(self, in_dim=5, d=64, rho=(128, 64), mode='edge', heads=4,
                 nblocks=2, edge_hidden=64, K=4, act='relu', use_dens=False,
                 ffn_mult=4, n_seeds=2, nrbf=16, reinject=True, count_nbhd=False,
                 logt_init=1.0):
        super().__init__()
        self.mode = mode
        self.use_dens = use_dens          # if False, slice feats to first 5 (raw set only)
        self.in_dim = in_dim
        self.reinject = reinject
        self.count_nbhd = count_nbhd
        self.register_buffer('fmean', torch.zeros(in_dim))
        self.register_buffer('fstd', torch.ones(in_dim))
        self.phi = mlp([in_dim, d, d], act)
        if mode in ('countformer', 'cfrad'):
            self.count_enc = CountEncoder(d, K=K, hidden=edge_hidden, act=act,
                                          nbhd=count_nbhd, logt_init=logt_init)
            self.spatial = SpatialBias(heads, nrbf=nrbf)
            self.blocks = nn.ModuleList([TBlock(d, heads, ffn_mult, act) for _ in range(nblocks)])
            if mode == 'cfrad':
                # E3D-style sharp selection head over transformer-encoded tokens
                self.radhead = RadialPool(d, K=K, score_hidden=edge_hidden, act=act, logt_init=logt_init)
                pooled_dim = 8
            else:
                self.pma = PMA(d, heads, n_seeds=n_seeds)
                pooled_dim = n_seeds * d
        elif mode == 'edge':
            self.blocks = nn.ModuleList([EdgeBlock(d, edge_hidden, act) for _ in range(nblocks)])
            pooled_dim = d
        elif mode == 'radial':
            self.radial = RadialPool(in_dim, K=K, score_hidden=edge_hidden, act=act, logt_init=logt_init)
            pooled_dim = 8
        elif mode == 'attn_denom':
            self.blocks = nn.ModuleList([AttnDenom(d, heads) for _ in range(nblocks)])
            pooled_dim = d
        elif mode == 'attn':
            self.blocks = nn.ModuleList([PlainAttn(d, heads) for _ in range(nblocks)])
            pooled_dim = d
        else:
            raise ValueError(mode)
        self.rho = mlp([pooled_dim + 2, *rho, 2], act)

    def set_standardizer(self, feats, mask):
        m = mask.reshape(-1) > 0.5
        x = feats.reshape(-1, feats.shape[-1])[m][:, :self.in_dim]
        self.fmean.copy_(x.mean(0))
        self.fstd.copy_(x.std(0).clamp_min(1e-6))

    def forward(self, feats, mask, pvel):
        feats = feats[:, :, :self.in_dim]
        pos = feats[:, :, :2] * HALF                       # predator-relative world pos
        vel = feats[:, :, 2:4] * BOID_MAX + pvel[:, None, :] * PREDATOR_MAX_SPEED  # abs-ish vel
        xn = (feats - self.fmean) / self.fstd
        xn = xn * mask.unsqueeze(2)
        if self.mode in ('countformer', 'cfrad'):
            h = self.phi(xn)
            count_emb, d2 = self.count_enc(pos, mask, vel)  # degree/count (+nbhd) residual
            bias = self.spatial(d2)                        # spatial attention bias
            h = (h + count_emb) * mask.unsqueeze(2)
            for blk in self.blocks:
                h = blk(h, bias, mask)
                if self.reinject:
                    h = h + count_emb
                h = h * mask.unsqueeze(2)
            pooled = self.radhead(h, pos, vel, mask) if self.mode == 'cfrad' else self.pma(h, mask)
        elif self.mode == 'radial':
            pooled = self.radial(xn, pos, vel, mask)
        else:
            h = self.phi(xn) * mask.unsqueeze(2)
            if self.mode == 'edge':
                for blk in self.blocks:
                    h = blk(h, pos, mask) * mask.unsqueeze(2)
            else:
                for blk in self.blocks:
                    h = blk(h, mask) * mask.unsqueeze(2)
            pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)   # masked mean
        out = self.rho(torch.cat([pooled, pvel], dim=1))
        desired = clip_mag(out, PREDATOR_MAX_SPEED)
        cur = pvel * PREDATOR_MAX_SPEED
        return clip_mag(desired - cur, PREDATOR_MAX_FORCE)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
