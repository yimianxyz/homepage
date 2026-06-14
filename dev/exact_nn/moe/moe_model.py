#!/usr/bin/env python3
"""Phase-2 MoE policy model (SPEC_PHASE2_MOE §3): ONE NN, internally routed.

  gate g(situation)  -> scalar in (0,1)   (~1 planner N>5, ~0 endgame N<=5)
  planner expert E_p : per-slot MLP over the 28-dim planner block -> emb d
  endgame expert E_e : per-slot MLP over the 20-dim endgame block -> emb d
  shared head H      : the SAME output neurons in all cases:
                       logit[k] = H( g*E_p[k] + (1-g)*E_e[k] )
  argmax_k logit[k]  -> committed slot (planner: candidate; endgame: egBoid)

Phase-2 enabler: the planner block carries the ACTUAL rollout catch+boot and the
endgame block the ACTUAL scan-t, so each expert reduces to learning argmax/argmin
of visible scores (the ~37% rollout-bound ceiling is gone). float64 head keeps the
near-tie precision that sets residual S_dec.
"""
import torch
import torch.nn as nn

# Abramowitz-Stegun 7.1.26 erf (|err|<1.5e-7), bit-identical to prod cp_erf and to
# the JS deploy (moePolicy.js) -> NO train/deploy GELU mismatch. Used everywhere.
_AS = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]


def as_erf(x):
    s = torch.sign(x)
    ax = x.abs()
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((_AS[4] * t + _AS[3]) * t) + _AS[2]) * t + _AS[1]) * t + _AS[0]) * t * torch.exp(-ax * ax)
    return s * y


class ASGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + as_erf(x * 0.7071067811865476))


PLANNER_DIM = 29
ENDGAME_DIM = 20
GATE_DIM = 4
NSLOT = 16
NEG = -1e9


def mlp(dims, act=ASGELU, last_act=False):
    L = []
    for i in range(len(dims) - 1):
        L.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            L.append(act())
    return nn.Sequential(*L)


class SlotExpert(nn.Module):
    """Per-slot MLP + a light set-context (mean/max pool over valid slots, broadcast
    back) -> per-slot embedding (B,K,d). NO output LayerNorm: per-slot normalization
    would strip the cross-slot embedding SCALE that argmax (the committed decision)
    depends on — fatal to the tight planner near-ties (median dmargin ~0.02). The
    shared head reads raw embeddings; the gate routes so it sees each regime's."""
    def __init__(self, in_dim, d=128):
        super().__init__()
        self.proj = mlp([in_dim, d, d], last_act=True)
        self.post = mlp([3 * d, d, d])

    def forward(self, x, valid):                 # x (B,K,in), valid (B,K) bool
        h = self.proj(x)                         # (B,K,d)
        m = valid.unsqueeze(-1).to(h.dtype)
        cnt = m.sum(1, keepdim=True).clamp(min=1.0)
        mean = (h * m).sum(1, keepdim=True) / cnt
        mx = h.masked_fill(~valid.unsqueeze(-1), float('-inf')).amax(1, keepdim=True)
        mx = torch.nan_to_num(mx, neginf=0.0)
        ctx = torch.cat([h, mean.expand_as(h), mx.expand_as(h)], dim=-1)
        return self.post(ctx)                    # (B,K,d)


class Gate(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.net = mlp([GATE_DIM, d, 1])

    def forward(self, gf):                        # (B,GATE_DIM) -> (B,1)
        return torch.sigmoid(self.net(gf))


class Head(nn.Module):
    """Shared aggregation head: per-slot embedding -> scalar logit. float64 final."""
    def __init__(self, d=128, f64=True):
        super().__init__()
        self.f64 = f64
        self.body = nn.Linear(d, d)
        self.act = ASGELU()
        self.out = nn.Linear(d, 1)
        if f64:
            self.body.double(); self.out.double()

    def forward(self, e):                         # (B,K,d) -> (B,K)
        if self.f64:
            e = e.double()
        return self.out(self.act(self.body(e))).squeeze(-1)


CHEAP_SCORE_COL = PLANNER_DIM - 1   # planner block: cheapScore = prod's exact committed score
SCANT_COL = 18                      # endgame block: scan_t/100 (TMAX-clipped)


class MoEPolicy(nn.Module):
    def __init__(self, d=128, f64_head=True):
        super().__init__()
        self.d = d
        self.E_p = SlotExpert(PLANNER_DIM, d)
        self.E_e = SlotExpert(ENDGAME_DIM, d)
        self.gate = Gate()
        self.H = Head(d, f64_head)
        # Decisive-signal residual skip into the shared output neuron: the committed
        # decision IS argmax of the gated decisive signal (planner cheapScore /
        # endgame -scan_t); the MLP head alone can't reconstruct it to the ~0.02
        # near-tie precision, so the output aggregates H(emb) + w*dec. H learns only
        # the small correction. ONE learnable scalar, shared across all cases.
        self.w_skip = nn.Parameter(torch.ones(1, dtype=torch.float64 if f64_head else torch.float32))

    def forward(self, planner_block, endgame_block, gate_feat, slot_valid,
                p_valid, e_valid, g_override=None):
        """planner_block (B,16,29), endgame_block (B,16,20), gate_feat (B,4),
        slot_valid (B,16) bool = legal choice mask, p_valid/e_valid (B,16) bool =
        which slots feed each expert. g_override: force gate (pretraining)."""
        ep = self.E_p(planner_block, p_valid)            # (B,16,d)
        ee = self.E_e(endgame_block, e_valid)            # (B,16,d)
        if g_override is None:
            g = self.gate(gate_feat)                      # (B,1)
        elif torch.is_tensor(g_override):
            g = g_override.reshape(-1, 1).to(planner_block.dtype)
        else:
            g = torch.full((planner_block.shape[0], 1), float(g_override), device=planner_block.device)
        g3 = g.unsqueeze(-1)                             # (B,1,1)
        e = g3 * ep + (1.0 - g3) * ee                    # (B,16,d)
        logit = self.H(e)                                # (B,16) float64
        # gated decisive signal (higher = better in BOTH regimes): planner cheapScore;
        # endgame -scan_t/100 (argmin scan_t = argmax -scan_t)
        cheap = planner_block[:, :, CHEAP_SCORE_COL].double()
        nscant = (-endgame_block[:, :, SCANT_COL]).double()
        dec = g.double() * cheap + (1.0 - g.double()) * nscant     # (B,16)
        logit = logit + self.w_skip * dec
        logit = logit.masked_fill(~slot_valid, NEG)
        return logit, g.squeeze(-1).squeeze(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
