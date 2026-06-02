"""distill_v4 — full-set input + value-regression distillation of the best planner.

WHY v3 reverted to baseline (root cause, from first principles):
  The planner's action is a deterministic function of the FULL state,
  pi(s) = argmax_k gain(s, c_k(s)), where gain is true-dynamics rollout catches.
  There is NO theoretical boundary to approximating this with a net (it is a
  deterministic finite-computation map; universal approximation applies, and the
  only points where argmax is ambiguous are gain-TIES, which by definition cost
  nothing). v3's failure had two fixable causes:

  1. INPUT ALIASING. v3 fed only the M=16 nearest boids. gain(s,c) depends on the
     full 120-boid interacting future (cohesion drags distant boids in; predator
     avoidance scatters them). Two states with identical 16-nearest views but
     different far boids get different labels -> the target is not a function of
     the obs -> irreducible error (the observed 44.7% decisive TRAIN accuracy).
     FIX: full 120-boid permutation-invariant set encoder (deepsets / set-
     transformer / per-candidate cross-attention).

  2. LOSS COLLAPSE. v3 used hard cross-entropy on argmax(gain). With ~86% of
     frames all-tied, argmax defaults to candidate 0 (E3D) -> 92% of labels are
     "pick E3D" -> CE is minimised by the constant "always E3D" = baseline. CE
     also discards value magnitudes and gives no gradient separating the losers.
     FIX: regress the per-candidate VALUE (gain) directly (SmoothL1) and/or a
     listwise soft-ranking loss. Dense gradient on every candidate every frame;
     tie-agnostic (equal values -> harmless argmax).

  3. TEACHER + COVERAGE. Use the verified best planner config; DAgger relabels
     the NET's own closed-loop states so eval distribution == train distribution.

Usage (VM):
  python3 distill_v4.py --enc crossattn --loss both --K 16 --H 120 --D 8 \
      --gen_seeds 256 --iters 3 --epochs 60 --eval_n 512 \
      --weights predator_weights.json --out v4.json
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sim_torch as st
from sim_torch import Sim
import planner_probe as pp

PS, VS = 200.0, 6.0


# ----------------------------- features -----------------------------

def full_obs(sim):
    """Full permutation-invariant observation.
      boid_feat (B,N,5): [dx/PS, dy/PS, vx/VS, vy/VS, alive]   predator-relative
      mask      (B,N)  : alive bool
      pred_state(B,5)  : [pred_vx/VS, pred_vy/VS, frac_alive, cooldown_frac, size_frac]
    """
    dx = (sim.boid_pos[..., 0] - sim.pred_pos[:, None, 0]) / PS
    dy = (sim.boid_pos[..., 1] - sim.pred_pos[:, None, 1]) / PS
    vx = sim.boid_vel[..., 0] / VS
    vy = sim.boid_vel[..., 1] / VS
    al = sim.boid_alive.to(dx.dtype)
    boid_feat = torch.stack([dx * al, dy * al, vx * al, vy * al, al], dim=2)
    frac_alive = al.mean(dim=1, keepdim=True)
    # feed cooldown: ms since last feed, normalised by the cooldown window
    cd = ((sim._frame_ms - sim.pred_last_feed_ms) / st.PREDATOR_FEED_COOLDOWN_MS)
    cd = cd.clamp(0, 1).to(dx.dtype).unsqueeze(1)
    # predator size sets the catch radius (catch_radius = pred_size*0.7); it is a
    # dynamic state variable gain(s,c) depends on, so the net must see it.
    sz = ((sim.pred_size - st.PREDATOR_BASE_SIZE) /
          (st.PREDATOR_MAX_SIZE - st.PREDATOR_BASE_SIZE)).clamp(0, 1).to(dx.dtype).unsqueeze(1)
    pred_state = torch.cat([sim.pred_vel / VS, frac_alive, cd, sz], dim=1)
    return boid_feat, sim.boid_alive, pred_state


def cand_features(sim, cand):
    """cand (B,K,2) absolute -> (B,K,4): [rel_x/PS, rel_y/PS, dist/PS, is_e3d]."""
    rel = cand - sim.pred_pos[:, None, :]
    dist = torch.sqrt((rel ** 2).sum(-1, keepdim=True))
    is_e3d = torch.zeros_like(dist)
    is_e3d[:, 0, 0] = 1.0
    return torch.cat([rel / PS, dist / PS, is_e3d], dim=-1)


# ----------------------------- networks -----------------------------

class MHA(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0
        self.h, self.hd = heads, dim // heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim); self.o = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        B, Lq, D = q.shape; Lk = k.shape[1]
        Q = self.q(q).view(B, Lq, self.h, self.hd).transpose(1, 2)
        K = self.k(k).view(B, Lk, self.h, self.hd).transpose(1, 2)
        V = self.v(v).view(B, Lk, self.h, self.hd).transpose(1, 2)
        sc = (Q @ K.transpose(-2, -1)) / math.sqrt(self.hd)
        if mask is not None:
            sc = sc.masked_fill(~mask[:, None, None, :], float('-inf'))
        return self.o((F.softmax(sc, -1) @ V).transpose(1, 2).contiguous().view(B, Lq, D))


class ISAB(nn.Module):
    def __init__(self, dim, m=16, heads=4):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, dim) * 0.1)
        self.m1 = MHA(dim, heads); self.m2 = MHA(dim, heads)
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.f1 = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.f2 = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, X, mask=None):
        B = X.shape[0]
        H = self.m1(self.I.expand(B, -1, -1), X, X, mask)
        H = self.ln1(H + self.f1(H))
        out = self.m2(X, H, H)
        return self.ln2(out + self.f2(out))


class SetScorer(nn.Module):
    """Full-set encoder + per-candidate value head.

    enc in {deepsets, transformer, crossattn}:
      deepsets   : phi(boid) -> masked mean+max pool -> g ; score = head([g, cand])
      transformer: ISAB x L -> masked mean+max pool -> g ; score = head([g, cand])
      crossattn  : ISAB x L -> boid tokens; each candidate is a query that cross-
                   attends to (boid tokens + predator token) -> per-cand context.
                   Strictly the most expressive: models "given this target, which
                   boids matter". Still permutation-invariant in the boid set.
    """
    def __init__(self, enc='crossattn', d=64, layers=2, heads=4, hid=128):
        super().__init__()
        self.enc = enc; self.d = d
        self.boid_embed = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.pred_embed = nn.Linear(5, d)
        if enc in ('transformer', 'crossattn'):
            self.isabs = nn.ModuleList([ISAB(d, 16, heads) for _ in range(layers)])
        if enc == 'crossattn':
            self.cand_q = nn.Linear(4, d)
            self.cross = MHA(d, heads)
            self.ln = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(d + 4, hid), nn.GELU(),
                                      nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        else:
            gd = 2 * d + d  # mean+max pool (2d) + predator (d)
            self.head = nn.Sequential(nn.Linear(gd + 4, hid), nn.GELU(),
                                      nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def _masked_pool(self, x, mask):
        m = mask.unsqueeze(-1).to(x.dtype)
        s = (x * m).sum(1)
        cnt = m.sum(1).clamp(min=1)
        mean = s / cnt
        mx = x.masked_fill(~mask.unsqueeze(-1), float('-inf')).max(1).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        return torch.cat([mean, mx], dim=-1)

    def forward(self, boid_feat, mask, pred_state, cand_feat):
        B, N, _ = boid_feat.shape
        K = cand_feat.shape[1]
        x = self.boid_embed(boid_feat) * mask.unsqueeze(-1).to(boid_feat.dtype)
        p = self.pred_embed(pred_state)                       # (B,d)
        if self.enc in ('transformer', 'crossattn'):
            for isab in self.isabs:
                x = isab(x, mask)
        if self.enc == 'crossattn':
            ptok = p.unsqueeze(1)                             # (B,1,d)
            mem = torch.cat([x, ptok], dim=1)                 # (B,N+1,d)
            memmask = torch.cat([mask, torch.ones(B, 1, dtype=torch.bool, device=mask.device)], dim=1)
            q = self.cand_q(cand_feat)                        # (B,K,d)
            ctx = self.ln(q + self.cross(q, mem, mem, memmask))  # (B,K,d)
            s = self.head(torch.cat([ctx, cand_feat], dim=-1)).squeeze(-1)
            return s
        g = torch.cat([self._masked_pool(x, mask), p], dim=-1)   # (B,3d)
        g = g.unsqueeze(1).expand(B, K, g.shape[-1])
        s = self.head(torch.cat([g, cand_feat], dim=-1)).squeeze(-1)
        return s


def num_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------- data gen -----------------------------

@torch.no_grad()
def gen_rollout(net, enc_dev, seeds, frames, device, K, H, D):
    """Roll the policy (planner if net is None, else the net) closed-loop; at each
    decision log full obs + candidate feats + per-candidate gain (teacher label)."""
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    B = sim.B
    roll = Sim(seeds=list(range(B * K)), weights=pp.WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(pp.E3D))
    rows = torch.arange(B, device=device)
    BF, MK, PSt, CF, GN = [], [], [], [], []
    held = None
    if net is not None:
        net.eval()
    f = 0
    while f < frames:
        if f % D == 0:
            cand = pp._candidate_targets(sim, K)
            bf, mk, ps = full_obs(sim)
            cf = cand_features(sim, cand)
            base = pp._save_state(sim); pp._load_state(roll, pp._tile_state(base, K))
            roll_tgt = cand.reshape(B * K, 2).contiguous()
            h = min(H, frames - f)
            c0 = roll.catches.clone()
            for _ in range(h):
                pp._step_with_target(roll, roll_tgt)
            gain = pp.rollout_gain(roll, c0, B, K)
            BF.append(bf.float().cpu()); MK.append(mk.cpu()); PSt.append(ps.float().cpu())
            CF.append(cf.float().cpu()); GN.append(gain.cpu())
            if net is None:
                best = gain.argmax(1)
            else:
                s = net(bf.float(), mk, ps.float(), cf.float())
                best = s.argmax(1)
            held = cand[rows, best]
        pp._step_with_target(sim, held)
        f += 1
    return (torch.cat(BF), torch.cat(MK), torch.cat(PSt), torch.cat(CF),
            torch.cat(GN), sim.catches.cpu().numpy())


# ----------------------------- train / eval -----------------------------

def diagnose(GN):
    g = GN
    gmax = g.max(1).values; gmin = g.min(1).values
    decisive = gmax > gmin
    lab = g.argmax(1)
    e3d_best = (lab == 0)
    print(f"  [diag] tie_frac={float((~decisive).float().mean()):.3f} "
          f"decisive_frac={float(decisive.float().mean()):.3f} "
          f"label_is_E3D={float(e3d_best.float().mean()):.3f} "
          f"E3D_best_among_decisive={float((lab[decisive]==0).float().mean()) if decisive.any() else 0:.3f}",
          flush=True)


def train(data, enc, device, epochs, loss_mode, tau, d, layers, hid, lr, bs, net=None):
    BF, MK, PSt, CF, GN = [t.to(device) for t in data]
    lab = GN.argmax(1)
    gmax = GN.max(1).values; gmin = GN.min(1).values
    decisive = gmax > gmin
    if net is None:
        net = SetScorer(enc, d, layers, hid=hid).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = BF.shape[0]
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = racc = dacc = dn = 0.0
        for j in range(0, n, bs):
            b = perm[j:j + bs]
            s = net(BF[b], MK[b], PSt[b], CF[b])
            loss = 0.0
            if loss_mode in ('value', 'both'):
                loss = loss + F.smooth_l1_loss(s, GN[b])
            if loss_mode in ('listnet', 'both'):
                tgt = F.softmax(GN[b] / tau, dim=1)
                loss = loss + F.kl_div(F.log_softmax(s, dim=1), tgt, reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
            pred = s.argmax(1)
            racc += float((pred == lab[b]).sum())
            db = decisive[b]
            if db.any():
                dacc += float((pred[db] == lab[b][db]).sum()); dn += float(db.sum())
        if ep % 15 == 0 or ep == epochs - 1:
            print(f"    ep{ep}: loss={tot/n:.4f} rank_acc={racc/n:.3f} "
                  f"decisive_acc={dacc/max(dn,1):.3f}", flush=True)
    return net


@torch.no_grad()
def eval_closed_loop(net, seeds, frames, device, K, hold_D=1):
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    net.eval()
    rows = torch.arange(sim.B, device=device)
    pick0 = total = 0
    held = None
    for f in range(frames):
        if held is None or f % hold_D == 0:
            cand = pp._candidate_targets(sim, K)
            bf, mk, ps = full_obs(sim)
            cf = cand_features(sim, cand)
            best = net(bf.float(), mk, ps.float(), cf.float()).argmax(1)
            pick0 += int((best == 0).sum()); total += int(best.numel())
            held = cand[rows, best]
        pp._step_with_target(sim, held)
    return sim.catches.cpu().numpy(), pick0 / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', choices=['deepsets', 'transformer', 'crossattn'], default='crossattn')
    ap.add_argument('--loss', choices=['value', 'listnet', 'both'], default='both')
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--H', type=int, default=120)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--gen_seeds', type=int, default=256)
    ap.add_argument('--gen_seedStart', type=int, default=400000)
    ap.add_argument('--gen_frames', type=int, default=1500)
    ap.add_argument('--iters', type=int, default=3, help='DAgger iterations')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--retain', type=int, default=2)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--hid', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--hold_D', type=int, default=1, help='eval: re-decide every hold_D frames')
    ap.add_argument('--eval_seedStart', type=int, default=200000)
    ap.add_argument('--eval_n', type=int, default=512)
    ap.add_argument('--eval_frames', type=int, default=1500)
    ap.add_argument('--weights', default='predator_weights.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    pp.WEIGHTS = st.load_weights(args.weights, device=device)
    gen_seeds = list(range(args.gen_seedStart, args.gen_seedStart + args.gen_seeds))
    eval_seeds = list(range(args.eval_seedStart, args.eval_seedStart + args.eval_n))
    t0 = time.time()

    print(f"[v4] enc={args.enc} loss={args.loss} K={args.K} H={args.H} D={args.D}", flush=True)
    print("[v4] iter 0: planner on-policy dataset", flush=True)
    d0 = gen_rollout(None, device, gen_seeds, args.gen_frames, device, args.K, args.H, args.D)
    print(f"  planner_mean={d0[5].mean():.2f} decisions={d0[0].shape[0]} {time.time()-t0:.0f}s", flush=True)
    diagnose(d0[4])
    bufs = [d0[:5]]
    net = None
    history = []
    for it in range(args.iters):
        keep = bufs[-args.retain:]
        data = tuple(torch.cat([b[i] for b in keep]) for i in range(5))
        print(f"[v4] iter {it} train: decisions={data[0].shape[0]}", flush=True)
        net = train(data, args.enc, device, args.epochs, args.loss, args.tau,
                    args.d, args.layers, args.hid, args.lr, args.bs, net=None)
        if it == 0:
            print(f"  params={num_params(net)}", flush=True)
        catches, pick0 = eval_closed_loop(net, eval_seeds, args.eval_frames, device, args.K, args.hold_D)
        mean = float(catches.mean()); se = float(catches.std(ddof=1) / np.sqrt(len(catches)))
        pct = 100.0 * (mean - 8.3447) / 8.3447
        print(f"[v4] iter {it} EVAL mean={mean:.3f}±{se:.3f} pct={pct:+.2f}% pick0={pick0:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        history.append(dict(iter=it, mean=mean, se=se, pct=pct, pick0=pick0))
        if it < args.iters - 1:
            dr = gen_rollout(net, device, gen_seeds, args.gen_frames, device, args.K, args.H, args.D)
            print(f"  net_rollout_mean={dr[5].mean():.2f}", flush=True)
            bufs.append(dr[:5])

    res = dict(enc=args.enc, loss=args.loss, K=args.K, H=args.H, D=args.D,
               params=num_params(net), planner_train_mean=float(d0[5].mean()),
               eval_n=args.eval_n, history=history, elapsed=time.time() - t0)
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh)


if __name__ == '__main__':
    main()
