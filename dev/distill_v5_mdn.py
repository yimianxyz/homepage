"""distill_v5_mdn — purest end-to-end: net outputs a CONTINUOUS patrol target.

No candidate set at eval. The net maps the full boid set -> a target offset
(predator-relative), which feeds production's exact analytic chase/seek. This is
the most production-aligned shape (production's net also emits a target/force),
but distilled from the PLANNER instead of the evolved-patrol heuristic.

The planner's committed target is multimodal (which boid to chase), so plain MSE
behaviour-cloning mode-averages to the centroid == baseline. We avoid that with a
MIXTURE-DENSITY head: predict M Gaussian components over the 2-D offset, train by
NLL, and at eval pick the highest-weight component's mean (a true mode, not the
average). DAgger relabels the net's own closed-loop states.

Usage (VM):
  python3 distill_v5_mdn.py --d 192 --layers 4 --comps 6 \
      --gen_seeds 512 --gen_chunk 256 --iters 3 --epochs 80 --eval_n 512 --out mdn.json
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sim_torch as st
from sim_torch import Sim
import planner_probe as pp
from distill_v4 import full_obs, ISAB, num_params

PS, VS = 200.0, 6.0
BASE = 8.3447


class MDNNet(nn.Module):
    def __init__(self, d=192, layers=4, heads=4, hid=384, comps=6):
        super().__init__()
        self.comps = comps
        self.boid_embed = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.pred_embed = nn.Linear(5, d)
        self.isabs = nn.ModuleList([ISAB(d, 16, heads) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(3 * d, hid), nn.GELU(),
                                  nn.Linear(hid, hid), nn.GELU())
        self.pi = nn.Linear(hid, comps)
        self.mu = nn.Linear(hid, comps * 2)
        self.ls = nn.Linear(hid, comps * 2)

    def _pool(self, x, mask):
        m = mask.unsqueeze(-1).to(x.dtype)
        mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
        mx = x.masked_fill(~mask.unsqueeze(-1), float('-inf')).max(1).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        return torch.cat([mean, mx], dim=-1)

    def forward(self, boid_feat, mask, pred_state):
        x = self.boid_embed(boid_feat) * mask.unsqueeze(-1).to(boid_feat.dtype)
        for isab in self.isabs:
            x = isab(x, mask)
        g = torch.cat([self._pool(x, mask), self.pred_embed(pred_state)], dim=-1)
        h = self.head(g)
        B = h.shape[0]
        pi = self.pi(h)                                  # (B,M)
        mu = self.mu(h).view(B, self.comps, 2)           # (B,M,2) offset/PS
        ls = self.ls(h).view(B, self.comps, 2).clamp(-4, 2)
        return pi, mu, ls

    def best_offset(self, boid_feat, mask, pred_state):
        pi, mu, _ = self.forward(boid_feat, mask, pred_state)
        k = pi.argmax(1)
        return mu[torch.arange(mu.shape[0], device=mu.device), k] * PS  # (B,2) abs offset


def mdn_nll(pi, mu, ls, y):
    """y (B,2) offset/PS. Diagonal-Gaussian mixture NLL."""
    yk = y.unsqueeze(1)                                  # (B,1,2)
    var = torch.exp(2 * ls)
    logN = (-0.5 * (((yk - mu) ** 2) / var + 2 * ls + np.log(2 * np.pi))).sum(-1)  # (B,M)
    logw = F.log_softmax(pi, dim=1)
    return -torch.logsumexp(logw + logN, dim=1).mean()


@torch.no_grad()
def gen(net, seeds, frames, device, K, H, D, chunk):
    """Planner (or net) closed-loop; log full_obs + committed target offset/PS."""
    BF, MK, PSt, TG = [], [], [], []
    catches = []
    for i in range(0, len(seeds), chunk):
        sub = seeds[i:i + chunk]
        sim = Sim(seeds=sub, weights=pp.WEIGHTS, device=device,
                  auto_target='evolved', auto_target_opts=dict(pp.E3D))
        B = sim.B
        roll = Sim(seeds=list(range(B * K)), weights=pp.WEIGHTS, device=device,
                   auto_target='evolved', auto_target_opts=dict(pp.E3D))
        rows = torch.arange(B, device=device)
        held = None
        f = 0
        while f < frames:
            if f % D == 0:
                bf, mk, ps = full_obs(sim)
                if net is None:
                    cand = pp._candidate_targets(sim, K)
                    base = pp._save_state(sim); pp._load_state(roll, pp._tile_state(base, K))
                    rt = cand.reshape(B * K, 2).contiguous()
                    h = min(H, frames - f); c0 = roll.catches.clone()
                    for _ in range(h):
                        pp._step_with_target(roll, rt)
                    gain = (roll.catches - c0).reshape(B, K).float()
                    held = cand[rows, gain.argmax(1)]
                else:
                    net.eval()
                    held = sim.pred_pos + net.best_offset(bf.float(), mk, ps.float())
                BF.append(bf.float().cpu()); MK.append(mk.cpu()); PSt.append(ps.float().cpu())
                TG.append(((held - sim.pred_pos) / PS).float().cpu())
            pp._step_with_target(sim, held)
            f += 1
        catches.append(sim.catches.cpu().numpy())
        print(f"    chunk {i//chunk}: rows~{B*frames//D} gen_mean={catches[-1].mean():.2f}", flush=True)
    return (torch.cat(BF), torch.cat(MK), torch.cat(PSt), torch.cat(TG),
            float(np.concatenate(catches).mean()))


@torch.no_grad()
def eval_cl(net, seeds, frames, device, D):
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    net.eval(); held = None
    for f in range(frames):
        if f % D == 0:
            bf, mk, ps = full_obs(sim)
            held = sim.pred_pos + net.best_offset(bf.float(), mk, ps.float())
        pp._step_with_target(sim, held)
    return sim.catches.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=192)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--hid', type=int, default=384)
    ap.add_argument('--comps', type=int, default=6)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--H', type=int, default=120)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--gen_seeds', type=int, default=512)
    ap.add_argument('--gen_chunk', type=int, default=256)
    ap.add_argument('--gen_seedStart', type=int, default=400000)
    ap.add_argument('--gen_frames', type=int, default=1500)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--retain', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bs', type=int, default=8192)
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
    print(f"[mdn] d={args.d} L={args.layers} comps={args.comps} gen_seeds={args.gen_seeds} "
          f"K={args.K} H={args.H} D={args.D}", flush=True)

    print("[mdn] iter 0: planner data gen", flush=True)
    *data, gmean = gen(None, gen_seeds, args.gen_frames, device, args.K, args.H, args.D, args.gen_chunk)
    print(f"  planner_mean={gmean:.2f} rows={data[0].shape[0]} {time.time()-t0:.0f}s", flush=True)
    bufs = [tuple(data)]
    net = MDNNet(args.d, args.layers, args.heads, args.hid, args.comps).to(device)
    print(f"  params={num_params(net)}", flush=True)
    history = []
    for it in range(args.iters):
        keep = bufs[-args.retain:]
        BF, MK, PSt, TG = [torch.cat([b[i] for b in keep]).to(device) for i in range(4)]
        n = BF.shape[0]
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        print(f"[mdn] iter {it} train rows={n}", flush=True)
        for ep in range(args.epochs):
            net.train(); perm = torch.randperm(n, device=device); tot = 0.0
            for j in range(0, n, args.bs):
                b = perm[j:j + args.bs]
                pi, mu, ls = net(BF[b], MK[b], PSt[b])
                loss = mdn_nll(pi, mu, ls, TG[b])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
                tot += loss.item() * len(b)
            sched.step()
            if ep % 20 == 0 or ep == args.epochs - 1:
                print(f"    ep{ep}: nll={tot/n:.4f}", flush=True)
        catches = eval_cl(net, eval_seeds, args.eval_frames, device, args.D)
        mean = float(catches.mean()); se = float(catches.std(ddof=1) / np.sqrt(len(catches)))
        print(f"[mdn] iter {it} EVAL mean={mean:.3f}±{se:.3f} vs_base={100*(mean-BASE)/BASE:+.1f}% "
              f"of_planner={100*mean/21.40:.1f}% {time.time()-t0:.0f}s", flush=True)
        history.append(dict(iter=it, mean=mean, se=se))
        if args.out:
            with open(args.out, 'w') as fh:
                json.dump(dict(args=vars(args), params=num_params(net),
                               planner_mean=gmean, history=history), fh)
        if it < args.iters - 1:
            print(f"[mdn] iter {it} DAgger relabel", flush=True)
            *dr, drm = gen(net, gen_seeds, args.gen_frames, device, args.K, args.H, args.D, args.gen_chunk)
            print(f"  net_rollout_mean={drm:.2f}", flush=True)
            bufs.append(tuple(dr))
    print("DONE", json.dumps(history), flush=True)


if __name__ == '__main__':
    main()
