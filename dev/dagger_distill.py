"""DAgger distillation of the lookahead planner into a reactive pointer net.

The planner (planner_probe.run_planner) reaches 14-22 catches by choosing, every
D frames, the best of K candidate patrol targets via true-dynamics rollout. v2/v3
pointer-net distillation on PLANNER-visited states reverted toward baseline in
closed loop — the classic distribution-shift failure (the net visits states the
planner never showed it, and compounding errors snowball).

DAgger fix: iterate (train -> roll the NET closed-loop -> relabel the NET-visited
states with the planner's per-candidate gains -> aggregate -> retrain). The net
is trained on the distribution of states IT induces, which is the only honest
target distribution.

Per-decision label = the same K-candidate rollout gain the planner uses; training
uses the v3 lesson (decisive-frame weighting) so sparse-gain ties to cand 0 (E3D)
don't swamp the signal.

Usage (VM):
  python3 dagger_distill.py --gen_seeds 384 --K 8 --H 60 --D 15 --M 16 \
      --iters 4 --epochs 80 --eval_n 512 --weights predator_weights.json --out dag.json
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sim_torch as st
from sim_torch import Sim
import planner_probe as pp

PS = 200.0


class PointerNet(nn.Module):
    def __init__(self, fdim, gdim=96, ghid=128, shid=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(fdim, ghid), nn.ReLU(),
                                 nn.Linear(ghid, gdim), nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(gdim + 3, shid), nn.ReLU(),
                                    nn.Linear(shid, shid), nn.ReLU(),
                                    nn.Linear(shid, 1))

    def forward(self, ob, cand_rel):
        B, K, _ = cand_rel.shape
        g = self.enc(ob)
        dist = torch.sqrt((cand_rel ** 2).sum(-1, keepdim=True))
        cd = torch.cat([cand_rel / PS, dist / PS], dim=-1)
        g_exp = g.unsqueeze(1).expand(B, K, g.shape[1])
        s = self.scorer(torch.cat([g_exp, cd], dim=-1)).squeeze(-1)
        return s


def _planner_label_at(sim, roll, K, H, D, M, frames_left):
    """At the current state of `sim`, compute (obs, cand_rel, gain) by rolling
    each of K candidates H frames in `roll`. Pure labelling; does not step sim."""
    cand = pp._candidate_targets(sim, K)                 # (B,K,2) absolute
    e3d_rel = cand[:, 0] - sim.pred_pos
    ob = pp.planner_obs(sim, M, e3d_rel)
    cand_rel = cand - sim.pred_pos[:, None, :]
    base = pp._save_state(sim)
    tiled = pp._tile_state(base, K)
    pp._load_state(roll, tiled)
    roll_tgt = cand.reshape(sim.B * K, 2).contiguous()
    h = min(H, frames_left)
    c0 = roll.catches.clone()
    for _ in range(h):
        pp._step_with_target(roll, roll_tgt)
    gain = (roll.catches - c0).reshape(sim.B, K).float()
    return ob, cand_rel, gain, cand


@torch.no_grad()
def rollout_with_labels(net, seeds, frames, device, K, H, D, M):
    """Roll NET closed-loop; at each decision point record planner labels for the
    NET-visited state, then step the env with the NET's chosen candidate."""
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    roll = Sim(seeds=list(range(sim.B * K)), weights=pp.WEIGHTS, device=device,
               auto_target='evolved', auto_target_opts=dict(pp.E3D))
    rows = torch.arange(sim.B, device=device)
    O, C, G = [], [], []
    held = None
    if net is not None:
        net.eval()
    f = 0
    while f < frames:
        if f % D == 0:
            ob, cand_rel, gain, cand = _planner_label_at(sim, roll, K, H, D, M, frames - f)
            O.append(ob.float().cpu()); C.append(cand_rel.float().cpu()); G.append(gain.cpu())
            if net is None:
                best = gain.argmax(dim=1)                 # iter 0: planner on-policy
            else:
                s = net(ob.float(), cand_rel.float())
                best = s.argmax(dim=1)
            held = cand[rows, best]
        pp._step_with_target(sim, held)
        f += 1
    return (torch.cat(O).numpy(), torch.cat(C).numpy(), torch.cat(G).numpy(),
            sim.catches.cpu().numpy())


def train(obs, cand, gain, fdim, device, epochs, mode='weighted', bs=8192, lr=1e-3,
          gdim=96, ghid=128, shid=64, net=None):
    X = torch.from_numpy(obs).to(device)
    Cr = torch.from_numpy(cand).to(device)
    Gn = torch.from_numpy(gain).to(device)
    lab = Gn.argmax(dim=1)
    g2, _ = Gn.topk(2, dim=1)
    margin = (g2[:, 0] - g2[:, 1]).clamp(min=0)
    decisive = Gn.max(dim=1).values > Gn.min(dim=1).values
    if mode == 'decisive':
        keep = decisive
        X, Cr, lab = X[keep], Cr[keep], lab[keep]; w = None
    elif mode == 'weighted':
        w = margin / (margin.mean() + 1e-9)
    else:
        w = None
    if net is None:
        net = PointerNet(fdim, gdim, ghid, shid).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = tacc = 0.0
        for j in range(0, n, bs):
            b = perm[j:j + bs]
            s = net(X[b], Cr[b])
            if w is None:
                loss = F.cross_entropy(s, lab[b])
            else:
                loss = (F.cross_entropy(s, lab[b], reduction='none') * w[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b); tacc += (s.argmax(1) == lab[b]).float().sum().item()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    epoch {ep}: loss={tot/n:.4f} train_acc={tacc/n:.3f}", flush=True)
    return net


@torch.no_grad()
def eval_closed_loop(net, seeds, frames, device, K, M):
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    net.eval()
    rows = torch.arange(sim.B, device=device)
    pick0 = total = 0
    for _ in range(frames):
        cand = pp._candidate_targets(sim, K)
        e3d_rel = cand[:, 0] - sim.pred_pos
        ob = pp.planner_obs(sim, M, e3d_rel).float()
        cand_rel = (cand - sim.pred_pos[:, None, :]).float()
        best = net(ob, cand_rel).argmax(dim=1)
        pick0 += int((best == 0).sum()); total += int(best.numel())
        pp._step_with_target(sim, cand[rows, best])
    return sim.catches.cpu().numpy(), pick0 / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_seeds', type=int, default=384)
    ap.add_argument('--gen_seedStart', type=int, default=400000)
    ap.add_argument('--gen_frames', type=int, default=1500)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--H', type=int, default=60)
    ap.add_argument('--D', type=int, default=15)
    ap.add_argument('--M', type=int, default=16)
    ap.add_argument('--iters', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--mode', default='weighted')
    ap.add_argument('--gdim', type=int, default=96)
    ap.add_argument('--ghid', type=int, default=128)
    ap.add_argument('--shid', type=int, default=64)
    ap.add_argument('--retain', type=int, default=2,
                    help='cap aggregated dataset to the last N rollouts worth of data')
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

    # iter 0: planner on-policy dataset
    print("[dagger] iter 0: planner on-policy dataset", flush=True)
    O, C, G, ptr = rollout_with_labels(None, gen_seeds, args.gen_frames, device,
                                       args.K, args.H, args.D, args.M)
    print(f"  planner_mean={ptr.mean():.2f} samples={O.shape[0]} {time.time()-t0:.0f}s", flush=True)
    fdim = O.shape[1]
    bufs = [(O, C, G)]
    net = None
    history = []
    for it in range(args.iters):
        # aggregate the last `retain` rollouts
        keep = bufs[-args.retain:]
        Oa = np.concatenate([b[0] for b in keep])
        Ca = np.concatenate([b[1] for b in keep])
        Ga = np.concatenate([b[2] for b in keep])
        print(f"[dagger] iter {it} train: agg_samples={Oa.shape[0]} mode={args.mode}", flush=True)
        net = train(Oa, Ca, Ga, fdim, device, args.epochs, mode=args.mode,
                    gdim=args.gdim, ghid=args.ghid, shid=args.shid, net=None)
        catches, pick0 = eval_closed_loop(net, eval_seeds, args.eval_frames, device, args.K, args.M)
        mean = float(catches.mean()); se = float(catches.std(ddof=1) / np.sqrt(len(catches)))
        pct = 100.0 * (mean - 8.3447) / 8.3447
        print(f"[dagger] iter {it} EVAL mean={mean:.3f}±{se:.3f} pct={pct:+.2f}% "
              f"pick0={pick0:.3f} {time.time()-t0:.0f}s", flush=True)
        history.append(dict(iter=it, mean=mean, se=se, pct=pct, pick0=pick0,
                            agg_samples=int(Oa.shape[0])))
        # collect a fresh net-on-policy rollout for the next iter (skip after last)
        if it < args.iters - 1:
            print(f"[dagger] iter {it} collect net-on-policy rollout", flush=True)
            Or, Cr2, Gr, nm = rollout_with_labels(net, gen_seeds, args.gen_frames, device,
                                                  args.K, args.H, args.D, args.M)
            print(f"  net_rollout_mean={nm.mean():.2f} samples={Or.shape[0]}", flush=True)
            bufs.append((Or, Cr2, Gr))

    res = dict(K=args.K, H=args.H, D=args.D, M=args.M, fdim=fdim, iters=args.iters,
               epochs=args.epochs, mode=args.mode, planner_train_mean=float(ptr.mean()),
               eval_n=args.eval_n, eval_seedStart=args.eval_seedStart,
               history=history, elapsed=time.time() - t0)
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh)


if __name__ == '__main__':
    main()
