"""Pointer-net distillation v3 — diagnose & fix the baseline-reversion of v2.

v2 (distill_planner2.py) trained a pointer net to score K candidates and pick
argmax, reached high TRAIN accuracy (K8: 0.90) yet closed-loop reverted to
baseline (~8.3). Hypothesis: the per-candidate gain = (catches over H=60 frames)
is SPARSE — with ~8 catches/1500 frames, most decisions have ALL candidates at
gain 0, so `gain.argmax` ties to index 0 = the E3D candidate. The label set is
then dominated by "pick E3D", CE is minimised by always predicting cand 0, and
that IS baseline behaviour. The planner gains only on the MINORITY of decisive
frames; those get swamped.

This script:
  1. Prints the gain-distribution diagnostics (degenerate-tie fraction, argmax
     histogram, frac picking cand 0, frac with any positive gain).
  2. Trains THREE pointer nets from ONE dataset and evals each closed-loop:
       - all       : v2 baseline (CE on every frame, ties -> cand 0)
       - decisive  : CE only on frames where gain.max > gain.min (choice matters)
       - weighted  : CE on every frame, per-sample weight = (max - 2nd max) gain
  3. Logs each net's closed-loop pick distribution (how often it picks cand 0).

Usage (VM):
  python3 distill_planner3.py --gen_seeds 384 --K 8 --H 60 --D 15 --M 16 \
      --epochs 100 --eval_n 512 --eval_seedStart 200000 \
      --weights predator_weights.json --out v3.json
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


def gen_dataset(seeds, frames, device, K, H, D, M, chunk=64):
    O, C, G, cper = [], [], [], []
    for i in range(0, len(seeds), chunk):
        sl = seeds[i:i + chunk]
        obs, cand, gain, catches = pp.run_planner_log_cand(sl, frames, device, K, H, D, M)
        O.append(obs); C.append(cand); G.append(gain); cper.append(catches)
        print(f"  gen chunk {i//chunk}: planner_mean={float(np.mean(catches)):.2f} "
              f"samples={obs.shape[0]}", flush=True)
    return (np.concatenate(O), np.concatenate(C), np.concatenate(G),
            float(np.mean(np.concatenate(cper))))


def diagnose_gain(Gn):
    """Gn: (n,K) torch tensor of per-candidate gains. Print label diagnostics."""
    n, K = Gn.shape
    gmax, _ = Gn.max(dim=1)
    gmin, _ = Gn.min(dim=1)
    lab = Gn.argmax(dim=1)
    degenerate = (gmax == gmin).float().mean().item()        # all candidates equal
    any_pos = (gmax > 0).float().mean().item()               # some candidate catches
    frac_c0 = (lab == 0).float().mean().item()               # argmax is E3D
    hist = torch.bincount(lab, minlength=K).cpu().numpy().tolist()
    # among DECISIVE frames (gmax>gmin), how often is E3D still best?
    dec = gmax > gmin
    frac_c0_dec = (lab[dec] == 0).float().mean().item() if dec.any() else float('nan')
    print(f"[diag] n={n} K={K} degenerate_tie_frac={degenerate:.3f} "
          f"any_positive_gain_frac={any_pos:.3f} frac_argmax_is_E3D={frac_c0:.3f}",
          flush=True)
    print(f"[diag] decisive_frac={1-degenerate:.3f} "
          f"frac_E3D_best_among_decisive={frac_c0_dec:.3f} argmax_hist={hist}",
          flush=True)
    return dict(degenerate_tie_frac=degenerate, any_positive_gain_frac=any_pos,
                frac_argmax_is_E3D=frac_c0, decisive_frac=1 - degenerate,
                frac_E3D_best_among_decisive=frac_c0_dec, argmax_hist=hist)


def train(obs, cand, gain, fdim, device, epochs, mode='all', bs=8192, lr=1e-3,
          gdim=96, ghid=128, shid=64):
    X = torch.from_numpy(obs).to(device)
    Cr = torch.from_numpy(cand).to(device)
    Gn = torch.from_numpy(gain).to(device)
    lab = Gn.argmax(dim=1)
    gmax, _ = Gn.max(dim=1)
    # second-best gain for the margin weight
    g2, _ = Gn.topk(2, dim=1)
    margin = (g2[:, 0] - g2[:, 1])                  # max - 2nd max, >=0
    decisive = gmax > Gn.min(dim=1).values

    if mode == 'decisive':
        keep = decisive
        X, Cr, lab = X[keep], Cr[keep], lab[keep]
        w = None
        print(f"  [train mode=decisive] kept {int(keep.sum())}/{len(keep)} samples", flush=True)
    elif mode == 'weighted':
        # weight = margin, normalised to mean 1 (so LR scale is comparable)
        w = margin.clamp(min=0)
        w = w / (w.mean() + 1e-9)
        print(f"  [train mode=weighted] mean_margin={margin.mean():.4f} "
              f"nonzero_margin_frac={(margin>0).float().mean():.3f}", flush=True)
    else:
        w = None

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
                ce = F.cross_entropy(s, lab[b], reduction='none')
                loss = (ce * w[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
            tacc += (s.argmax(1) == lab[b]).float().sum().item()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  [{mode}] epoch {ep}: loss={tot/n:.4f} train_acc={tacc/n:.3f}", flush=True)
    return net


@torch.no_grad()
def eval_closed_loop(net, seeds, frames, device, K, M):
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    net.eval()
    rows = torch.arange(sim.B, device=device)
    pick0 = 0; total = 0
    for _ in range(frames):
        cand = pp._candidate_targets(sim, K)
        e3d_rel = cand[:, 0] - sim.pred_pos
        ob = pp.planner_obs(sim, M, e3d_rel).float()
        cand_rel = (cand - sim.pred_pos[:, None, :]).float()
        s = net(ob, cand_rel)
        best = s.argmax(dim=1)
        pick0 += int((best == 0).sum()); total += int(best.numel())
        target = cand[rows, best]
        pp._step_with_target(sim, target)
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
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--gdim', type=int, default=96)
    ap.add_argument('--ghid', type=int, default=128)
    ap.add_argument('--shid', type=int, default=64)
    ap.add_argument('--modes', default='all,decisive,weighted')
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

    t0 = time.time()
    gen_seeds = list(range(args.gen_seedStart, args.gen_seedStart + args.gen_seeds))
    print(f"[gen] {len(gen_seeds)} seeds K={args.K} H={args.H} D={args.D} M={args.M}", flush=True)
    obs, cand, gain, ptr_train = gen_dataset(gen_seeds, args.gen_frames, device,
                                             args.K, args.H, args.D, args.M)
    fdim = obs.shape[1]
    print(f"[gen] {obs.shape[0]} samples fdim={fdim} planner_train_mean={ptr_train:.2f} "
          f"{time.time()-t0:.0f}s", flush=True)

    diag = diagnose_gain(torch.from_numpy(gain))

    eval_seeds = list(range(args.eval_seedStart, args.eval_seedStart + args.eval_n))
    results = {}
    for mode in args.modes.split(','):
        mode = mode.strip()
        print(f"[train] mode={mode} epochs={args.epochs}", flush=True)
        net = train(obs, cand, gain, fdim, device, args.epochs, mode=mode,
                    gdim=args.gdim, ghid=args.ghid, shid=args.shid)
        catches, pick0 = eval_closed_loop(net, eval_seeds, args.eval_frames, device, args.K, args.M)
        mean = float(catches.mean()); se = float(catches.std(ddof=1) / np.sqrt(len(catches)))
        pct = 100.0 * (mean - 8.3447) / 8.3447
        results[mode] = dict(distilled_mean=mean, distilled_se=se, pct_vs_baseline=pct,
                             eval_pick0_frac=pick0)
        print(f"[eval] mode={mode} mean={mean:.3f}±{se:.3f} pct={pct:+.2f}% "
              f"pick0_frac={pick0:.3f}", flush=True)

    res = dict(K=args.K, H=args.H, D=args.D, M=args.M, fdim=fdim,
               epochs=args.epochs, planner_train_mean=ptr_train,
               eval_n=args.eval_n, eval_seedStart=args.eval_seedStart,
               diag=diag, modes=results, elapsed=time.time() - t0)
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh)


if __name__ == '__main__':
    main()
