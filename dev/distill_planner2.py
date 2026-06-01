"""Distill the planner's target choice into a reactive POINTER net (v2).

v1 (distill_planner.py) regressed a 2D target and FAILED (7.5-7.8 < 8.14 baseline)
because the planner's choice is discrete/multimodal and MSE averages competing
targets into a meaningless midpoint. v2 matches the planner's action space: the net
SCORES the same K candidates the planner ranks (E3D target + K-1 nearest boids,
lead-adjusted) and picks argmax. Supervised by the planner's per-candidate rollout
gain (cross-entropy on the argmax, optional value-regression aux).

Deployable: candidate construction is the existing E3D target + nearest-boid lead
(computable in JS); the net is a small global-encoder + per-candidate scorer; pick
argmax; seek that point with the existing analytic chase+seek. Reactive, memoryless.

Usage (VM):
  python3 distill_planner2.py --gen_seeds 384 --K 8 --H 60 --D 15 --M 16 \
      --epochs 80 --eval_n 512 --eval_seedStart 200000 --weights predator_weights.json \
      --save ptr.pt --out ptr.json
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
        # ob (B,F); cand_rel (B,K,2) raw predator-rel offsets
        B, K, _ = cand_rel.shape
        g = self.enc(ob)                                  # (B,gd)
        dist = torch.sqrt((cand_rel ** 2).sum(-1, keepdim=True))   # (B,K,1)
        cd = torch.cat([cand_rel / PS, dist / PS], dim=-1)         # (B,K,3)
        g_exp = g.unsqueeze(1).expand(B, K, g.shape[1])
        s = self.scorer(torch.cat([g_exp, cd], dim=-1)).squeeze(-1)  # (B,K)
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


def train(obs, cand, gain, fdim, device, epochs, bs=8192, lr=1e-3,
          gdim=96, ghid=128, shid=64, val_aux=0.3):
    X = torch.from_numpy(obs).to(device)
    Cr = torch.from_numpy(cand).to(device)
    Gn = torch.from_numpy(gain).to(device)
    lab = Gn.argmax(dim=1)
    net = PointerNet(fdim, gdim, ghid, shid).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = X.shape[0]
    # normalize gain targets for the value-aux head (regress scores->gain)
    gmu, gsd = Gn.mean(), Gn.std() + 1e-6
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = tacc = 0.0
        for j in range(0, n, bs):
            b = perm[j:j + bs]
            s = net(X[b], Cr[b])
            ce = F.cross_entropy(s, lab[b])
            aux = ((s - (Gn[b] - gmu) / gsd) ** 2).mean()
            loss = ce + val_aux * aux
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
            tacc += (s.argmax(1) == lab[b]).float().sum().item()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  epoch {ep}: loss={tot/n:.4f} train_acc={tacc/n:.3f}", flush=True)
    return net


@torch.no_grad()
def eval_closed_loop(net, seeds, frames, device, K, M):
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    net.eval()
    rows = torch.arange(sim.B, device=device)
    for _ in range(frames):
        cand = pp._candidate_targets(sim, K)             # (B,K,2) absolute
        e3d_rel = cand[:, 0] - sim.pred_pos
        ob = pp.planner_obs(sim, M, e3d_rel).float()
        cand_rel = (cand - sim.pred_pos[:, None, :]).float()
        s = net(ob, cand_rel)
        best = s.argmax(dim=1)
        target = cand[rows, best]
        pp._step_with_target(sim, target)
    return sim.catches.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_seeds', type=int, default=384)
    ap.add_argument('--gen_seedStart', type=int, default=400000)
    ap.add_argument('--gen_frames', type=int, default=1500)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--H', type=int, default=60)
    ap.add_argument('--D', type=int, default=15)
    ap.add_argument('--M', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--gdim', type=int, default=96)
    ap.add_argument('--ghid', type=int, default=128)
    ap.add_argument('--shid', type=int, default=64)
    ap.add_argument('--val_aux', type=float, default=0.3)
    ap.add_argument('--eval_seedStart', type=int, default=200000)
    ap.add_argument('--eval_n', type=int, default=512)
    ap.add_argument('--eval_frames', type=int, default=1500)
    ap.add_argument('--weights', default='predator_weights.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save', default=None)
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

    print(f"[train] epochs={args.epochs}", flush=True)
    net = train(obs, cand, gain, fdim, device, args.epochs,
                gdim=args.gdim, ghid=args.ghid, shid=args.shid, val_aux=args.val_aux)

    eval_seeds = list(range(args.eval_seedStart, args.eval_seedStart + args.eval_n))
    print(f"[eval] closed-loop n={args.eval_n}", flush=True)
    catches = eval_closed_loop(net, eval_seeds, args.eval_frames, device, args.K, args.M)
    mean = float(catches.mean()); se = float(catches.std(ddof=1) / np.sqrt(len(catches)))
    res = dict(K=args.K, H=args.H, D=args.D, M=args.M, fdim=fdim,
               gdim=args.gdim, ghid=args.ghid, shid=args.shid,
               epochs=args.epochs, planner_train_mean=ptr_train,
               eval_n=args.eval_n, eval_seedStart=args.eval_seedStart,
               distilled_mean=mean, distilled_se=se,
               pct_vs_baseline=100.0 * (mean - 8.3447) / 8.3447,
               elapsed=time.time() - t0)
    print(json.dumps(res), flush=True)
    if args.save:
        torch.save({'state_dict': net.state_dict(), 'fdim': fdim, 'K': args.K,
                    'M': args.M, 'gdim': args.gdim, 'ghid': args.ghid,
                    'shid': args.shid}, args.save)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh)


if __name__ == '__main__':
    main()
