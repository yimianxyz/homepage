"""Distill the lookahead planner's target-selection function into a REACTIVE net.

Pipeline (all on one GPU):
  1. gen  : run the planner over many seeds, log (obs, target_rel) per frame.
  2. train: MLP  obs -> target_rel  (predator-relative 2D point), MSE.
  3. eval : closed-loop. Each frame: build the SAME target-free obs, net predicts
            target_rel, target = pred_pos + rel, then production's analytic
            seek(target) + in-range chase(nearest). Pure reactive, memoryless,
            browser-deployable (net -> target -> existing predator steering).

This keeps production's architecture and only replaces E3D's hand-derived target
function with one learned from the planner. The open question is how much of the
14-22 planner ceiling survives reactive distillation.

Usage (on a VM):
  python3 distill_planner.py --gen_seeds 384 --gen_frames 1500 \
      --K 8 --H 60 --D 15 --M 16 --epochs 40 \
      --eval_seedStart 200000 --eval_n 512 --weights predator_weights.json \
      --save net_planner.pt --out distill_result.json
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn

import sim_torch as st
from sim_torch import Sim, fast_limit, PREDATOR_MAX_SPEED
import planner_probe as pp


class TargetNet(nn.Module):
    def __init__(self, fdim, hidden=(128, 64)):
        super().__init__()
        layers = []
        d = fdim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def gen_dataset(seeds, frames, device, K, H, D, M, chunk=64):
    """Generate planner dataset in seed-chunks to bound rollout memory."""
    obs_all, tgt_all = [], []
    cper = []
    for i in range(0, len(seeds), chunk):
        sl = seeds[i:i + chunk]
        obs, tgt, catches = pp.run_planner_log(sl, frames, device, K, H, D, M)
        obs_all.append(obs); tgt_all.append(tgt); cper.append(catches)
        print(f"  gen chunk {i//chunk}: seeds {sl[0]}..{sl[-1]} "
              f"planner_mean={float(np.mean(catches)):.2f} samples={obs.shape[0]}",
              flush=True)
    return (np.concatenate(obs_all), np.concatenate(tgt_all),
            float(np.mean(np.concatenate(cper))))


def train_net(obs, tgt, fdim, device, epochs, bs=4096, lr=1e-3, hidden=(128, 64)):
    X = torch.from_numpy(obs).to(device)
    Y = torch.from_numpy(tgt).to(device) / 200.0   # scale target to obs units
    net = TargetNet(fdim, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for j in range(0, n, bs):
            b = perm[j:j + bs]
            pred = net(X[b])
            loss = ((pred - Y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep}: mse={tot/n:.5f}", flush=True)
    return net


@torch.no_grad()
def eval_closed_loop(net, seeds, frames, device, M):
    """Reactive closed-loop: net predicts target each frame, analytic seek+chase."""
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    net.eval()
    for _ in range(frames):
        e3d_rel = pp._e3d_target(sim) - sim.pred_pos
        ob = pp.planner_obs(sim, M, e3d_rel).float()
        rel = net(ob).double() * 200.0           # undo scaling
        target = sim.pred_pos + rel
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
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--hidden', default='128,64')
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
    hidden = tuple(int(x) for x in args.hidden.split(','))

    t0 = time.time()
    gen_seeds = list(range(args.gen_seedStart, args.gen_seedStart + args.gen_seeds))
    print(f"[gen] {len(gen_seeds)} seeds K={args.K} H={args.H} D={args.D} M={args.M}", flush=True)
    obs, tgt, planner_train_mean = gen_dataset(
        gen_seeds, args.gen_frames, device, args.K, args.H, args.D, args.M)
    fdim = obs.shape[1]
    print(f"[gen] done: {obs.shape[0]} samples, fdim={fdim}, "
          f"planner_train_mean={planner_train_mean:.2f}, {time.time()-t0:.0f}s", flush=True)

    print(f"[train] epochs={args.epochs} hidden={hidden}", flush=True)
    net = train_net(obs, tgt, fdim, device, args.epochs, hidden=hidden)

    eval_seeds = list(range(args.eval_seedStart, args.eval_seedStart + args.eval_n))
    print(f"[eval] closed-loop n={args.eval_n} seedStart={args.eval_seedStart}", flush=True)
    catches = eval_closed_loop(net, eval_seeds, args.eval_frames, device, args.M)
    mean = float(catches.mean()); se = float(catches.std(ddof=1) / np.sqrt(len(catches)))

    res = dict(gen_seeds=args.gen_seeds, K=args.K, H=args.H, D=args.D, M=args.M,
               fdim=fdim, hidden=list(hidden), epochs=args.epochs,
               planner_train_mean=planner_train_mean,
               eval_n=args.eval_n, eval_seedStart=args.eval_seedStart,
               distilled_mean=mean, distilled_se=se,
               pct_vs_baseline=100.0 * (mean - 8.3447) / 8.3447,
               elapsed=time.time() - t0)
    print(json.dumps(res), flush=True)
    if args.save:
        torch.save({'state_dict': net.state_dict(), 'fdim': fdim,
                    'hidden': list(hidden), 'M': args.M}, args.save)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh)


if __name__ == '__main__':
    main()
