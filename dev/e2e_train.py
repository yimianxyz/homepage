"""End-to-end predator policy: egocentric obs -> MLP -> steering, trained by ES
on the fast parallel sim. No hand-crafted patrol target or attack rule — the
policy sees a polar density+flow grid over all boids and outputs steering.

Smoke-test goal: from random init, does catches climb at all in a few gens?
If yes, the obs+policy+method works and we scale / escalate (PPO, APG).

  python3 dev/e2e_train.py --K 64 --S 16 --frames 2000 --gens 10 \
      --hidden 32 --sigma 0.1 --lr 0.05 --device cuda --out ~/ckpt/e2e1
"""
import argparse, json, math, time
from pathlib import Path
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import Sim, fast_limit, PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE, N_BOIDS
from e2e_obs import build_obs_egocentric

A, R = 8, 3
OBS_DIM = 3 * A * R + 3  # 75


def make_layer_shapes(hidden):
    # obs -> hidden (relu) -> 2 (linear)
    return [(OBS_DIM, hidden), (hidden, 2)]


def param_count(hidden):
    return sum(i * o + o for i, o in make_layer_shapes(hidden))


def unflatten_batched(theta_KP, hidden):
    """theta_KP (K,P) -> list of (W (K,in,out), b (K,out))."""
    K = theta_KP.shape[0]
    layers = []
    off = 0
    for (i, o) in make_layer_shapes(hidden):
        nW = i * o
        W = theta_KP[:, off:off + nW].view(K, i, o); off += nW
        b = theta_KP[:, off:off + o]; off += o
        layers.append((W, b))
    return layers


class E2ESim(Sim):
    def __init__(self, seeds, theta_KP, hidden, S, device='cuda'):
        # dummy weights so parent init doesn't choke; we override _step_predator
        w = {'featureDim': 45, 'inputMean': None, 'inputStd': None,
             'outputScale': 1.0, 'clipMagnitude': 0.0, 'layers': []}
        super().__init__(seeds=seeds, weights=w, device=device,
                         sequential=False, auto_target='random')
        self.K = theta_KP.shape[0]
        self.S = S
        self.hidden = hidden
        self.pol = unflatten_batched(theta_KP, hidden)
        self._force_cap = torch.tensor(PREDATOR_MAX_FORCE, dtype=torch.float64, device=device)

    def _step_predator(self):
        obs = build_obs_egocentric(self.pred_pos, self.pred_vel,
                                   self.boid_pos, self.boid_vel, self.boid_alive)  # (B,75) f32
        x = obs.view(self.K, self.S, OBS_DIM)
        W0, b0 = self.pol[0]; W1, b1 = self.pol[1]
        h = torch.relu(torch.bmm(x, W0) + b0.unsqueeze(1))
        out = torch.bmm(h, W1) + b1.unsqueeze(1)         # (K,S,2)
        steer = out.reshape(self.B, 2).double()
        sx, sy = fast_limit(steer[:, 0], steer[:, 1], PREDATOR_MAX_FORCE)
        nvx = self.pred_vel[:, 0] + sx
        nvy = self.pred_vel[:, 1] + sy
        nvx, nvy = fast_limit(nvx, nvy, PREDATOR_MAX_SPEED)
        self.pred_vel[:, 0] = nvx; self.pred_vel[:, 1] = nvy
        self.pred_pos[:, 0] += nvx; self.pred_pos[:, 1] += nvy
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] > self._wrap_w_max, self._wrap_neg20, self.pred_pos[:, 0])
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] < self._wrap_neg20, self._wrap_w_max, self.pred_pos[:, 0])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] > self._wrap_h_max, self._wrap_neg20, self.pred_pos[:, 1])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] < self._wrap_neg20, self._wrap_h_max, self.pred_pos[:, 1])


def evaluate(theta_KP, hidden, seeds, frames, device):
    K = theta_KP.shape[0]; S = len(seeds)
    seeds_exp = list(seeds) * K
    sim = E2ESim(seeds_exp, theta_KP, hidden, S, device=device)
    for _ in range(frames):
        sim.step()
    per = torch.tensor(sim.catches.float().tolist(), device=device).view(K, S)
    return per.mean(dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=64)
    p.add_argument('--S', type=int, default=16)
    p.add_argument('--frames', type=int, default=2000)
    p.add_argument('--gens', type=int, default=10)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--sigma', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--top_k', type=int, default=16)
    p.add_argument('--seedStart', type=int, default=2000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--holdout', type=int, default=16, help='fixed holdout seeds (from 9000) for true-progress eval')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    assert a.K % 2 == 0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'e2e_log.jsonl', 'a')
    def log(o):
        print(json.dumps(o), flush=True); logf.write(json.dumps(o) + '\n'); logf.flush()

    P = param_count(a.hidden)
    g = torch.Generator(device='cpu').manual_seed(a.seed)
    # small init
    theta = (torch.randn(P, generator=g) * 0.1).to(a.device).to(torch.float32)
    log({'phase': 'start', 'P': P, 'hidden': a.hidden, 'K': a.K, 'S': a.S,
         'frames': a.frames, 'sigma': a.sigma, 'lr': a.lr, 'obs_dim': OBS_DIM})
    H = a.K // 2
    best = -1.0
    # fixed holdout seeds to measure TRUE central-policy progress (not seed noise)
    holdout = list(range(9000, 9000 + a.holdout))
    best_holdout = -1.0
    for gen in range(a.gens):
        t0 = time.time()
        rng = torch.Generator(device='cpu').manual_seed(a.seed + gen * 7919)
        eps_h = torch.randn(H, P, generator=rng).to(a.device).to(torch.float32)
        eps = torch.cat([eps_h, -eps_h], dim=0)
        cand = theta.unsqueeze(0) + a.sigma * eps
        allp = torch.cat([cand, theta.unsqueeze(0)], dim=0)  # (K+1,P)
        seeds = list(range(a.seedStart + gen * a.S, a.seedStart + gen * a.S + a.S))
        means = evaluate(allp, a.hidden, seeds, a.frames, a.device)
        rew = means[:a.K]; base = float(means[a.K].item())
        pair_max = torch.maximum(rew[:H], rew[H:])
        elite = torch.argsort(pair_max, descending=True)[:a.top_k]
        diffs = (rew[:H][elite] - rew[H:][elite]).unsqueeze(1)
        grad = (diffs * eps_h[elite]).sum(0) / (a.top_k * a.sigma)
        std = torch.cat([rew[:H][elite], rew[H:][elite]]).std().clamp_min(1e-6)
        theta = theta + a.lr * grad / std
        gmax = float(means.max().item())
        if gmax > best:
            best = gmax
            torch.save({'theta': allp[int(means.argmax())].cpu(), 'hidden': a.hidden, 'P': P}, out / 'best.pt')
        # TRUE progress: eval the central theta on the FIXED holdout set
        hscore = float(evaluate(theta.unsqueeze(0), a.hidden, holdout, a.frames, a.device)[0].item())
        if hscore > best_holdout:
            best_holdout = hscore
            torch.save({'theta': theta.cpu(), 'hidden': a.hidden, 'P': P}, out / 'best_central.pt')
        torch.save({'theta': theta.cpu(), 'hidden': a.hidden, 'P': P}, out / 'last.pt')
        log({'gen': gen, 'baseline': base, 'mean_pert': float(rew.mean()),
             'best_pert': float(rew.max()), 'best_so_far': best,
             'holdout': hscore, 'best_holdout': best_holdout,
             'gen_s': time.time() - t0, 'seeds_start': seeds[0]})
    log({'phase': 'done', 'best': best})


if __name__ == '__main__':
    main()
