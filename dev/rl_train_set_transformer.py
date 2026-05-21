"""ES training of a Set Transformer policy. Builds per-boid features
(no K-nearest pooling) so the network has access to the full set and
can decide attention patterns itself.

Per-boid feature (5 floats per boid):
  dx (relative x to predator)
  dy (relative y to predator)
  vx (boid x velocity)
  vy (boid y velocity)
  d  (distance to predator, redundant but useful as positional encoding)

Predator state (4 floats):
  pred_vx, pred_vy
  autoTarget_dx, autoTarget_dy

The policy is a SetTransformerPolicy (set_transformer.py).
"""
import argparse
import json
import time
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, 'dev')
from sim_torch import (
    Sim, mulberry32_seq, fast_set_magnitude, fast_limit,
    PREDATOR_RANGE, PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE,
    N_BOIDS, CANVAS_W, CANVAS_H, FRAME_MS, EPSILON,
    PREDATOR_FEED_COOLDOWN_MS, PREDATOR_GROWTH, PREDATOR_MAX_SIZE,
    PREDATOR_BASE_SIZE, PREDATOR_DECAY,
)
from set_transformer import SetTransformerPolicy, num_params


def build_set_features(pred_pos, pred_vel, boid_pos, boid_vel, boid_alive,
                       auto_target, device):
    """Build per-boid features for the Set Transformer.

    Returns:
      boid_feats: (B, N, 5)
      pred_state: (B, 4)
      boid_mask: (B, N) bool
    """
    B = pred_pos.shape[0]
    N = boid_pos.shape[1]
    dx = boid_pos[:, :, 0] - pred_pos[:, None, 0]
    dy = boid_pos[:, :, 1] - pred_pos[:, None, 1]
    d = torch.sqrt(dx * dx + dy * dy + 1e-12)
    vx = boid_vel[:, :, 0]
    vy = boid_vel[:, :, 1]
    boid_feats = torch.stack([dx, dy, vx, vy, d], dim=-1).float()
    pred_state = torch.stack([
        pred_vel[:, 0].float(),
        pred_vel[:, 1].float(),
        (auto_target[:, 0] - pred_pos[:, 0]).float(),
        (auto_target[:, 1] - pred_pos[:, 1]).float(),
    ], dim=-1)
    return boid_feats, pred_state, boid_alive


class SetSim(Sim):
    """Sim with Set Transformer policy injection."""
    def __init__(self, seeds, policy_module, num_boids=N_BOIDS,
                 auto_target='flock_centroid', device='cpu'):
        self.policy_module = policy_module
        self.weights = {'featureDim': 0}  # dummy
        self.seeds = list(seeds)
        self.B = len(self.seeds)
        self.N = num_boids
        self.auto_target_mode = auto_target
        self.device = device
        self._initialize()

    def _step_predator(self):
        self._update_auto_target()
        boid_feats, pred_state, boid_mask = build_set_features(
            self.pred_pos.float(), self.pred_vel.float(),
            self.boid_pos.float(), self.boid_vel.float(), self.boid_alive,
            self.pred_auto.float(), self.device,
        )
        with torch.no_grad():
            steering = self.policy_module(boid_feats, boid_mask, pred_state).double()
        new_vx = self.pred_vel[:, 0] + steering[:, 0]
        new_vy = self.pred_vel[:, 1] + steering[:, 1]
        new_vx, new_vy = fast_limit(new_vx, new_vy, PREDATOR_MAX_SPEED)
        self.pred_vel[:, 0] = new_vx
        self.pred_vel[:, 1] = new_vy
        self.pred_pos[:, 0] += new_vx
        self.pred_pos[:, 1] += new_vy
        d = self.device
        dt = torch.float64
        m = CANVAS_W + 20
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] > m, torch.tensor(-20.0, dtype=dt, device=d), self.pred_pos[:, 0])
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] < -20, torch.tensor(float(m), dtype=dt, device=d), self.pred_pos[:, 0])
        m = CANVAS_H + 20
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] > m, torch.tensor(-20.0, dtype=dt, device=d), self.pred_pos[:, 1])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] < -20, torch.tensor(float(m), dtype=dt, device=d), self.pred_pos[:, 1])


def flatten_params(module):
    return torch.cat([p.detach().flatten() for p in module.parameters()])


def set_flat_params(module, flat):
    off = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[off:off+n].view_as(p))
        off += n


def evaluate(policy, theta_flat, seeds, max_frames, device):
    set_flat_params(policy, theta_flat)
    sim = SetSim(seeds=seeds, policy_module=policy, device=device,
                 auto_target='flock_centroid')
    out = sim.run(max_frames)
    return out['mean_catches'], out['per_seed_catches']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--num_inducing', type=int, default=16)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--K', type=int, default=16,
                   help='Population size (must be even).')
    p.add_argument('--B', type=int, default=16)
    p.add_argument('--frames', type=int, default=2500)
    p.add_argument('--sigma', type=float, default=0.02)
    p.add_argument('--lr', type=float, default=0.5)
    p.add_argument('--gens', type=int, default=100)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='dev/checkpoints/set_teacher')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--init_from', default=None)
    p.add_argument('--fixed_seeds', type=int, default=16)
    p.add_argument('--top_k', type=int, default=4)
    p.add_argument('--max_step_norm', type=float, default=0.05)
    p.add_argument('--eval_every', type=int, default=5)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    policy = SetTransformerPolicy(d_model=args.d_model,
                                   num_inducing=args.num_inducing,
                                   num_heads=args.num_heads).to(args.device)
    n_p = num_params(policy)
    print(f"SetTransformerPolicy: {n_p} params, d_model={args.d_model}, m={args.num_inducing}, h={args.num_heads}")

    theta = flatten_params(policy).to(args.device).to(torch.float32)
    if args.init_from:
        loaded = torch.load(args.init_from, map_location=args.device)
        theta = loaded['theta'].to(args.device).to(torch.float32) if isinstance(loaded, dict) else loaded.to(args.device).to(torch.float32)
        print(f"Loaded init from {args.init_from}, {theta.numel()} params")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'es_log.jsonl'
    log_lines = []
    def log(obj):
        line = json.dumps(obj)
        print(line, flush=True)
        log_lines.append(line)
        log_path.write_text('\n'.join(log_lines) + '\n')

    log({'phase': 'start', 'd_model': args.d_model, 'num_inducing': args.num_inducing,
         'num_heads': args.num_heads, 'num_params': n_p, 'K': args.K, 'B': args.B,
         'frames': args.frames, 'sigma': args.sigma, 'lr': args.lr, 'gens': args.gens,
         'top_k': args.top_k, 'fixed_seeds': args.fixed_seeds,
         'max_step_norm': args.max_step_norm})

    seeds = list(range(100, 100 + args.fixed_seeds))
    t0 = time.time()
    best_baseline = -1
    H = args.K // 2
    for gen in range(args.gens):
        gen_t0 = time.time()
        rng = torch.Generator(device='cpu').manual_seed(gen * 1000003 + 7)
        eps_half = torch.randn(H, n_p, generator=rng).to(args.device).float()
        eps = torch.cat([eps_half, -eps_half], dim=0)
        rewards = torch.zeros(args.K, device=args.device, dtype=torch.float32)
        for i in range(args.K):
            cand = theta + args.sigma * eps[i]
            r, _ = evaluate(policy, cand, seeds, args.frames, args.device)
            rewards[i] = r
        base_r, _ = evaluate(policy, theta, seeds, args.frames, args.device)

        # ARS elite selection
        pair_max = torch.maximum(rewards[:H], rewards[H:])
        elite_idx = torch.argsort(pair_max, descending=True)[:args.top_k]
        diffs = (rewards[:H][elite_idx] - rewards[H:][elite_idx]).unsqueeze(1)
        grad = (diffs * eps[:H][elite_idx]).sum(dim=0) / (args.top_k * args.sigma)
        reward_std = torch.cat([rewards[:H][elite_idx], rewards[H:][elite_idx]]).std().clamp_min(1e-6)
        grad = grad / reward_std
        step = args.lr * grad
        step_norm = step.norm().item()
        if step_norm > args.max_step_norm:
            step = step * (args.max_step_norm / step_norm)
        theta = theta + step

        stats = {
            'gen': gen,
            'baseline_catches': float(base_r),
            'mean_perturbation_catches': float(rewards.mean()),
            'best_perturbation_catches': float(rewards.max()),
            'std_perturbation_catches': float(rewards.std()),
            'grad_norm': float(grad.norm()),
            'step_norm': float(step.norm()),
            'gen_seconds': time.time() - gen_t0,
            'total_seconds': time.time() - t0,
        }
        log(stats)

        if base_r > best_baseline:
            best_baseline = float(base_r)
            torch.save({'theta': theta.cpu(), 'arch': 'set_transformer',
                        'd_model': args.d_model, 'num_inducing': args.num_inducing,
                        'num_heads': args.num_heads, 'gen': gen,
                        'baseline_catches': best_baseline},
                       out_dir / 'best.pt')

        if gen % args.eval_every == 0 or gen == args.gens - 1:
            ckpt = out_dir / f'ckpt_gen{gen:04d}.pt'
            torch.save({'theta': theta.cpu(), 'arch': 'set_transformer',
                        'd_model': args.d_model, 'num_inducing': args.num_inducing,
                        'num_heads': args.num_heads, 'gen': gen,
                        'baseline_catches': float(base_r)}, ckpt)

    log({'phase': 'done', 'best_baseline': best_baseline,
         'total_seconds': time.time() - t0})


if __name__ == '__main__':
    main()
