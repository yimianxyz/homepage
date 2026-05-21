"""ES training of a teacher predator policy on the GPU sim_torch env.

OpenAI-ES style (Salimans 2017): each generation samples K antithetic
perturbation pairs from a Gaussian, evaluates each in `B` parallel sim
seeds, and uses the rank-weighted rewards to compute a gradient
estimate.

Usage:
    python3 dev/rl_train_gpu.py --arch H=32 --K 32 --B 64 \
        --sigma 0.05 --lr 0.01 --gens 200 --feature_dim 45 \
        --seed_pool_size 10000 --device cuda \
        --out dev/checkpoints/teacher_H32 \
        --eval_every 5
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
    Sim, mulberry32_seq, build_features, fast_set_magnitude, fast_limit,
    PREDATOR_RANGE, PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE, POLICY_K, POLICY_PAD,
    N_BOIDS, MAX_SPEED, MAX_FORCE, DESIRED_SEPARATION, NEIGHBOR_DISTANCE,
    SEP_MULT, COH_MULT, ALI_MULT, EPSILON, CANVAS_W, CANVAS_H, BORDER_OFFSET,
    PREDATOR_TURN_FACTOR, PREDATOR_BASE_SIZE, PREDATOR_MAX_SIZE,
    PREDATOR_GROWTH, PREDATOR_DECAY, PREDATOR_FEED_COOLDOWN_MS, FRAME_MS,
    fast_mag,
)


# ----- Teacher NN definition -----------------------------------------------
#
# Plain MLP: feature_dim → hidden... → 2 (steering xy).
# Output is squashed to PREDATOR_MAX_FORCE magnitude before being applied.
# Easy to scale: pass --arch like "H=32" or "H=64,32" for 2 hidden layers.
class TeacherNN:
    def __init__(self, feature_dim, hidden_dims, device='cuda', dtype=torch.float32):
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.dtype = dtype
        # Build the parameter list as a flat tensor; we serialise/perturb in
        # this space for ES.
        dims = [feature_dim] + list(hidden_dims) + [2]
        self.shapes = []
        size = 0
        for i in range(len(dims) - 1):
            self.shapes.append(('W', (dims[i], dims[i + 1])))
            size += dims[i] * dims[i + 1]
            self.shapes.append(('b', (dims[i + 1],)))
            size += dims[i + 1]
        self.num_params = size
        # Small init (Glorot-ish).
        flat = torch.zeros(size, device=device, dtype=dtype)
        off = 0
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            scale = (1.0 / (in_d + out_d)) ** 0.5
            flat[off:off + in_d * out_d] = torch.randn(in_d * out_d, device=device, dtype=dtype) * scale
            off += in_d * out_d
            off += out_d   # biases stay 0
        self.theta = flat

    def unpack(self, theta_flat):
        """Return list of (W, b) tensors per layer."""
        layers = []
        off = 0
        dims = [self.feature_dim] + list(self.hidden_dims) + [2]
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            W = theta_flat[off:off + in_d * out_d].view(in_d, out_d)
            off += in_d * out_d
            b = theta_flat[off:off + out_d]
            off += out_d
            layers.append((W, b))
        return layers

    def forward(self, features, theta_flat):
        """Run features (B, feature_dim) through the policy with theta_flat
        params. Returns steering (B, 2) clipped to PREDATOR_MAX_FORCE."""
        layers = self.unpack(theta_flat)
        x = features
        for i, (W, b) in enumerate(layers):
            x = x @ W + b
            if i < len(layers) - 1:
                x = torch.relu(x)
        # tanh squash + scale to MAX_FORCE
        x = torch.tanh(x) * PREDATOR_MAX_FORCE
        return x


# ----- Custom sim that uses our teacher policy via flat theta --------------
class TeacherSim(Sim):
    """Subclass of Sim that takes a TeacherNN + flat theta instead of a
    pre-baked weight dict. Lets ES perturb theta without serialising."""

    def __init__(self, seeds, teacher, theta_flat, feature_dim,
                 num_boids=N_BOIDS, auto_target='flock_centroid', device='cpu'):
        self.teacher = teacher
        self.theta_flat = theta_flat
        self.feature_dim_override = feature_dim
        # Build a fake "weights" dict so Sim.__init__ has something to attach
        # (we override the forward in _step_predator).
        self.weights = {'featureDim': feature_dim}
        self.seeds = list(seeds)
        self.B = len(self.seeds)
        self.N = num_boids
        self.auto_target_mode = auto_target
        self.device = device
        self._initialize()

    def _step_predator(self):
        self._update_auto_target()
        feats = build_features(
            self.pred_pos.float(), self.pred_vel.float(),
            self.boid_pos.float(), self.boid_vel.float(), self.boid_alive,
            self.pred_auto.float(), self.feature_dim_override, self.device,
        )
        steering = self.teacher.forward(feats, self.theta_flat).double()
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


# ----- ES driver ------------------------------------------------------------
def evaluate(teacher, theta_flat, seeds, max_frames, feature_dim, device):
    sim = TeacherSim(seeds=seeds, teacher=teacher, theta_flat=theta_flat,
                     feature_dim=feature_dim, device=device,
                     auto_target='flock_centroid')
    out = sim.run(max_frames)
    return out['mean_catches'], out['per_seed_catches']


def es_iteration(teacher, theta_flat, sigma, K, B, max_frames,
                 seeds, feature_dim, device, lr, gen, log):
    """One ES generation. Returns updated theta_flat and stats."""
    n_params = teacher.num_params
    # Antithetic pairs: K perturbation directions × 2 (+ and -).
    # K must be even.
    H = K // 2
    rng = torch.Generator(device='cpu').manual_seed(int(gen) * 1000003 + 7)
    eps_half = torch.randn(H, n_params, generator=rng).to(device).to(theta_flat.dtype)
    eps = torch.cat([eps_half, -eps_half], dim=0)   # (K, n_params)
    rewards = torch.zeros(K, device=device, dtype=theta_flat.dtype)
    for i in range(K):
        cand = theta_flat + sigma * eps[i]
        r, _ = evaluate(teacher, cand, seeds, max_frames, feature_dim, device)
        rewards[i] = r
    # Baseline reward of current θ (single eval, same seeds)
    base_r, _ = evaluate(teacher, theta_flat, seeds, max_frames, feature_dim, device)

    # Rank-based fitness shaping (more stable than raw rewards).
    order = torch.argsort(rewards, descending=True)
    ranks = torch.empty_like(order, dtype=theta_flat.dtype)
    ranks[order] = torch.arange(K, device=device, dtype=theta_flat.dtype)
    # Centered ranks in [-0.5, 0.5]
    centered = ranks / (K - 1) - 0.5
    # OpenAI-ES gradient estimate
    grad = (eps * centered.unsqueeze(1)).sum(dim=0) / (K * sigma)
    # Step
    new_theta = theta_flat + lr * grad

    stats = {
        'gen': gen,
        'baseline_catches': float(base_r),
        'mean_perturbation_catches': float(rewards.mean()),
        'best_perturbation_catches': float(rewards.max()),
        'std_perturbation_catches': float(rewards.std()),
        'grad_norm': float(grad.norm()),
    }
    log(stats)
    return new_theta, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', default='H=32',
                   help='Hidden dims, e.g. "H=32" or "H=64,32" for 2 layers.')
    p.add_argument('--feature_dim', type=int, default=45)
    p.add_argument('--K', type=int, default=32,
                   help='Population size per generation (must be even for antithetic).')
    p.add_argument('--B', type=int, default=64,
                   help='Sim batch size (seeds per perturbation eval).')
    p.add_argument('--frames', type=int, default=5000)
    p.add_argument('--sigma', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--gens', type=int, default=100)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed_pool_size', type=int, default=10000,
                   help='Sample B seeds from [0, seed_pool_size) each generation.')
    p.add_argument('--out', default='dev/checkpoints/teacher')
    p.add_argument('--init_seed', type=int, default=1234)
    p.add_argument('--init_from', default=None,
                   help='Optional path to a starting theta (.pt file).')
    p.add_argument('--eval_every', type=int, default=5,
                   help='Save checkpoint every N generations.')
    p.add_argument('--log_every', type=int, default=1)
    args = p.parse_args()

    # Parse arch
    arch_str = args.arch.replace('H=', '').replace('h=', '')
    hidden_dims = [int(x) for x in arch_str.split(',') if x]

    torch.manual_seed(args.init_seed)
    teacher = TeacherNN(
        feature_dim=args.feature_dim,
        hidden_dims=hidden_dims,
        device=args.device,
    )
    theta = teacher.theta.clone()

    if args.init_from is not None and Path(args.init_from).exists():
        loaded = torch.load(args.init_from, map_location=args.device)
        if isinstance(loaded, dict) and 'theta' in loaded:
            theta = loaded['theta'].to(args.device)
        else:
            theta = loaded.to(args.device)
        print(f"Initialised from {args.init_from}")

    print(f"Arch: feature_dim={args.feature_dim}, hidden={hidden_dims}, params={teacher.num_params}")
    print(f"ES: K={args.K} (antithetic), B={args.B}, sigma={args.sigma}, lr={args.lr}, gens={args.gens}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'es_log.jsonl'
    log_lines = []

    def log(obj):
        line = json.dumps(obj)
        print(line, flush=True)
        log_lines.append(line)
        log_path.write_text('\n'.join(log_lines) + '\n')

    log({
        'phase': 'start',
        'arch': args.arch, 'feature_dim': args.feature_dim,
        'hidden_dims': hidden_dims, 'num_params': teacher.num_params,
        'K': args.K, 'B': args.B, 'sigma': args.sigma, 'lr': args.lr,
        'gens': args.gens, 'frames': args.frames, 'device': args.device,
    })

    seed_rng = np.random.default_rng(args.init_seed + 17)
    t0 = time.time()
    best_baseline = -1
    for gen in range(args.gens):
        # Sample B seeds for this generation (rotating, to keep noise low
        # per-perturbation but prevent overfitting across generations).
        seeds = sorted(seed_rng.choice(args.seed_pool_size, size=args.B, replace=False).tolist())
        gen_t0 = time.time()
        theta, stats = es_iteration(
            teacher, theta, args.sigma, args.K, args.B, args.frames,
            seeds, args.feature_dim, args.device, args.lr, gen, log)
        stats['gen_seconds'] = time.time() - gen_t0
        stats['total_seconds'] = time.time() - t0
        if stats['baseline_catches'] > best_baseline:
            best_baseline = stats['baseline_catches']
            # Save best checkpoint
            torch.save({
                'theta': theta.cpu(),
                'feature_dim': args.feature_dim,
                'hidden_dims': hidden_dims,
                'arch': args.arch,
                'gen': gen,
                'baseline_catches': best_baseline,
            }, out_dir / 'best.pt')

        if gen % args.eval_every == 0 or gen == args.gens - 1:
            ckpt_path = out_dir / f'ckpt_gen{gen:04d}.pt'
            torch.save({
                'theta': theta.cpu(),
                'feature_dim': args.feature_dim,
                'hidden_dims': hidden_dims,
                'arch': args.arch,
                'gen': gen,
                'baseline_catches': stats['baseline_catches'],
            }, ckpt_path)
            log({'phase': 'checkpoint', 'gen': gen, 'path': str(ckpt_path),
                 'baseline_catches': stats['baseline_catches']})

    log({'phase': 'done', 'best_baseline': best_baseline,
         'total_seconds': time.time() - t0})


if __name__ == '__main__':
    main()
