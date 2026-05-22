"""OpenAI-ES style NES on the new sim_torch sequential+graph eval.

Differences from rl_train_v2.py (which uses ARS-V1 elite selection):
  - All K samples contribute to the gradient (no elite selection).
  - Rank-based utility weights:  u_i = max(0, log(K/2 + 1) - log(rank_i + 1))
    (Wierstra et al. 2014). Standardized to sum-to-1, mean-zero.
  - Antithetic sampling for variance reduction.
  - Optional small weight decay on theta (l2 toward init) to prevent
    runaway drift on flat plateaus.

Why try this alongside ARS?
  ARS-V1 with top-K' elite selection can over-fit to outliers, especially
  on a flat-plateau landscape where every direction's signal is noise.
  NES with rank-based weights gives a much smoother estimate of the
  natural gradient — slower convergence on hard landscapes but less
  spurious drift.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import Sim, load_weights
from rl_train_v2 import (flatten_weights, param_size,
                          flat_to_batched_weights, evaluate_batched,
                          export_weights_to_js)


def rank_utility_weights(K: int) -> torch.Tensor:
    """OpenAI-ES / NES rank-based utility weights.

    Centered (mean 0) so the gradient doesn't drift; standardized by
    population size so the step magnitude is comparable across K.
    Returns a (K,) tensor.
    """
    raw = np.maximum(0.0, math.log(K / 2 + 1) - np.log(np.arange(1, K + 1)))
    u = raw / raw.sum() - 1.0 / K
    return torch.from_numpy(u.astype(np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init_from', required=True)
    p.add_argument('--K', type=int, default=128, help='Even.')
    p.add_argument('--S', type=int, default=16)
    p.add_argument('--seedStart', type=int, default=100)
    p.add_argument('--frames', type=int, default=1500)
    p.add_argument('--sigma', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=0.01,
                   help='Smaller default than ARS — NES uses all K.')
    p.add_argument('--max_step_norm', type=float, default=0.02)
    p.add_argument('--gens', type=int, default=300)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--ckpt_every', type=int, default=10)
    p.add_argument('--weight_decay', type=float, default=0.0,
                   help='Per-gen l2 pull toward initial theta. 0 disables.')
    args = p.parse_args()
    assert args.K % 2 == 0

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'nes_log.jsonl'
    log_f = open(log_path, 'a')
    def log(o):
        line = json.dumps(o)
        print(line, flush=True); log_f.write(line + '\n'); log_f.flush()

    torch.manual_seed(args.seed)
    template = load_weights(args.init_from, device=args.device)
    P = param_size(template)
    theta = flatten_weights(template).to(args.device)
    theta_init = theta.clone()
    print(f"loaded init from {args.init_from}, P={P}")

    utility = rank_utility_weights(args.K).to(args.device)

    # CUDA warmup
    if args.device == 'cuda' and torch.cuda.is_available():
        warm_seeds = list(range(args.seedStart, args.seedStart + args.S))
        wbw = flat_to_batched_weights(theta.unsqueeze(0), template)
        ws = Sim(seeds=warm_seeds, weights=wbw, device=args.device, sequential=True)
        for _ in range(3): ws.step()
        torch.cuda.synchronize(); del ws; torch.cuda.empty_cache()

    log({
        'phase': 'start', 'algo': 'nes-openai',
        'K': args.K, 'S': args.S, 'frames': args.frames,
        'sigma': args.sigma, 'lr': args.lr, 'max_step_norm': args.max_step_norm,
        'P': P, 'init_from': args.init_from, 'device': args.device,
        'weight_decay': args.weight_decay, 'seed': args.seed,
    })

    H = args.K // 2
    best_baseline = -1.0
    t_start = time.time()
    for gen in range(args.gens):
        gen_t0 = time.time()
        seeds = list(range(args.seedStart, args.seedStart + args.S))

        rng = torch.Generator(device='cpu').manual_seed(args.seed + gen * 100003)
        eps_half = torch.randn(H, P, generator=rng).to(args.device).float()
        eps = torch.cat([eps_half, -eps_half], dim=0)
        candidates = theta.unsqueeze(0) + args.sigma * eps
        all_thetas = torch.cat([candidates, theta.unsqueeze(0)], dim=0)

        means, _ = evaluate_batched(all_thetas, template,
                                    seeds=seeds, frames=args.frames,
                                    device=args.device, use_graph=True)
        rewards = means[:args.K]
        baseline = float(means[args.K].item())

        theta_pre = theta.clone()
        gen_max_idx = int(means.argmax().item())
        gen_max_score = float(means[gen_max_idx].item())
        gen_max_theta = all_thetas[gen_max_idx].detach().clone()

        # NES gradient via rank-based utility weights.
        sorted_idx = torch.argsort(rewards, descending=True)
        weighted = utility.gather(0, torch.argsort(sorted_idx))
        grad = (weighted.unsqueeze(1) * eps).sum(dim=0) / args.sigma
        step = args.lr * grad
        step_norm = float(step.norm().item())
        if step_norm > args.max_step_norm:
            step = step * (args.max_step_norm / step_norm)
            step_norm = args.max_step_norm

        theta = theta + step
        if args.weight_decay > 0:
            theta = theta + args.weight_decay * (theta_init - theta)

        gen_t = time.time() - gen_t0
        log({
            'gen': gen,
            'baseline_catches': baseline,
            'mean_reward': float(rewards.mean().item()),
            'best_reward': float(rewards.max().item()),
            'std_reward': float(rewards.std().item()),
            'gen_max_score': gen_max_score,
            'gen_max_is_pert': gen_max_idx < args.K,
            'best_so_far': best_baseline,
            'grad_norm': float(grad.norm().item()),
            'step_norm': step_norm,
            'gen_seconds': gen_t,
            'total_seconds': time.time() - t_start,
            'seeds_start': args.seedStart,
        })

        if gen_max_score > best_baseline:
            best_baseline = gen_max_score
            torch.save({
                'theta': gen_max_theta.cpu(), 'P': P, 'gen': gen,
                'baseline_catches': best_baseline,
                'is_perturbation': gen_max_idx < args.K,
                'args': vars(args),
            }, out_dir / 'best.pt')
            export_weights_to_js(gen_max_theta, template,
                                 str(out_dir / 'best.json'))

        if gen % args.ckpt_every == 0 or gen == args.gens - 1:
            torch.save({
                'theta': theta_pre.cpu(), 'P': P, 'gen': gen,
                'baseline_catches': baseline, 'args': vars(args),
            }, out_dir / f'ckpt_gen{gen:04d}.pt')
            export_weights_to_js(theta_pre, template,
                                 str(out_dir / f'ckpt_gen{gen:04d}.json'))

    log({'phase': 'done', 'best_baseline_catches': best_baseline,
         'total_seconds': time.time() - t_start})
    log_f.close()


if __name__ == '__main__':
    main()
