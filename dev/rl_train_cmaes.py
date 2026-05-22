"""CMA-ES trainer on the same sim_torch sequential+graph eval.

The Algorithm:
  Maintains mean theta and full P×P covariance C (P ~ hundreds → fine).
  Each gen samples K candidates from N(theta, sigma^2 * C), evaluates
  them all in one batched sim run, then:
    - Updates theta = weighted mean of top mu candidates.
    - Updates C via rank-mu update (and rank-1 update via cumulation path).
    - Updates sigma via the conjugate evolution path length.

Why CMA-ES on top of vanilla ES?
  CMA-ES learns the local landscape covariance, so once it finds a
  fertile direction it stretches the sampling distribution along that
  direction (and shrinks it in flat directions). Useful when the
  natural step sizes are anisotropic — typical for NN weight space
  where different layers have very different sensitivities.

This is a minimal-but-correct CMA-ES port; weights/parameters follow
Hansen (2016) "The CMA Evolution Strategy: A Tutorial".

Usage:
    python3 dev/rl_train_cmaes.py --init_from js/predator_weights.json \\
        --K 32 --S 16 --frames 1500 --sigma 0.05 --gens 200 \\
        --device cuda --out dev/checkpoints/cmaes_shipped
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


def cmaes_constants(N: int, K: int):
    """Default CMA-ES strategy parameters (Hansen 2016, eqs. 49–53).
    N: number of parameters. K: population size.
    Returns dict with mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu.
    """
    mu = K // 2
    raw_w = np.log((K + 1) / 2.0) - np.log(np.arange(1, mu + 1))
    weights = raw_w / raw_w.sum()
    mu_eff = 1.0 / (weights ** 2).sum()
    c_sigma = (mu_eff + 2.0) / (N + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (N + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mu_eff / N) / (N + 4.0 + 2.0 * mu_eff / N)
    c_1 = 2.0 / ((N + 1.3) ** 2 + mu_eff)
    c_mu = min(1.0 - c_1,
               2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)
               / ((N + 2.0) ** 2 + mu_eff))
    return dict(mu=mu, weights=weights, mu_eff=mu_eff,
                c_sigma=c_sigma, d_sigma=d_sigma,
                c_c=c_c, c_1=c_1, c_mu=c_mu)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init_from', required=True)
    p.add_argument('--K', type=int, default=32,
                   help='Population size. Memory cost is O(K*P).')
    p.add_argument('--S', type=int, default=16)
    p.add_argument('--seedStart', type=int, default=100)
    p.add_argument('--frames', type=int, default=1500)
    p.add_argument('--sigma', type=float, default=0.05)
    p.add_argument('--gens', type=int, default=200)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--ckpt_every', type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'cmaes_log.jsonl'
    log_f = open(log_path, 'a')
    def log(o):
        line = json.dumps(o)
        print(line, flush=True); log_f.write(line + '\n'); log_f.flush()

    template = load_weights(args.init_from, device=args.device)
    N = param_size(template)
    theta = flatten_weights(template).to(args.device).double()
    sigma = args.sigma

    rng = np.random.default_rng(args.seed)

    # Covariance state — keep on CPU/NumPy for the eigendecomposition.
    C = np.eye(N, dtype=np.float64)
    p_sigma = np.zeros(N, dtype=np.float64)
    p_c = np.zeros(N, dtype=np.float64)

    cc = cmaes_constants(N, args.K)
    mu, weights = cc['mu'], cc['weights']
    mu_eff = cc['mu_eff']
    c_sigma, d_sigma = cc['c_sigma'], cc['d_sigma']
    c_c, c_1, c_mu = cc['c_c'], cc['c_1'], cc['c_mu']
    chi_N = math.sqrt(N) * (1 - 1.0/(4*N) + 1.0/(21*N*N))  # E||N(0,I)||

    log({
        'phase': 'start',
        'N': N, 'K': args.K, 'mu': mu, 'mu_eff': mu_eff,
        'S': args.S, 'frames': args.frames, 'sigma_init': sigma,
        'gens': args.gens, 'init_from': args.init_from,
    })

    # Warmup CUDA
    if args.device == 'cuda' and torch.cuda.is_available():
        warm_seeds = list(range(args.seedStart, args.seedStart + args.S))
        ww = flat_to_batched_weights(theta.float().unsqueeze(0), template)
        ws = Sim(seeds=warm_seeds, weights=ww, device=args.device,
                 sequential=True)
        for _ in range(3): ws.step()
        torch.cuda.synchronize(); del ws; torch.cuda.empty_cache()

    best_baseline = -1.0
    t_start = time.time()
    for gen in range(args.gens):
        gen_t0 = time.time()
        seeds = list(range(args.seedStart, args.seedStart + args.S))

        # Sample K offspring from N(theta, sigma^2 C)
        try:
            evals, B_mat = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-20)
            BD = B_mat * np.sqrt(evals)
        except np.linalg.LinAlgError:
            BD = np.eye(N)

        z = rng.standard_normal((args.K, N))
        y = z @ BD.T                                          # (K, N)
        candidates_np = theta.cpu().numpy() + sigma * y       # (K, N)
        candidates = torch.from_numpy(candidates_np).float().to(args.device)
        all_thetas = torch.cat([candidates, theta.float().unsqueeze(0)], dim=0)

        # Evaluate K + baseline (baseline doesn't affect CMA-ES update; logged for tracking)
        means, _ = evaluate_batched(all_thetas, template,
                                    seeds=seeds, frames=args.frames,
                                    device=args.device, use_graph=True)
        rewards = means[:args.K].cpu().numpy()
        baseline = float(means[args.K].item())

        # Top mu by reward (descending)
        order = np.argsort(-rewards)[:mu]
        y_sel = y[order]                                     # (mu, N)
        z_sel = z[order]                                     # (mu, N)

        # Mean update
        new_theta_np = theta.cpu().numpy() + sigma * (weights @ y_sel)
        theta = torch.from_numpy(new_theta_np).to(args.device)

        # p_sigma update
        Cinv_sqrt = (B_mat * (1.0 / np.sqrt(evals))) @ B_mat.T
        p_sigma = (1 - c_sigma) * p_sigma + \
                  math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (Cinv_sqrt @ (weights @ y_sel))
        # Sigma update
        new_sigma = sigma * math.exp((c_sigma / d_sigma) *
                                     (np.linalg.norm(p_sigma) / chi_N - 1))
        # h_sigma — gate the rank-1 update if step size has been growing too fast
        h_sigma = 1.0 if (np.linalg.norm(p_sigma)
                          / math.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
                          < (1.4 + 2.0/(N + 1)) * chi_N) else 0.0
        # p_c update
        p_c = (1 - c_c) * p_c + \
              h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * (weights @ y_sel)
        # C update: rank-1 + rank-mu
        rank_1 = np.outer(p_c, p_c)
        rank_mu = (y_sel * weights[:, None]).T @ y_sel
        C = (1 - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
        # Symmetrize for numerical safety
        C = (C + C.T) / 2

        sigma = new_sigma

        gen_t = time.time() - gen_t0
        log({
            'gen': gen,
            'baseline_catches': baseline,
            'mean_reward': float(rewards.mean()),
            'best_reward': float(rewards.max()),
            'top_mu_mean': float(rewards[order].mean()),
            'sigma': sigma,
            'p_sigma_norm': float(np.linalg.norm(p_sigma)),
            'gen_seconds': gen_t,
            'total_seconds': time.time() - t_start,
        })

        if baseline > best_baseline:
            best_baseline = baseline
            torch.save({
                'theta': theta.cpu(),
                'C': torch.from_numpy(C),
                'sigma': sigma,
                'gen': gen,
                'baseline_catches': best_baseline,
                'args': vars(args),
            }, out_dir / 'best.pt')
            export_weights_to_js(theta.float(), template,
                                 str(out_dir / 'best.json'))

        if gen % args.ckpt_every == 0 or gen == args.gens - 1:
            torch.save({
                'theta': theta.cpu(),
                'C': torch.from_numpy(C),
                'sigma': sigma,
                'gen': gen,
                'baseline_catches': baseline,
                'args': vars(args),
            }, out_dir / f'ckpt_gen{gen:04d}.pt')
            export_weights_to_js(theta.float(), template,
                                 str(out_dir / f'ckpt_gen{gen:04d}.json'))

    log({'phase': 'done', 'best_baseline': best_baseline,
         'total_seconds': time.time() - t_start})
    log_f.close()


if __name__ == '__main__':
    main()
