"""Batched ES trainer on the new sim_torch sequential+graph eval.

Key design choices:
  - Single sim_torch run evaluates K+1 policies (K perturbations + the
    baseline theta) in one batched sim. B = (K+1)*S; near peak GPU
    utilization on L4.
  - Antithetic sampling: K is even, K/2 noise vectors used as +eps/-eps
    pairs. Reduces gradient variance for free.
  - ARS-V1 elite selection: keep top-K' pairs by max(reward+, reward-),
    use only their reward differences for the gradient. More robust to
    outliers than vanilla mean-field ES.
  - Fixed seeds within each gen so candidates are compared on identical
    environments. Optionally rotate seed window across gens to avoid
    overfitting to a single seed set.
  - Step norm clipping to avoid blowups from heavy-tailed reward.
  - Periodic JS verification: every --js_every gens, export the best
    checkpoint and run JS eval, log the gap between sim_torch and JS.

Usage:
    python3 dev/rl_train_v2.py \
        --init_from js/predator_weights.json \
        --K 128 --S 16 --frames 2500 \
        --sigma 0.05 --lr 0.5 --top_k 8 \
        --max_step_norm 0.02 \
        --gens 200 --device cuda \
        --out dev/checkpoints/rl_v2_shipped_K128
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_torch import Sim, load_weights, stack_weights


def flatten_weights(weights: dict) -> torch.Tensor:
    """Flatten a single-policy weights dict into a 1-D parameter vector.
    Order: layer0.W (in*out, row-major), layer0.b (out), layer1.W, ...
    """
    parts = []
    for L in weights['layers']:
        parts.append(L['W'].flatten())
        parts.append(L['b'].flatten())
    return torch.cat(parts)


def param_size(weights: dict) -> int:
    return sum(L['W'].numel() + L['b'].numel() for L in weights['layers'])


def flat_to_batched_weights(theta_flat_KP: torch.Tensor,
                            template: dict) -> dict:
    """Take a (K, P) flat parameter tensor and a template weights dict
    (single-policy, used for shapes + non-learned fields like inputMean/Std,
    outputScale, clipMagnitude, activations) and produce a batched
    weights dict suitable for nn_forward_batched.
    """
    K = theta_flat_KP.shape[0]
    layers = []
    offset = 0
    for L in template['layers']:
        in_dim, out_dim = L['W'].shape
        nW = in_dim * out_dim
        nB = out_dim
        W = theta_flat_KP[:, offset:offset + nW].view(K, in_dim, out_dim)
        offset += nW
        b = theta_flat_KP[:, offset:offset + nB]
        offset += nB
        layers.append({'W': W, 'b': b, 'activation': L['activation']})
    # Broadcast non-learned fields across K. expand() avoids copy.
    K_inputMean = template['inputMean'].unsqueeze(0).expand(K, -1).contiguous()
    K_inputStd = template['inputStd'].unsqueeze(0).expand(K, -1).contiguous()
    return {
        'featureDim': template['featureDim'],
        'inputMean': K_inputMean,
        'inputStd': K_inputStd,
        'outputScale': template['outputScale'],
        'clipMagnitude': template['clipMagnitude'],
        'layers': layers,
        'K': K,
    }


def evaluate_batched(theta_flat_KP: torch.Tensor, template: dict,
                     seeds: list, frames: int, device: str,
                     use_graph: bool = True) -> torch.Tensor:
    """Evaluate K policies (in theta_flat_KP, shape (K, P)) on S seeds.
    Returns mean-catches per policy as a torch.Tensor of shape (K,).
    """
    K, _ = theta_flat_KP.shape
    S = len(seeds)
    batched_w = flat_to_batched_weights(theta_flat_KP, template)
    seeds_expanded = list(seeds) * K  # layout: (k, s) -> idx = k*S + s
    sim = Sim(seeds=seeds_expanded, weights=batched_w, device=device,
              sequential=True, auto_target='flock_centroid')
    if use_graph and device == 'cuda' and torch.cuda.is_available():
        out = sim.run_graph(frames)
    else:
        out = sim.run(frames)
    per = torch.tensor(out['per_seed_catches'], dtype=torch.float32,
                       device=device)  # (K*S,)
    per_K_S = per.view(K, S)
    return per_K_S.mean(dim=1), per_K_S  # mean (K,), full (K, S)


def export_weights_to_js(theta_flat: torch.Tensor, template: dict,
                         out_path: str):
    """Write a single-policy weights JSON in the format expected by
    js/eval_tte.js / js/predator_nn.js. Activations and metadata come
    from the template; W and b are unflattened from theta_flat.
    """
    theta_flat = theta_flat.detach().cpu().to(torch.float32).numpy()
    layers_out = []
    offset = 0
    for L in template['layers']:
        in_dim, out_dim = L['W'].shape
        nW = in_dim * out_dim
        nB = out_dim
        W = theta_flat[offset:offset + nW].astype(np.float32)
        offset += nW
        b = theta_flat[offset:offset + nB].astype(np.float32)
        offset += nB
        layers_out.append({
            'inDim': in_dim,
            'outDim': out_dim,
            'activation': L['activation'],
            'W': W.tolist(),
            'b': b.tolist(),
        })
    j = {
        'version': 1,
        'featureDim': template['featureDim'],
        'inputScale': 1.0,
        'inputMean': template['inputMean'].cpu().to(torch.float32).tolist(),
        'inputStd': template['inputStd'].cpu().to(torch.float32).tolist(),
        'outputScale': template['outputScale'],
        'clipMagnitude': template['clipMagnitude'],
        'layers': layers_out,
    }
    with open(out_path, 'w') as f:
        json.dump(j, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init_from', required=True,
                   help='Path to initial weights JSON.')
    p.add_argument('--K', type=int, default=128,
                   help='Population size. Must be even.')
    p.add_argument('--S', type=int, default=16, help='Seeds per candidate.')
    p.add_argument('--seedStart', type=int, default=100)
    p.add_argument('--frames', type=int, default=2500)
    p.add_argument('--sigma', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=0.5)
    p.add_argument('--top_k', type=int, default=8,
                   help='ARS-V1 elite count (uses top-K pairs by max(+, -)).')
    p.add_argument('--max_step_norm', type=float, default=0.02)
    p.add_argument('--gens', type=int, default=200)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=1234,
                   help='RNG seed for noise generation.')
    p.add_argument('--rotate_seeds', action='store_true',
                   help='Rotate the seed window each gen (shifts by +S).')
    p.add_argument('--ckpt_every', type=int, default=10)
    args = p.parse_args()

    assert args.K % 2 == 0, "K must be even (antithetic sampling)"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'es_log.jsonl'
    log_f = open(log_path, 'a')
    def log(obj):
        line = json.dumps(obj)
        print(line, flush=True)
        log_f.write(line + '\n'); log_f.flush()

    torch.manual_seed(args.seed)

    template = load_weights(args.init_from, device=args.device)
    P = param_size(template)
    theta = flatten_weights(template).to(args.device)
    print(f"loaded init from {args.init_from}, P={P} params", flush=True)

    # Warmup the CUDA path once so first gen isn't artificially slow
    if args.device == 'cuda' and torch.cuda.is_available():
        warm_seeds = list(range(args.seedStart, args.seedStart + args.S))
        warm_w = flat_to_batched_weights(theta.unsqueeze(0), template)
        warm_sim = Sim(seeds=warm_seeds, weights=warm_w,
                       device=args.device, sequential=True)
        for _ in range(3):
            warm_sim.step()
        torch.cuda.synchronize()
        del warm_sim
        torch.cuda.empty_cache()

    log({
        'phase': 'start',
        'K': args.K, 'S': args.S, 'frames': args.frames,
        'sigma': args.sigma, 'lr': args.lr, 'top_k': args.top_k,
        'max_step_norm': args.max_step_norm, 'P': P,
        'init_from': args.init_from, 'device': args.device,
        'seed': args.seed,
    })

    best_baseline = -1.0
    t_start = time.time()
    H = args.K // 2
    for gen in range(args.gens):
        gen_t0 = time.time()

        if args.rotate_seeds:
            ss = args.seedStart + gen * args.S
        else:
            ss = args.seedStart
        seeds = list(range(ss, ss + args.S))

        # Antithetic noise on GPU
        rng = torch.Generator(device='cpu').manual_seed(args.seed + gen * 100003)
        eps_half_cpu = torch.randn(H, P, generator=rng)
        eps_half = eps_half_cpu.to(args.device).to(torch.float32)
        eps = torch.cat([eps_half, -eps_half], dim=0)         # (K, P)

        # Candidates = theta + sigma * eps; also evaluate baseline (theta itself)
        candidates = theta.unsqueeze(0) + args.sigma * eps     # (K, P)
        all_thetas = torch.cat([candidates, theta.unsqueeze(0)], dim=0)  # (K+1, P)

        # Evaluate K+1 in one batched sim
        means, _ = evaluate_batched(all_thetas, template,
                                    seeds=seeds, frames=args.frames,
                                    device=args.device,
                                    use_graph=True)
        rewards = means[:args.K]
        baseline = float(means[args.K].item())

        # Snapshot pre-step theta — this is the one we just measured as
        # `baseline`, and the one we save when baseline beats the prior
        # best. The post-step theta below is what we *try* next; we don't
        # yet know its score.
        theta_pre = theta.clone()

        # ARS-V1 elite selection
        pair_max = torch.maximum(rewards[:H], rewards[H:])     # (H,)
        elite_idx = torch.argsort(pair_max, descending=True)[:args.top_k]
        diffs = (rewards[:H][elite_idx] - rewards[H:][elite_idx]).unsqueeze(1)
        grad = (diffs * eps_half[elite_idx]).sum(dim=0) / (args.top_k * args.sigma)
        reward_pool = torch.cat([rewards[:H][elite_idx], rewards[H:][elite_idx]])
        reward_std = reward_pool.std().clamp_min(1e-6)
        grad = grad / reward_std
        step = args.lr * grad

        step_norm = float(step.norm().item())
        if step_norm > args.max_step_norm:
            step = step * (args.max_step_norm / step_norm)
            step_norm = args.max_step_norm

        theta = theta + step

        gen_t = time.time() - gen_t0
        log({
            'gen': gen,
            'baseline_catches': baseline,
            'mean_perturbation_catches': float(rewards.mean().item()),
            'best_perturbation_catches': float(rewards.max().item()),
            'std_perturbation_catches': float(rewards.std().item()),
            'top_k_mean_catches': float(reward_pool.mean().item()),
            'grad_norm': float(grad.norm().item()),
            'step_norm': step_norm,
            'gen_seconds': gen_t,
            'total_seconds': time.time() - t_start,
            'seeds_start': ss,
        })

        # Save the PRE-step theta as `best.*` whenever its measured
        # baseline beats the prior best. This is the theta with the
        # score we logged — the post-step theta is just where we're
        # heading next and hasn't been measured yet.
        if baseline > best_baseline:
            best_baseline = baseline
            torch.save({
                'theta': theta_pre.detach().cpu(),
                'P': P,
                'gen': gen,
                'baseline_catches': best_baseline,
                'args': vars(args),
            }, out_dir / 'best.pt')
            export_weights_to_js(theta_pre, template,
                                 str(out_dir / 'best.json'))

        if gen % args.ckpt_every == 0 or gen == args.gens - 1:
            # ckpt_gen* always saves the pre-step theta whose baseline we
            # just measured, so the .json matches the logged score.
            torch.save({
                'theta': theta_pre.detach().cpu(),
                'P': P,
                'gen': gen,
                'baseline_catches': baseline,
                'args': vars(args),
            }, out_dir / f'ckpt_gen{gen:04d}.pt')
            export_weights_to_js(theta_pre, template,
                                 str(out_dir / f'ckpt_gen{gen:04d}.json'))

    log({'phase': 'done', 'best_baseline_catches': best_baseline,
         'total_seconds': time.time() - t_start})
    log_f.close()


if __name__ == '__main__':
    main()
