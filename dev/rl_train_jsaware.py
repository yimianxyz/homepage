"""JS-aware ES: same batched sim_torch ES inner loop, but every N gens
JS-verify the current central theta and track the JS-best one as
`best.json`. Optionally roll back theta to the JS-best if recent JS
trends degrade.

Motivation: sim_torch ES has ρ ≈ 0.55 with JS. After 20 gens of pure
sim_torch ES on the v6 init, central theta JS went 22.6 (init) → 23.19
(gen 10) → 22.69 (gen 20) — there's a real signal early, then it gets
lost in sim_torch noise. JS-aware selection captures that signal.

Run remote, with JS eval done via subprocess (node dev/eval_tte.js).
Each JS verify is ~7 min on 4 workers, so a JS check every 5 gens
costs ~1.4 min/gen amortized — small fraction of the ~4 min sim_torch
gen, but it gives the real selection signal.
"""
import argparse
import json
import os
import subprocess
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


def run_js_eval(weights_path: str, seeds: int = 16, frames: int = 5000,
                workers: int = 4, auto_target: str = 'flock_centroid') -> float:
    """Run node dev/eval_tte.js on the given weights JSON and return
    meanCatches. Returns -inf on failure."""
    out_report = weights_path.replace('.json', '_jseval.json')
    eval_script = Path(__file__).resolve().parent / 'eval_tte.js'
    cmd = [
        'node', str(eval_script), weights_path,
        '--seeds', str(seeds),
        '--seedStart', '100',
        '--maxFrames', str(frames),
        '--workers', str(workers),
        '--autoTarget', auto_target,
        '--report', out_report,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=1800,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out_report) as f:
            return float(json.load(f)['meanCatches'])
    except Exception as e:
        print(f"  [JS eval failed] {e}", flush=True)
        return float('-inf')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init_from', required=True)
    p.add_argument('--K', type=int, default=128)
    p.add_argument('--S', type=int, default=16)
    p.add_argument('--seedStart', type=int, default=100)
    p.add_argument('--frames', type=int, default=1500)
    p.add_argument('--sigma', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=0.5)
    p.add_argument('--top_k', type=int, default=8)
    p.add_argument('--max_step_norm', type=float, default=0.02)
    p.add_argument('--gens', type=int, default=200)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--ckpt_every', type=int, default=5)
    p.add_argument('--js_every', type=int, default=5,
                   help='How often to JS-verify central theta.')
    p.add_argument('--js_seeds', type=int, default=16)
    p.add_argument('--js_frames', type=int, default=5000)
    p.add_argument('--rollback_on_regression', action='store_true',
                   help='If recent JS check is below best_js by threshold, '
                        'reset theta to best_js_theta.')
    p.add_argument('--rollback_threshold', type=float, default=2.0)
    args = p.parse_args()
    assert args.K % 2 == 0

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'jsaware_log.jsonl'
    log_f = open(log_path, 'a')
    def log(o):
        line = json.dumps(o)
        print(line, flush=True); log_f.write(line + '\n'); log_f.flush()

    torch.manual_seed(args.seed)
    template = load_weights(args.init_from, device=args.device)
    P = param_size(template)
    theta = flatten_weights(template).to(args.device)

    # Initial JS check
    init_js_path = str(out_dir / 'init.json')
    export_weights_to_js(theta, template, init_js_path)
    print("running initial JS eval...", flush=True)
    js0 = run_js_eval(init_js_path, args.js_seeds, args.js_frames)
    print(f"  init JS = {js0:.3f}", flush=True)
    best_js_score = js0
    best_js_theta = theta.clone()
    # Save initial as best.json
    export_weights_to_js(best_js_theta, template, str(out_dir / 'best.json'))
    torch.save({'theta': best_js_theta.cpu(), 'P': P, 'gen': -1,
                'js_score': best_js_score, 'args': vars(args)},
               out_dir / 'best.pt')

    # CUDA warmup
    if args.device == 'cuda' and torch.cuda.is_available():
        warm_seeds = list(range(args.seedStart, args.seedStart + args.S))
        ww = flat_to_batched_weights(theta.unsqueeze(0), template)
        ws = Sim(seeds=warm_seeds, weights=ww, device=args.device, sequential=True)
        for _ in range(3): ws.step()
        torch.cuda.synchronize(); del ws; torch.cuda.empty_cache()

    log({'phase': 'start', 'algo': 'jsaware', 'init_js': best_js_score,
         'K': args.K, 'S': args.S, 'frames': args.frames,
         'sigma': args.sigma, 'lr': args.lr, 'top_k': args.top_k,
         'js_every': args.js_every, 'P': P, 'init_from': args.init_from,
         'rollback': args.rollback_on_regression})

    H = args.K // 2
    t_start = time.time()
    last_js_score = best_js_score
    for gen in range(args.gens):
        gen_t0 = time.time()
        seeds = list(range(args.seedStart, args.seedStart + args.S))

        # ES step (same as rl_train_v2 ARS-V1)
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

        pair_max = torch.maximum(rewards[:H], rewards[H:])
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
        theta_pre = theta.clone()
        theta = theta + step

        # Optionally JS-verify the new central theta
        js_score = None
        if gen % args.js_every == 0:
            tmp_path = str(out_dir / f'gen{gen:04d}.json')
            export_weights_to_js(theta, template, tmp_path)
            js_score = run_js_eval(tmp_path, args.js_seeds, args.js_frames)
            last_js_score = js_score
            if js_score > best_js_score:
                best_js_score = js_score
                best_js_theta = theta.clone()
                export_weights_to_js(best_js_theta, template,
                                     str(out_dir / 'best.json'))
                torch.save({'theta': best_js_theta.cpu(), 'P': P, 'gen': gen,
                            'js_score': best_js_score, 'sim_baseline': baseline,
                            'args': vars(args)},
                           out_dir / 'best.pt')
            elif args.rollback_on_regression and \
                 js_score < best_js_score - args.rollback_threshold:
                print(f"  [rollback] JS {js_score:.2f} << best {best_js_score:.2f},"
                      f" reverting theta", flush=True)
                theta = best_js_theta.clone()
                last_js_score = best_js_score

        gen_t = time.time() - gen_t0
        log({
            'gen': gen, 'baseline_catches': baseline,
            'mean_pert_catches': float(rewards.mean().item()),
            'best_pert_catches': float(rewards.max().item()),
            'js_score': js_score, 'last_js_score': last_js_score,
            'best_js_score': best_js_score,
            'step_norm': step_norm, 'grad_norm': float(grad.norm().item()),
            'gen_seconds': gen_t, 'total_seconds': time.time() - t_start,
        })

        if gen % args.ckpt_every == 0 or gen == args.gens - 1:
            torch.save({'theta': theta_pre.cpu(), 'P': P, 'gen': gen,
                        'baseline_catches': baseline, 'args': vars(args)},
                       out_dir / f'ckpt_gen{gen:04d}.pt')
            export_weights_to_js(theta_pre, template,
                                 str(out_dir / f'ckpt_gen{gen:04d}.json'))

    log({'phase': 'done', 'best_js_score': best_js_score,
         'total_seconds': time.time() - t_start})
    log_f.close()


if __name__ == '__main__':
    main()
