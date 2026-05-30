"""Milestone-0 feasibility experiments for distilling the production predator
policy into a single memoryless end-to-end NN (raw obs -> steering).

Two questions decide whether the goal is reachable at all:

  FEAS-1  Is the production policy effectively MEMORYLESS?
          The patrol target (pred_auto) FREEZES whenever a boid is in chase
          range and only updates otherwise (sim_torch.py _update_auto_target).
          That frozen target still feeds build_features during the chase, so
          the true policy is a function of (state, frozen_target) = history.
          A memoryless raw-obs NN can only reproduce f(current_state).
          Test: run an "always recompute target from current state" variant
          and compare per-seed catches to production. Identical => the freeze
          does not affect the outcome => the policy is effectively memoryless
          and the clean distillation target is
              f(state) = nn_forward(build_features(state, evolved_target(state)))

  FEAS-2  How exact must the NN force be for IDENTICAL per-seed catches?
          Perturb the production steering by Gaussian noise of std sigma and
          measure per-seed catch divergence vs sigma. This is the action-error
          budget epsilon* the trained NN must beat. (The system is chaotic:
          sim_torch and JS disagreed 8.25 vs 5.25 on the same policy, so the
          tolerance may be tiny.)

Run (CPU, small) :  python3 dev/distill_e2e/feas.py --seeds 16 --frames 1500
Run (GPU, large):  python3 dev/distill_e2e/feas.py --seeds 256 --frames 1500 --device cuda
"""
import argparse
import json
import sys
import time

import torch

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/workspace/dev')
from sim_torch import Sim, load_weights  # noqa: E402

# Shipped evolved patrol config (E3D) — the production policy on main.
E3D = dict(cluster_r=178.09, dens_pow=2.373, reach_scale=1515.0, sharp=9.25,
           lead_scale=0.454, lead_max=230.6, nbhd=0.461, momentum=0.0)


class NoisySim(Sim):
    """Production sim with additive Gaussian noise on the predator steering.
    Reproduces _step_predator exactly except for the injected noise, so
    sigma=0 is bit-identical to stock Sim."""

    def __init__(self, *a, noise_sigma=0.0, noise_seed=12345, **kw):
        super().__init__(*a, **kw)
        self.noise_sigma = float(noise_sigma)
        self._gen = torch.Generator(device=self.device if self.device != 'cpu' else 'cpu')
        self._gen.manual_seed(int(noise_seed))

    def _step_predator(self):
        self._update_auto_target()
        from sim_torch import build_features, nn_forward, nn_forward_batched, fast_limit, PREDATOR_MAX_SPEED
        feats = build_features(
            self.pred_pos.float(), self.pred_vel.float(),
            self.boid_pos.float(), self.boid_vel.float(), self.boid_alive,
            self.pred_auto.float(), self.weights['featureDim'], self.device,
            dtype=torch.float32,
        )
        if 'K' in self.weights:
            steering = nn_forward_batched(feats, self.weights).double()
        else:
            steering = nn_forward(feats, self.weights).double()
        if self.noise_sigma > 0:
            noise = torch.randn(steering.shape, generator=self._gen,
                                device=steering.device, dtype=steering.dtype) * self.noise_sigma
            steering = steering + noise
        new_vx = self.pred_vel[:, 0] + steering[:, 0]
        new_vy = self.pred_vel[:, 1] + steering[:, 1]
        new_vx, new_vy = fast_limit(new_vx, new_vy, PREDATOR_MAX_SPEED)
        self.pred_vel[:, 0] = new_vx
        self.pred_vel[:, 1] = new_vy
        self.pred_pos[:, 0] += new_vx
        self.pred_pos[:, 1] += new_vy
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] > self._wrap_w_max, self._wrap_neg20, self.pred_pos[:, 0])
        self.pred_pos[:, 0] = torch.where(self.pred_pos[:, 0] < self._wrap_neg20, self._wrap_w_max, self.pred_pos[:, 0])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] > self._wrap_h_max, self._wrap_neg20, self.pred_pos[:, 1])
        self.pred_pos[:, 1] = torch.where(self.pred_pos[:, 1] < self._wrap_neg20, self._wrap_h_max, self.pred_pos[:, 1])


def run_policy(seeds, weights, frames, device, always_recompute=False, noise_sigma=0.0):
    if noise_sigma > 0:
        sim = NoisySim(seeds=seeds, weights=weights, device=device,
                       auto_target='evolved', auto_target_opts=dict(E3D),
                       noise_sigma=noise_sigma)
    else:
        sim = Sim(seeds=seeds, weights=weights, device=device,
                  auto_target='evolved', auto_target_opts=dict(E3D))
    sim.always_recompute_target = bool(always_recompute)
    out = sim.run(frames)
    return out['per_seed_catches']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='/workspace/js/predator_weights.json')
    ap.add_argument('--seeds', type=int, default=16)
    ap.add_argument('--seedStart', type=int, default=50000)
    ap.add_argument('--frames', type=int, default=1500)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    seeds = list(range(args.seedStart, args.seedStart + args.seeds))
    weights = load_weights(args.weights, device=args.device)
    print(f"# feas: weights featureDim={weights['featureDim']} seeds={args.seeds} "
          f"frames={args.frames} device={args.device}")

    t0 = time.time()
    prod = run_policy(seeds, weights, args.frames, args.device)
    t_prod = time.time() - t0
    prod_t = torch.tensor(prod)
    print(f"# production rollout {t_prod:.1f}s  mean={prod_t.float().mean():.3f}")

    # FEAS-1: memorylessness
    recomp = run_policy(seeds, weights, args.frames, args.device, always_recompute=True)
    recomp_t = torch.tensor(recomp)
    n_match = int((recomp_t == prod_t).sum())
    print("\n=== FEAS-1: memorylessness (always-recompute target vs production) ===")
    print(f"per_seed production      : {prod}")
    print(f"per_seed always-recompute: {recomp}")
    print(f"exact per-seed match: {n_match}/{len(seeds)}  "
          f"mean_abs_diff={ (recomp_t-prod_t).abs().float().mean():.4f}  "
          f"mean prod={prod_t.float().mean():.3f} recomp={recomp_t.float().mean():.3f}")
    feas1_memoryless = (n_match == len(seeds))
    print(f"VERDICT: production is {'EFFECTIVELY MEMORYLESS' if feas1_memoryless else 'HISTORY-DEPENDENT'} "
          f"on this block")

    # FEAS-2: action->catch sensitivity
    print("\n=== FEAS-2: action->catch sensitivity (perturb steering by N(0,sigma)) ===")
    print(f"{'sigma':>10} {'exact_match':>12} {'mean_abs_diff':>14} {'mean_catches':>13}")
    sens = []
    for sigma in [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        per = run_policy(seeds, weights, args.frames, args.device, noise_sigma=sigma)
        pt = torch.tensor(per)
        nm = int((pt == prod_t).sum())
        mad = (pt - prod_t).abs().float().mean().item()
        sens.append(dict(sigma=sigma, exact_match=nm, mean_abs_diff=mad,
                         mean_catches=pt.float().mean().item()))
        print(f"{sigma:>10.0e} {nm:>9}/{len(seeds)} {mad:>14.4f} {pt.float().mean():>13.3f}")

    out = dict(seeds=seeds, frames=args.frames, device=args.device,
               production_per_seed=prod, recompute_per_seed=recomp,
               feas1_memoryless=feas1_memoryless, feas1_match=n_match,
               feas2_sensitivity=sens)
    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feas_result.json')
    with open(res_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n# wrote {res_path}")


if __name__ == '__main__':
    main()
