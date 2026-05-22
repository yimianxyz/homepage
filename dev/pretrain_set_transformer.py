"""Supervised pretraining of Set Transformer policy against the rule.

Rolls out the rule policy in sim_torch, recording per-boid features and
the rule's action each frame, then trains the SetTransformerPolicy to
match. After pretrain, use as ES init via --init_from.

Usage:
    python3 dev/pretrain_set_transformer.py \
        --d_model 32 --num_inducing 8 --num_heads 4 \
        --rollout_seeds 32 --rollout_frames 2000 \
        --epochs 50 --lr 1e-3 --batch 256 \
        --device cuda --out dev/checkpoints/setxf_supervised.pt
"""
import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'dev')
from sim_torch import (
    Sim, mulberry32_seq, fast_set_magnitude, fast_limit,
    PREDATOR_RANGE, PREDATOR_MAX_SPEED, PREDATOR_MAX_FORCE,
    N_BOIDS, CANVAS_W, CANVAS_H, FRAME_MS, EPSILON,
    build_features as build_features_rule,
)
from rl_train_set_transformer import (
    SetSim, build_set_features, flatten_params, set_flat_params,
)
from set_transformer import SetTransformerPolicy, num_params


def collect_rule_rollouts(rollout_seeds, rollout_frames, device):
    """Roll out rule policy in sim_torch, record per-boid features and
    rule actions every frame. Returns (boid_feats, boid_mask, pred_state,
    rule_action) as torch tensors stacked across all frames+seeds."""
    from sim_torch import nn_forward as nn_forward_rule  # not used; just rule
    from policy_spec_py import rule_policy_torch  # we'll implement
    # Use the existing Sim class with rule (no nnFn) — but sim_torch's Sim
    # uses nnFn. Switch to plain rule.
    # Easiest: re-use the Sim class but pass a callable that computes
    # rule_policy from rule-style features.
    # Actually simpler: bypass Sim and re-implement minimal rollout.
    raise NotImplementedError("not used; see inline below")


class RuleSim(Sim):
    """Sim with rule_policy (analytical, from policy_spec_py port below)."""
    def __init__(self, seeds, num_boids=N_BOIDS, auto_target='flock_centroid', device='cpu'):
        self.weights = {'featureDim': 0}
        self.seeds = list(seeds)
        self.B = len(self.seeds)
        self.N = num_boids
        self.auto_target_mode = auto_target
        self.device = device
        self._initialize()

    def rule_action(self):
        """Compute rule's steering action for the current sim state, batched."""
        # Use the rule-style features (45-dim) to compute rule action.
        from sim_torch import build_features as build_features_rule
        feats = build_features_rule(
            self.pred_pos.float(), self.pred_vel.float(),
            self.boid_pos.float(), self.boid_vel.float(), self.boid_alive,
            self.pred_auto.float(), 45, self.device,
        )
        # Apply rule_policy from policy_spec: head to nearest boid if d1<R else seek_auto.
        vx = feats[:, 0]
        vy = feats[:, 1]
        dx1 = feats[:, 7]
        dy1 = feats[:, 8]
        d1 = feats[:, 23]
        dxA = feats[:, 2]
        dyA = feats[:, 3]
        POLICY_PAD = 2000.0
        in_range = (d1 < PREDATOR_RANGE) & (dx1 != POLICY_PAD)
        tx = torch.where(in_range, dx1, dxA)
        ty = torch.where(in_range, dy1, dyA)
        dx0, dy0 = fast_set_magnitude(tx, ty, PREDATOR_MAX_SPEED)
        sx = dx0 - vx
        sy = dy0 - vy
        sx, sy = fast_limit(sx, sy, PREDATOR_MAX_FORCE)
        return torch.stack([sx, sy], dim=-1).float()

    def _step_predator(self):
        self._update_auto_target()
        steering = self.rule_action().double()
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


def collect(rollout_seeds, rollout_frames, device):
    """Roll out rule and record per-boid features + rule action each frame.
    Returns (boid_feats, boid_masks, pred_states, rule_actions) flattened
    across batch and time as torch tensors on CPU."""
    sim = RuleSim(seeds=list(range(rollout_seeds)), device=device)
    B = sim.B
    N = sim.N
    boid_feats_list = []
    pred_state_list = []
    boid_mask_list = []
    action_list = []
    for f in range(rollout_frames):
        sim._update_auto_target()
        bf, ps, bm = build_set_features(
            sim.pred_pos.float(), sim.pred_vel.float(),
            sim.boid_pos.float(), sim.boid_vel.float(), sim.boid_alive,
            sim.pred_auto.float(), device,
        )
        action = sim.rule_action()
        boid_feats_list.append(bf.cpu())
        pred_state_list.append(ps.cpu())
        boid_mask_list.append(bm.cpu())
        action_list.append(action.cpu())
        # step the rest
        sim._step_predator()
        sim._step_boids()
        sim._check_catches()
        sim._decay_size()
        sim.frame += 1
    bf_all = torch.cat(boid_feats_list, dim=0)
    ps_all = torch.cat(pred_state_list, dim=0)
    bm_all = torch.cat(boid_mask_list, dim=0)
    act_all = torch.cat(action_list, dim=0)
    return bf_all, bm_all, ps_all, act_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d_model', type=int, default=32)
    p.add_argument('--num_inducing', type=int, default=8)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--rollout_seeds', type=int, default=32)
    p.add_argument('--rollout_frames', type=int, default=2000)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    print(f"Building rule rollouts: {args.rollout_seeds} seeds × {args.rollout_frames} frames")
    t0 = time.time()
    bf, bm, ps, act = collect(args.rollout_seeds, args.rollout_frames, args.device)
    elapsed = time.time() - t0
    print(f"Collected {bf.shape[0]} samples in {elapsed:.1f}s")
    # bf: (T, N, 5); bm: (T, N); ps: (T, 4); act: (T, 2)

    # Build model
    policy = SetTransformerPolicy(d_model=args.d_model,
                                  num_inducing=args.num_inducing,
                                  num_heads=args.num_heads).to(args.device)
    print(f"Set Transformer: {num_params(policy)} params")

    # Train/val split
    n = bf.shape[0]
    perm = torch.randperm(n)
    val_n = int(n * args.val_frac)
    val_idx = perm[:val_n].to(args.device)
    train_idx = perm[val_n:].to(args.device)
    bf = bf.to(args.device)
    bm = bm.to(args.device)
    ps = ps.to(args.device)
    act = act.to(args.device)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    best_state = None
    print(f"Train n={len(train_idx)}, Val n={len(val_idx)}")
    for ep in range(args.epochs):
        order = train_idx[torch.randperm(len(train_idx), device=args.device)]
        running = 0
        nb = 0
        for off in range(0, len(order), args.batch):
            ids = order[off:off+args.batch]
            policy.train()
            pred = policy(bf[ids], bm[ids], ps[ids])
            loss = loss_fn(pred, act[ids])
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); nb += 1
        with torch.no_grad():
            policy.eval()
            vp = policy(bf[val_idx], bm[val_idx], ps[val_idx])
            val_loss = loss_fn(vp, act[val_idx]).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep={ep:3d} train_loss={running/nb:.3e} val_loss={val_loss:.3e} best={best_val:.3e}")

    policy.load_state_dict(best_state)
    theta_flat = flatten_params(policy).cpu()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'theta': theta_flat,
        'arch': 'set_transformer',
        'd_model': args.d_model,
        'num_inducing': args.num_inducing,
        'num_heads': args.num_heads,
        'supervised_val_mse': best_val,
        'source': 'rule rollouts',
    }, args.out)
    print(f"Saved {args.out}  val_mse={best_val:.3e}, theta_size={theta_flat.numel()}")


if __name__ == '__main__':
    main()
