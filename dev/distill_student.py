"""Distill a trained teacher back into a small student NN matching the
shipped architecture (K4_H4_relu, featureDim=35).

Workflow:
  1. Load teacher .pt
  2. Generate (features, teacher_action) pairs by rolling out sim_torch
     with the teacher policy across many seeds
  3. Train a small student (H=4 relu, linear output + clip) supervised
     to match teacher's actions
  4. Save student weights as JS-compatible JSON (same format as shipped)

Usage:
    python3 dev/distill_student.py --teacher dev/checkpoints/best.pt \
        --seeds 256 --frames 5000 \
        --student-arch H=4 --epochs 80 \
        --out dev/weights/student_distilled.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, 'dev')
from rl_train_gpu import TeacherNN, TeacherSim
from sim_torch import build_features, fast_set_magnitude, fast_limit, PREDATOR_MAX_FORCE


PREDATOR_MAX_FORCE_VAL = 0.05


def collect_rollouts(teacher, theta, feature_dim, seeds, max_frames, device,
                     student_feature_dim=35):
    """Roll out sim with teacher, record (features, action) every frame.

    Returns: (X, Y) torch tensors on CPU. X has student's featureDim slice.
    """
    from sim_torch import Sim
    sim = TeacherSim(seeds=seeds, teacher=teacher, theta_flat=theta,
                     feature_dim=feature_dim, device=device,
                     auto_target='flock_centroid')
    # We want to record (features, action) per step per batch.
    # Total rows: max_frames * len(seeds)
    B = len(seeds)
    total = max_frames * B
    X = np.zeros((total, student_feature_dim), dtype=np.float32)
    Y = np.zeros((total, 2), dtype=np.float32)
    idx = 0
    for f in range(max_frames):
        sim._update_auto_target()
        feats = build_features(
            sim.pred_pos.float(), sim.pred_vel.float(),
            sim.boid_pos.float(), sim.boid_vel.float(), sim.boid_alive,
            sim.pred_auto.float(), feature_dim, device,
        )
        # Teacher action
        action = teacher.forward(feats, theta).double()
        # Slice features to student's dim (first student_feature_dim slots).
        # Use the full feature build for completeness then slice.
        feat_cpu = feats[:, :student_feature_dim].detach().cpu().numpy()
        act_cpu = action.detach().cpu().numpy()
        X[idx:idx + B] = feat_cpu
        Y[idx:idx + B] = act_cpu
        idx += B
        # Step the rest of the predator/boid logic (matches Sim.step)
        new_vx = sim.pred_vel[:, 0] + action[:, 0]
        new_vy = sim.pred_vel[:, 1] + action[:, 1]
        from sim_torch import PREDATOR_MAX_SPEED, CANVAS_W, CANVAS_H
        new_vx, new_vy = fast_limit(new_vx, new_vy, PREDATOR_MAX_SPEED)
        sim.pred_vel[:, 0] = new_vx
        sim.pred_vel[:, 1] = new_vy
        sim.pred_pos[:, 0] += new_vx
        sim.pred_pos[:, 1] += new_vy
        m = CANVAS_W + 20
        sim.pred_pos[:, 0] = torch.where(sim.pred_pos[:, 0] > m, torch.tensor(-20.0, dtype=torch.float64, device=device), sim.pred_pos[:, 0])
        sim.pred_pos[:, 0] = torch.where(sim.pred_pos[:, 0] < -20, torch.tensor(float(m), dtype=torch.float64, device=device), sim.pred_pos[:, 0])
        m = CANVAS_H + 20
        sim.pred_pos[:, 1] = torch.where(sim.pred_pos[:, 1] > m, torch.tensor(-20.0, dtype=torch.float64, device=device), sim.pred_pos[:, 1])
        sim.pred_pos[:, 1] = torch.where(sim.pred_pos[:, 1] < -20, torch.tensor(float(m), dtype=torch.float64, device=device), sim.pred_pos[:, 1])
        sim._step_boids()
        sim._check_catches()
        sim._decay_size()
        sim.frame += 1
    return X[:idx], Y[:idx]


class StudentMLP(torch.nn.Module):
    """Matches the shipped K4_H4_relu shape with linear output (no tanh).
    The shipped NN does `linear_out * outputScale; clip_magnitude` — so
    output is bounded but not squashed."""
    def __init__(self, feature_dim, hidden_dims):
        super().__init__()
        dims = [feature_dim] + list(hidden_dims) + [2]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', required=True)
    p.add_argument('--seeds', type=int, default=256)
    p.add_argument('--frames', type=int, default=5000)
    p.add_argument('--student_arch', default='H=4')
    p.add_argument('--student_feature_dim', type=int, default=35)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # Load teacher
    ckpt = torch.load(args.teacher, map_location=device)
    teacher = TeacherNN(ckpt['feature_dim'], ckpt['hidden_dims'], device=device)
    theta = ckpt['theta'].to(device).float()
    print(f"Loaded teacher: feature_dim={ckpt['feature_dim']}, hidden={ckpt['hidden_dims']}")

    # Collect rollouts
    seeds = list(range(args.seeds))
    print(f"Collecting rollouts: {len(seeds)} seeds × {args.frames} frames...")
    t0 = time.time()
    X, Y = collect_rollouts(teacher, theta, ckpt['feature_dim'], seeds,
                             args.frames, device, args.student_feature_dim)
    print(f"Collected {X.shape[0]} (features, action) pairs in {time.time()-t0:.1f}s")

    # Compute normalization stats from collected features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    X_norm = (X - mean) / std
    # Target: scale to ±1 range (since shipped uses outputScale=0.05)
    Y_scaled = Y / PREDATOR_MAX_FORCE_VAL

    X_t = torch.from_numpy(X_norm.astype(np.float32)).to(device)
    Y_t = torch.from_numpy(Y_scaled.astype(np.float32)).to(device)

    # Train/val split
    perm = np.random.permutation(X.shape[0])
    val_n = max(1024, int(0.1 * len(perm)))
    val_idx = torch.from_numpy(perm[:val_n].astype(np.int64)).to(device)
    train_idx = torch.from_numpy(perm[val_n:].astype(np.int64)).to(device)

    # Train student
    hidden_dims = [int(x) for x in args.student_arch.replace('H=', '').split(',') if x]
    student = StudentMLP(args.student_feature_dim, hidden_dims).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    print(f"Student arch: feature_dim={args.student_feature_dim}, hidden={hidden_dims}")

    best_val = float('inf')
    best_state = None
    for ep in range(args.epochs):
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        running_loss = 0
        n_batches = 0
        for off in range(0, len(order), args.batch):
            ids = order[off:off + args.batch]
            xb = X_t[ids]
            yb = Y_t[ids]
            opt.zero_grad()
            pred = student(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / n_batches
        with torch.no_grad():
            val_pred = student(X_t[val_idx])
            val_loss = loss_fn(val_pred, Y_t[val_idx]).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep={ep:3d} train_loss={train_loss:.3e} val_loss={val_loss:.3e} best_val={best_val:.3e}")

    student.load_state_dict(best_state)
    print(f"Best val MSE: {best_val:.3e}")

    # Export as JS predator_nn JSON
    layers_json = []
    layer_count = 0
    for i, mod in enumerate(student.layers):
        if isinstance(mod, torch.nn.Linear):
            in_d = mod.in_features
            out_d = mod.out_features
            W = mod.weight.detach().cpu().T.contiguous().flatten().tolist()
            b = mod.bias.detach().cpu().flatten().tolist()
            # Activation: relu on hidden, linear on output
            is_output = (i == len(list(student.layers)) - 1)
            act = 'linear' if is_output else 'relu'
            layers_json.append({'inDim': in_d, 'outDim': out_d, 'activation': act,
                                'W': W, 'b': b})
            layer_count += 1
    out = {
        'version': 1,
        'id': f"distilled_student_{args.student_arch.replace('=', '')}",
        'K': 4,
        'featureDim': args.student_feature_dim,
        'inputMean': mean.tolist(),
        'inputStd': std.tolist(),
        'outputScale': PREDATOR_MAX_FORCE_VAL,
        'clipMagnitude': PREDATOR_MAX_FORCE_VAL,
        'layers': layers_json,
        '_meta': {
            'teacher_ckpt': args.teacher,
            'val_mse': best_val,
            'seeds_used': args.seeds,
            'frames_per_seed': args.frames,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
