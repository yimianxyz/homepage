"""Supervised distillation of a big MLP teacher against the analytical rule.

The rl_train_gpu.py ES starting from random init is slow because the
gradient signal from sparse rewards is weak when the policy is bad. So
we warm-start the teacher to rule-level performance (which catches ~22
in sim_torch under flock_centroid). ES then has a *useful* signal: any
direction that nudges the policy toward better-than-rule behaviour.

The teacher's architecture matches rl_train_gpu.TeacherNN exactly so
its checkpoint can be loaded as --init_from.

Usage:
    python3 dev/train_teacher_supervised.py \
        --dataset dev/dataset_v3.bin --arch H=32 --feature_dim 35 \
        --epochs 80 --lr 3e-3 --batch 256 --device cuda \
        --out dev/checkpoints/supervised_H32_init.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


PREDATOR_MAX_FORCE = 0.05


def load_dataset(bin_path):
    bp = Path(bin_path)
    mp = Path(bin_path).with_suffix('.meta.json')
    with open(mp) as f:
        meta = json.load(f)
    fd = meta['featureDim']
    od = meta['outputDim']
    row = fd + od
    arr = np.fromfile(bp, dtype=np.float32).reshape(-1, row)
    print(f"loaded {arr.shape[0]} rows, featureDim={fd}, outputDim={od}")
    return arr, fd, od


class TeacherMLP(torch.nn.Module):
    """Same architecture as rl_train_gpu.TeacherNN: ReLU hidden + tanh output * MAX_FORCE."""
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
        return torch.tanh(self.layers(x)) * PREDATOR_MAX_FORCE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='dev/dataset_v3.bin')
    p.add_argument('--arch', default='H=32')
    p.add_argument('--feature_dim', type=int, default=35,
                   help='Should match the dataset featureDim.')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arr, fd, od = load_dataset(args.dataset)
    if fd != args.feature_dim:
        print(f"WARN: dataset featureDim={fd} but --feature_dim={args.feature_dim}; using dataset's")
        args.feature_dim = fd

    hidden_dims = [int(x) for x in args.arch.replace('H=', '').split(',') if x]
    print(f"arch: feature_dim={args.feature_dim}, hidden={hidden_dims}")

    # Train/val split
    n = arr.shape[0]
    perm = np.random.permutation(n)
    val_n = int(n * args.val_frac)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    X_train = torch.from_numpy(arr[train_idx, :fd]).to(args.device)
    Y_train = torch.from_numpy(arr[train_idx, fd:]).to(args.device)
    X_val = torch.from_numpy(arr[val_idx, :fd]).to(args.device)
    Y_val = torch.from_numpy(arr[val_idx, fd:]).to(args.device)

    model = TeacherMLP(args.feature_dim, hidden_dims).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    best_val = float('inf')
    best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        # Shuffle
        order = torch.randperm(len(train_idx), device=args.device)
        running_loss = 0
        n_batches = 0
        for off in range(0, len(order), args.batch):
            ids = order[off:off + args.batch]
            xb = X_train[ids]
            yb = Y_train[ids]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / n_batches
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, Y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"ep={ep:3d} train_loss={train_loss:.3e} val_loss={val_loss:.3e} best_val={best_val:.3e}")
    print(f"Done. Best val MSE: {best_val:.3e}  Time: {time.time()-t0:.1f}s")

    # Load best state, extract flat theta in rl_train_gpu.TeacherNN format
    model.load_state_dict(best_state)
    flat_pieces = []
    layer_count = 0
    for mod in model.layers:
        if isinstance(mod, torch.nn.Linear):
            # rl_train_gpu uses (in, out) for W -> matches Linear's weight transposed.
            # PyTorch nn.Linear stores weight as (out, in); we transpose.
            W = mod.weight.detach().T.contiguous().flatten()
            b = mod.bias.detach().clone().flatten()
            flat_pieces.append(W)
            flat_pieces.append(b)
            layer_count += 1
    theta = torch.cat(flat_pieces).cpu()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'theta': theta,
        'feature_dim': args.feature_dim,
        'hidden_dims': hidden_dims,
        'arch': args.arch,
        'supervised_val_mse': best_val,
        'source': 'supervised distillation from ' + args.dataset,
    }, args.out)
    print(f"Saved {args.out}  (theta size {theta.numel()}, val_mse {best_val:.3e})")


if __name__ == '__main__':
    main()
