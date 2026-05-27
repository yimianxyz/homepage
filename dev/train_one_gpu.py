"""GPU-native NN trainer matching JS dev/train_one.js (Adam + MSE).

Loads the same .bin dataset, trains a feed-forward NN with PyTorch, and
saves the weights in the JS-compatible JSON format (so predator_nn.js can
load it without modification).

Speedup vs JS: typically 10-30x on L4 (JS ~5-7 min for H=8 80 epochs; this
~10-30 sec).

Usage:
    python3 dev/train_one_gpu.py \
        --arch '{"id":"K4_H8","layers":[{"units":8,"activation":"relu"},{"units":2,"activation":"linear"}]}' \
        --dataset datasets/rule_v3_smd_a5_80seeds_5000f.bin \
        --epochs 80 --batch 256 --lr 3e-3 --seed 1234 \
        --out weights/rule_v3_H8_gpu.json \
        --report reports/rule_v3_H8_gpu.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_dataset(bin_path):
    arr = np.fromfile(bin_path, dtype=np.float32)
    meta_path = str(bin_path).replace('.bin', '.meta.json')
    meta = json.load(open(meta_path))
    row = meta['rowFloats']
    n = arr.size // row
    arr = arr.reshape(n, row)
    return arr, meta, n


def build_model(arch, input_dim, seed, device, dtype):
    g = torch.Generator(device='cpu').manual_seed(seed)
    layers = []
    prev = input_dim
    for L in arch['layers']:
        in_dim, out_dim, act = prev, L['units'], L['activation']
        std = (2.0 / in_dim) ** 0.5 if act == 'relu' else (1.0 / in_dim) ** 0.5
        lin = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(out_dim, in_dim, generator=g, dtype=torch.float32) * std)
            lin.bias.zero_()
        layers.append(lin)
        prev = out_dim
    return layers


def forward(layers, x, activations):
    a = x
    for L, act in zip(layers, activations):
        z = L(a)
        if act == 'relu':
            a = torch.relu(z)
        elif act == 'tanh':
            a = torch.tanh(z)
        elif act == 'sigmoid':
            a = torch.sigmoid(z)
        else:
            a = z
    return a


def save_model(layers, activations, arch, args, mean, std, out_path):
    layers_json = []
    for L, act in zip(layers, activations):
        layers_json.append({
            'inDim': L.in_features,
            'outDim': L.out_features,
            'activation': act,
            # JS expects row-major [inDim*outDim] with W[i*outDim + j] = w(i,j)
            # PyTorch stores weight as (out_features, in_features).
            # We transpose to (in_features, out_features) then flatten C-order.
            'W': L.weight.detach().cpu().t().contiguous().flatten().tolist(),
            'b': L.bias.detach().cpu().tolist(),
        })
    out = {
        'version': 1,
        'id': arch['id'],
        'K': 4,  # POLICY_K — matches the JS spec
        'featureDim': len(mean),
        'inputMean': mean.tolist(),
        'inputStd': std.tolist(),
        'outputScale': args.outputScale,
        'clipMagnitude': 0.05,
        'layers': layers_json,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True)
    p.add_argument('--archFile', default=None)
    p.add_argument('--dataset', required=True)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--valFrac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--outputScale', type=float, default=0.05)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    p.add_argument('--report', default=None)
    a = p.parse_args()

    arch = json.loads(open(a.archFile).read()) if a.archFile else json.loads(a.arch)

    t_start = time.time()
    arr, meta, n = load_dataset(a.dataset)
    feat_dim = meta['featureDim']
    row = meta['rowFloats']
    print(json.dumps({'phase': 'loaded', 'n': n, 'featureDim': feat_dim}))

    X_all = arr[:, :feat_dim]
    Y_all = arr[:, feat_dim:feat_dim + 2]

    # input normalization (over the full dataset)
    mean = X_all.mean(axis=0)
    std = X_all.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    Xn = (X_all - mean) / std
    Yn = Y_all / a.outputScale

    rng = np.random.RandomState(a.seed)
    perm = rng.permutation(n)
    nv = int(n * a.valFrac)
    val_idx = perm[:nv]
    trn_idx = perm[nv:]

    device = a.device if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    X = torch.from_numpy(Xn.astype(np.float32)).to(device)
    Y = torch.from_numpy(Yn.astype(np.float32)).to(device)
    train_idx = torch.from_numpy(trn_idx.astype(np.int64)).to(device)
    val_idx_t = torch.from_numpy(val_idx.astype(np.int64)).to(device)

    activations = [L['activation'] for L in arch['layers']]
    layers = build_model(arch, feat_dim, a.seed, device, dtype)
    for L in layers:
        L.to(device).to(dtype)

    params = []
    for L in layers:
        params.extend(L.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)  # default β1=0.9, β2=0.999, eps=1e-8

    n_train = len(train_idx)
    batch = a.batch
    history = []
    best_val = float('inf')
    best_state = None
    best_epoch = 0

    for epoch in range(a.epochs):
        perm_e = train_idx[torch.randperm(n_train, device=device)]
        n_batches = n_train // batch
        train_loss_sum = 0.0
        for bi in range(n_batches):
            idx = perm_e[bi * batch:(bi + 1) * batch]
            xb = X[idx]
            yb = Y[idx]
            pred = forward(layers, xb, activations)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item()

        with torch.no_grad():
            xv = X[val_idx_t]
            yv = Y[val_idx_t]
            pv = forward(layers, xv, activations)
            val_loss = ((pv - yv) ** 2).mean().item()

        train_loss_avg = train_loss_sum / max(1, n_batches)
        history.append({'epoch': epoch, 'trainLoss': train_loss_avg, 'valLoss': val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = [(L.weight.detach().clone(), L.bias.detach().clone()) for L in layers]

    if best_state is not None:
        for L, (W, B) in zip(layers, best_state):
            with torch.no_grad():
                L.weight.copy_(W)
                L.bias.copy_(B)

    save_model(layers, activations, arch, a, mean, std, a.out)

    report = {
        'arch': arch,
        'dataset': a.dataset,
        'epochs': a.epochs,
        'batch': a.batch,
        'lr': a.lr,
        'seed': a.seed,
        'featureDim': feat_dim,
        'inputMean': mean.tolist(),
        'inputStd': std.tolist(),
        'outputScale': a.outputScale,
        'finalValLoss': history[-1]['valLoss'],
        'bestValLoss': best_val,
        'bestEpoch': best_epoch,
        'history': history,
        'weightsPath': a.out,
        'elapsedMs': int((time.time() - t_start) * 1000),
        'device': device,
    }
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({'phase': 'done', 'bestValLoss': best_val, 'bestEpoch': best_epoch,
                       'elapsedMs': report['elapsedMs']}))


if __name__ == '__main__':
    main()
