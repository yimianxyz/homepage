#!/usr/bin/env python3
"""Train a drop-in value_net.json that predicts per-candidate rollout-gain, so the
cheap policy's prune (and scoring prior) pick better candidates. Data = log_cands.js
JSONL rows {f:[K][19], c:[4], g:[K]}. Net = 23->48->48->1, GELU hidden, linear out —
identical arch/format to the shipped value_net.json (so it's a drop-in replacement).

  python3 dev/train_prune_net.py --data /tmp/cands_all.jsonl --out js/value_net_distilled.json --epochs 60
"""
import json, argparse, math
import numpy as np
import torch
import torch.nn as nn


def load(paths):
    F, C, G = [], [], []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            f = np.array(r['f'], dtype=np.float32)        # (K,19)
            c = np.array(r['c'], dtype=np.float32)        # (4,)
            g = np.array(r['g'], dtype=np.float32)        # (K,)
            K = f.shape[0]
            x = np.concatenate([f, np.tile(c, (K, 1))], axis=1)   # (K,23)
            F.append(x); G.append(g)
    X = np.concatenate(F, axis=0)        # (N,23)
    Y = np.concatenate(G, axis=0)        # (N,)
    return X, Y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(23, 48)
        self.l1 = nn.Linear(48, 48)
        self.l2 = nn.Linear(48, 1)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.l0(x))
        x = torch.nn.functional.gelu(self.l1(x))
        return self.l2(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'

    X, Y = load(args.data)
    print(f'samples: {X.shape[0]}  feat-dim {X.shape[1]}')
    # standardize: 19 feats (fmu/fsd) + 4 ctx (xmu/xsd)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    fmu, fsd = mu[:19], sd[:19]
    xmu, xsd = mu[19:], sd[19:]
    Xs = (X - mu) / sd
    Xt = torch.tensor(Xs, device=dev); Yt = torch.tensor(Y, device=dev)
    n = Xt.shape[0]; idx = torch.randperm(n)
    ntr = int(n * 0.9)
    tr, va = idx[:ntr], idx[ntr:]

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    bs = 4096
    for ep in range(args.epochs):
        net.train(); perm = tr[torch.randperm(tr.shape[0])]
        for i in range(0, perm.shape[0], bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xt[b]); loss = lossf(out, Yt[b])
            loss.backward(); opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                vl = lossf(net(Xt[va]), Yt[va]).item()
            print(f'ep {ep}  val_mse {vl:.4f}')

    # export to value_net.json format (w as [out][in])
    net.eval()
    sd_ = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    out = {
        'fc': 19, 'fctx': 4, 'hidden': 48, 'depth': 2,
        'fmu': fmu.tolist(), 'fsd': fsd.tolist(),
        'xmu': xmu.tolist(), 'xsd': xsd.tolist(),
        'layers': [
            {'activation': 'gelu', 'b': sd_['l0.bias'].tolist(), 'w': sd_['l0.weight'].tolist()},
            {'activation': 'gelu', 'b': sd_['l1.bias'].tolist(), 'w': sd_['l1.weight'].tolist()},
            {'activation': 'linear', 'b': sd_['l2.bias'].tolist(), 'w': sd_['l2.weight'].tolist()},
        ],
    }
    json.dump(out, open(args.out, 'w'))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
