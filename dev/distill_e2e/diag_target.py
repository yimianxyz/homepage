"""Diagnostic: is the regression hard because the FORCE is ill-conditioned
(Reynolds: desired_vel - cur_vel), or because the raw grid can't express the
patrol target at all?

(A) baseline alignment: angle between production force and unit-direction-to
    pred_auto (the patrol target), split chase/patrol. If force is well-aligned
    with dir-to-target on patrol, then predicting the TARGET + a fixed Reynolds
    head reproduces patrol behaviour.
(B) learnability: train obs -> unit-dir-to-target (cosine loss); report patrol
    angular error. Low => grid CAN express the target, and the earlier ~60° wall
    was force ill-conditioning, not encoding.

  python3 diag_target.py --device cuda
"""
import argparse, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_net import E2ENet

HALF = 840.0


def torus_dir(auto, ppos):
    d = auto - ppos
    d = (d + HALF) % (2 * HALF) - HALF
    return F.normalize(d, dim=1)


def ang(a, b):
    c = (F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(1).clamp(-1, 1)
    return torch.rad2deg(torch.acos(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='dataset_train.pt')
    ap.add_argument('--val', default='dataset_val.pt')
    ap.add_argument('--hidden', default='64')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device
    tr = torch.load(args.train, map_location=dev)
    va = torch.load(args.val, map_location=dev)

    def dirlab(d):
        return torus_dir(d['auto'].to(dev), d['ppos'].to(dev))
    Ytr_dir, Yva_dir = dirlab(tr), dirlab(va)
    Xtr, Xva = tr['obs'].to(dev), va['obs'].to(dev)
    Dva = va['d1'].to(dev)
    inr = torch.isfinite(Dva) & (Dva < 80.0)

    # (A) how aligned is the production force with dir-to-target?
    fa = ang(va['force'].to(dev), Yva_dir)
    print("(A) angle(production_force, dir->auto):")
    print(f"    patrol med={fa[~inr].median():.1f}  chase med={fa[inr].median():.1f}")

    # (B) learn obs -> dir-to-target
    hidden = tuple(int(h) for h in args.hidden.split(',') if h.strip())
    net = E2ENet(Xtr.shape[1], hidden=hidden).to(dev)
    net.set_standardizer(Xtr)
    # drop the magnitude clip for this diagnostic: predict a raw direction vector
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = Xtr.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad()
            pred = net.net((Xtr[idx] - net.mean) / net.std)
            cos = (F.normalize(pred, dim=1) * Ytr_dir[idx]).sum(1)
            (1 - cos).mean().backward(); opt.step()
        sch.step()
    with torch.no_grad():
        pred = net.net((Xva - net.mean) / net.std)
    e = ang(pred, Yva_dir)
    print(f"(B) learn obs->dir(auto) [{args.hidden}]: patrol med={e[~inr].median():.1f} "
          f"chase med={e[inr].median():.1f}  overall med={e.median():.1f}")


if __name__ == '__main__':
    main()
