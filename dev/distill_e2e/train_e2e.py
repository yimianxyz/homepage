"""M2 — supervised training of an E2ENet to regress production steering force.

Loss = MSE on the 2-vec force. We also report angular error (deg) and the in-range
(chase) vs out-of-range (patrol) MSE split, since direction in each regime is what
drives behaviour. Saves net_<tag>.pt (state_dict + arch + standardizer).

Run:
  python3 train_e2e.py --hidden 16 --epochs 40 --device cuda --tag h16
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_net import E2ENet


def angular_err(pred, tgt):
    pn = F.normalize(pred, dim=1); tn = F.normalize(tgt, dim=1)
    cos = (pn * tn).sum(1).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def evaluate(net, obs, force, d1):
    net.eval()
    with torch.no_grad():
        pred = net(obs)
        mse = F.mse_loss(pred, force).item()
        ang = angular_err(pred, force)
        inr = (torch.isfinite(d1) & (d1 < 80.0))
        out = ~inr
        return dict(mse=mse, ang_med=ang.median().item(), ang_mean=ang.mean().item(),
                    mse_chase=F.mse_loss(pred[inr], force[inr]).item() if inr.any() else float('nan'),
                    mse_patrol=F.mse_loss(pred[out], force[out]).item() if out.any() else float('nan'),
                    ang_chase=ang[inr].median().item() if inr.any() else float('nan'),
                    ang_patrol=ang[out].median().item() if out.any() else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='dataset_train.pt')
    ap.add_argument('--val', default='dataset_val.pt')
    ap.add_argument('--hidden', default='16', help='comma list, empty = linear')
    ap.add_argument('--act', default='relu')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='h16')
    ap.add_argument('--outdir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    dev = args.device
    tr = torch.load(args.train, map_location=dev)
    va = torch.load(args.val, map_location=dev)
    Xtr, Ytr, Dtr = tr['obs'].to(dev), tr['force'].to(dev), tr['d1'].to(dev)
    Xva, Yva, Dva = va['obs'].to(dev), va['force'].to(dev), va['d1'].to(dev)
    in_dim = Xtr.shape[1]
    hidden = tuple(int(h) for h in args.hidden.split(',') if h.strip() != '')

    net = E2ENet(in_dim, hidden=hidden, act=args.act).to(dev)
    net.set_standardizer(Xtr)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    n = Xtr.shape[0]
    t0 = time.time()
    best = None
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            opt.zero_grad()
            loss = F.mse_loss(net(Xtr[idx]), Ytr[idx])
            loss.backward(); opt.step()
        sched.step()
        if not args.quiet and (ep % 5 == 0 or ep == args.epochs - 1):
            ev = evaluate(net, Xva, Yva, Dva)
            print(f"ep{ep:3d} val_mse={ev['mse']:.3e} ang_med={ev['ang_med']:.2f} "
                  f"chase={ev['ang_chase']:.2f} patrol={ev['ang_patrol']:.2f}")
    ev = evaluate(net, Xva, Yva, Dva)
    meta = dict(hidden=list(hidden), act=args.act, in_dim=in_dim,
                n_params=net.n_params(), epochs=args.epochs, train_seconds=round(time.time() - t0, 1),
                val=ev)
    path = os.path.join(args.outdir, f'net_{args.tag}.pt')
    torch.save(dict(state_dict=net.state_dict(), in_dim=in_dim, hidden=list(hidden),
                    act=args.act, meta=meta), path)
    print(json.dumps(meta, indent=2))
    print(f"# wrote {path}")


if __name__ == '__main__':
    main()
