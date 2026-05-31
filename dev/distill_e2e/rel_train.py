"""Train a counting-capable RelNet on the patrol steering-force distillation task.

  python3 rel_train.py --train setds_densAnt32_train.pt --val setds_densAnt32_val.pt \
      --mode edge --d 64 --nblocks 2 --epochs 300 --bs 1024 --lr 2e-3 \
      --tag edge_d64 --device cuda

Selects the best checkpoint on PATROL cos_med (the hard regime). Reports cos_med,
%>.99, %>.999 on patrol every few epochs. Chase (d1<80) is analytic seek-nearest, so we
train patrol-only by default. Goal: beat the 0.982 plateau, ideally -> ~1.0.
"""
import argparse, json, time
import torch
from rel_net import RelNet


def load(path):
    d = torch.load(path, map_location='cpu')
    F = d['feats'].float(); M = (d['mask'] > 0.5).float()
    P = d['pvel'].float(); Y = d['force'].float(); D = d['d1']
    if P.abs().max() > 2.0:               # stored as raw vel -> normalize
        P = P / 2.5
    patrol = (D > 80.0) | ~torch.isfinite(D)
    return F, M, P, Y, patrol


def cos_stats(pred, tgt, msk):
    pn = pred / (pred.norm(dim=1, keepdim=True) + 1e-9)
    tn = tgt / (tgt.norm(dim=1, keepdim=True) + 1e-9)
    c = (pn * tn).sum(1).clamp(-1, 1)[msk]
    return (c.median().item(), (c > 0.99).float().mean().item() * 100,
            (c > 0.999).float().mean().item() * 100, c.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True); ap.add_argument('--val', required=True)
    ap.add_argument('--mode', default='edge')
    ap.add_argument('--d', type=int, default=64); ap.add_argument('--rho', default='128,64')
    ap.add_argument('--nblocks', type=int, default=2); ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--edge_hidden', type=int, default=64); ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--ffn_mult', type=int, default=4); ap.add_argument('--n_seeds', type=int, default=2)
    ap.add_argument('--nrbf', type=int, default=16); ap.add_argument('--no_reinject', action='store_true')
    ap.add_argument('--count_nbhd', action='store_true', help='count residual also carries nbhd centroid/vel')
    ap.add_argument('--act', default='relu'); ap.add_argument('--use_dens', action='store_true')
    ap.add_argument('--in_dim', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=300); ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--patrolall', action='store_true', help='train on all frames (default patrol-only)')
    ap.add_argument('--tag', default='rel'); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = a.device
    if a.use_dens:
        a.in_dim = 8

    Ft, Mt, Pt, Yt, pat_t = load(a.train)
    Fv, Mv, Pv, Yv, pat_v = load(a.val)
    if not a.patrolall:
        Ft, Mt, Pt, Yt = Ft[pat_t], Mt[pat_t], Pt[pat_t], Yt[pat_t]
    rho = tuple(int(x) for x in a.rho.split(',')) if a.rho else ()
    net = RelNet(in_dim=a.in_dim, d=a.d, rho=rho, mode=a.mode, heads=a.heads,
                 nblocks=a.nblocks, edge_hidden=a.edge_hidden, K=a.K, act=a.act,
                 use_dens=a.use_dens, ffn_mult=a.ffn_mult, n_seeds=a.n_seeds,
                 nrbf=a.nrbf, reinject=not a.no_reinject, count_nbhd=a.count_nbhd).to(dev)
    net.set_standardizer(Ft, Mt)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd)
    print(f"# mode={a.mode} d={a.d} nblocks={a.nblocks} params={net.n_params()} "
          f"in_dim={a.in_dim} train={tuple(Ft.shape)} dev={dev}", flush=True)

    Yv_dev = Yv.to(dev)
    def evaluate(bs=2048):
        net.eval(); outs = []
        with torch.no_grad():
            for s in range(0, Fv.shape[0], bs):
                outs.append(net(Fv[s:s+bs].to(dev), Mv[s:s+bs].to(dev), Pv[s:s+bs].to(dev)))
        return torch.cat(outs)

    n = Ft.shape[0]; best = -1; t0 = time.time()
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(n)
        for s in range(0, n, a.bs):
            idx = perm[s:s+a.bs]
            pred = net(Ft[idx].to(dev), Mt[idx].to(dev), Pt[idx].to(dev))
            y = Yt[idx].to(dev)
            pn = pred / (pred.norm(dim=1, keepdim=True) + 1e-9)
            tn = y / (y.norm(dim=1, keepdim=True) + 1e-9)
            loss = (1 - (pn * tn).sum(1)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        if ep % 5 == 0 or ep == a.epochs - 1:
            pv = evaluate()
            cm, p99, p999, cmean = cos_stats(pv.cpu(), Yv, pat_v)
            if cm > best:
                best = cm
                torch.save(dict(state=net.state_dict(), args=vars(a), best=best,
                                params=net.n_params()), f'relnet_{a.tag}.pt')
            print(f"ep{ep:3d} pat_cosM={cm:.4f} %>.99={p99:5.1f} %>.999={p999:5.1f} "
                  f"mean={cmean:.4f} best={best:.4f} t={time.time()-t0:.0f}s", flush=True)
    print(f"# DONE tag={a.tag} best_pat_cosM={best:.4f} params={net.n_params()}", flush=True)


if __name__ == '__main__':
    main()
