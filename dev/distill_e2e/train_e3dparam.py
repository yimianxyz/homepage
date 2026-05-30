"""Parametric-E3D net: the EXACT computeEvolvedTarget functional form with its ~9 constants
as learnable nn.Parameters. Trained with cosine loss on PATROL frames (chase = analytic
seek-nearest, no net). This tests the Occam endpoint: generic DeepSets/attn plateau at
patrol cos_med ~0.982 on perfect fp32 data; the production policy is an exact analytic
function, so the MINIMAL learnable net that reaches 100% should be E3D-with-learned-scalars.

  warm  : init at known E3D constants -> validates differentiable forward == oracle (~100% @ ep0)
  rand  : init perturbed -> does SGD recover 100%? (proves learnability, not just hand-coding)

  python3 train_e3dparam.py setds_densAnt32_train.pt --val setds_densAnt32_val.pt \
      --init rand --epochs 600 --lr 5e-2 --device cuda
"""
import argparse, torch, torch.nn as nn
HALF, BOID_MAX, MAXSPEED, MAXFORCE = 840.0, 6.0, 2.5, 0.05
# known E3D constants (computeEvolvedTarget)
K = dict(cr=178.09, dens_pow=2.373, reach=1515.0, sharp=9.25, nbhd=0.461, leads=0.454, leadm=230.6)


def fast_mag(x, y):
    ax, ay = x.abs(), y.abs()
    return torch.maximum(ax, ay) * 0.96 + torch.minimum(ax, ay) * 0.398


def seek_step(dx, dy, vx, vy):
    m = fast_mag(dx, dy); nz = m > 0
    dx0 = torch.where(nz, dx * MAXSPEED / (m + 1e-12), torch.zeros_like(dx))
    dy0 = torch.where(nz, dy * MAXSPEED / (m + 1e-12), torch.zeros_like(dy))
    sx, sy = dx0 - vx, dy0 - vy
    sm = fast_mag(sx, sy); over = sm > MAXFORCE
    f = torch.where(over, MAXFORCE / (sm + 1e-12), torch.ones_like(sm))
    return torch.stack([sx * f, sy * f], 1)


class E3DParam(nn.Module):
    """9 learnable scalars + the fixed E3D graph. count uses a soft (sigmoid) threshold so
    cluster_r is differentiable; everything else is the exact oracle math."""
    def __init__(self, init='warm', soft_temp=20.0):
        super().__init__()
        if init == 'warm':
            p = {k: v for k, v in K.items()}
        else:  # perturbed random start within plausible ranges
            g = torch.Generator().manual_seed(0)
            r = lambda lo, hi: (lo + (hi - lo) * torch.rand(1, generator=g)).item()
            p = dict(cr=r(80, 300), dens_pow=r(1.0, 4.0), reach=r(500, 3000),
                     sharp=r(3.0, 15.0), nbhd=r(0.1, 0.8), leads=r(0.1, 0.9), leadm=r(80, 400))
        sp_inv = lambda x: torch.tensor(float(x)).expm1().clamp(min=1e-6).log()  # inverse softplus
        self.log_cr = nn.Parameter(torch.tensor(float(p['cr'])).log())
        self.dens_pow = nn.Parameter(torch.tensor(float(p['dens_pow'])))
        self.log_reach = nn.Parameter(torch.tensor(float(p['reach'])).log())
        self.log_sharp = nn.Parameter(torch.tensor(float(p['sharp'])).log())
        self.nbhd_raw = nn.Parameter(torch.logit(torch.tensor(float(p['nbhd']))))
        self.leads_raw = nn.Parameter(sp_inv(p['leads']))           # softplus -> leads > 0
        self.log_leadm = nn.Parameter(torch.tensor(float(p['leadm'])).log())
        self.soft_temp = soft_temp

    def forward(self, f, m, pv, hard=False):
        cr = self.log_cr.exp(); R2 = cr * cr
        reach = self.log_reach.exp(); sharp = self.log_sharp.exp()
        nbhd = torch.sigmoid(self.nbhd_raw); leadm = self.log_leadm.exp()
        leads = nn.functional.softplus(self.leads_raw)
        pos = f[..., :2] * HALF
        vel = f[..., 2:4] * BOID_MAX + pv[:, None, :]
        off = pos[:, :, None, :] - pos[:, None, :, :]
        d2 = (off ** 2).sum(-1)
        def count(dd2):
            if hard:
                return (dd2 < R2).to(f.dtype)
            return torch.sigmoid((R2 - dd2) / (R2 / self.soft_temp))
        within = count(d2) * m[:, None, :]
        cnt = within.sum(-1)
        dpred = pos.norm(dim=-1)
        log_attract = self.dens_pow * torch.log(cnt + 1.0) - dpred / reach
        log_attract = log_attract.masked_fill(~(m > 0.5), -1e30)
        w = torch.softmax(sharp * log_attract, dim=1)             # = (a/amax)^sharp normalised
        cx = (w[..., None] * pos).sum(1)
        vx = (w[..., None] * vel).sum(1)
        cx_only = cx
        best = log_attract.argmax(dim=1)
        bi = torch.arange(f.shape[0], device=f.device)
        bpos = pos[bi, best]
        g2 = ((pos - bpos[:, None, :]) ** 2).sum(-1)
        nbr = count(g2) * m
        ns = nbr.sum(1, keepdim=True).clamp(min=1e-6)
        ncx = (nbr[..., None] * pos).sum(1) / ns
        nvx = (nbr[..., None] * vel).sum(1) / ns
        cx = (1 - nbhd) * cx_only + nbhd * ncx
        vx = (1 - nbhd) * vx + nbhd * nvx
        dcent = cx.norm(dim=1, keepdim=True)
        lead = (dcent / MAXSPEED * leads).clamp(0, float('inf'))
        lead = torch.minimum(lead, leadm)
        tgt = cx + lead * vx
        return seek_step(tgt[:, 0], tgt[:, 1], pv[:, 0], pv[:, 1])


def load(path, dev):
    d = torch.load(path, map_location='cpu')
    F = d['feats'].float(); M = (d['mask'] > 0.5).float(); P = d['pvel'].float()
    Y = d['force'].float(); D = d['d1']
    P = P * MAXSPEED if P.abs().max() < 2.0 else P
    patrol = ~(torch.isfinite(D) & (D < 80.0))
    return F, M, P, Y, patrol


def evalset(net, F, M, P, Y, patrol, dev, bs=4096):
    net.eval(); outs = []
    with torch.no_grad():
        for s in range(0, F.shape[0], bs):
            outs.append(net(F[s:s+bs].to(dev), M[s:s+bs].to(dev), P[s:s+bs].to(dev), hard=True).cpu())
    pred = torch.cat(outs)
    pn = pred / (pred.norm(dim=1, keepdim=True) + 1e-9)
    yn = Y / (Y.norm(dim=1, keepdim=True) + 1e-9)
    cos = (pn * yn).sum(1).clamp(-1, 1)[patrol]
    return cos.median().item(), (cos > 0.99).float().mean().item()*100, (cos > 0.999).float().mean().item()*100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('train'); ap.add_argument('--val', required=True)
    ap.add_argument('--init', default='rand'); ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--lr', type=float, default=5e-2); ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--soft_temp', type=float, default=20.0)
    ap.add_argument('--tag', default='e3dparam')
    a = ap.parse_args()
    dev = a.device
    Ft, Mt, Pt, Yt, pt = load(a.train, dev)
    Fv, Mv, Pv, Yv, pv = load(a.val, dev)
    Ft, Mt, Pt, Yt = Ft[pt], Mt[pt], Pt[pt], Yt[pt]   # train on PATROL only
    net = E3DParam(a.init, a.soft_temp).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    m0, p99_0, p999_0 = evalset(net, Fv, Mv, Pv, Yv, pv, dev)
    print(f"[{a.init}] ep0   val patrol cos_med={m0:.4f} %>.99={p99_0:5.1f} %>.999={p999_0:5.1f}")
    best = m0; torch.save(net.state_dict(), f"e3dparam_{a.tag}.pt")
    Yt_dev = Yt.to(dev); ytn = Yt_dev / (Yt_dev.norm(dim=1, keepdim=True) + 1e-9)
    n = Ft.shape[0]
    for ep in range(1, a.epochs + 1):
        net.train(); perm = torch.randperm(n)
        for s in range(0, n, a.bs):
            idx = perm[s:s+a.bs]
            pred = net(Ft[idx].to(dev), Mt[idx].to(dev), Pt[idx].to(dev))
            pn = pred / (pred.norm(dim=1, keepdim=True) + 1e-9)
            loss = (1 - (pn * ytn[idx]).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == a.epochs:
            mM, p99, p999 = evalset(net, Fv, Mv, Pv, Yv, pv, dev)
            cur = {k: round(float(v), 3) for k, v in dict(
                cr=net.log_cr.exp(), dpow=net.dens_pow, reach=net.log_reach.exp(),
                sharp=net.log_sharp.exp(), nbhd=torch.sigmoid(net.nbhd_raw),
                leads=nn.functional.softplus(net.leads_raw), leadm=net.log_leadm.exp()).items()}
            print(f"[{a.init}] ep{ep:<4d} val patrol cos_med={mM:.4f} %>.99={p99:5.1f} %>.999={p999:5.1f}  {cur}")
            if mM > best:
                best = mM; torch.save(net.state_dict(), f"e3dparam_{a.tag}.pt")
    print(f"# done best_pat_cosM={best:.4f}")


if __name__ == '__main__':
    main()
