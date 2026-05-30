"""EXACT E3D oracle on the NON-toroidal setds features. Reconstructs computeEvolvedTarget
(predator.js) from the stored per-boid features and compares seek(target) to the force label
on PATROL frames. This is the true achievable ceiling GIVEN the (fp16, non-toroidal) features:
  patrol cos ~1.0  -> features fully encode the target; nets stuck below are an OPT failure.
  patrol cos <1.0  -> fp16/feature loss caps it; better data-gen (fp32) is the lever.

Feature frame (set_obs, now non-toroidal):
  feats[..,0:2]=rel_pos/HALF (boid - pred, raw)   feats[..,2:4]=rel_vel/BOID_MAX (boid - pred)
absolute boid velocity = rel_vel + pred_vel, with pvel = pred_vel / MAXSPEED.

  python3 e3d_oracle_nt.py setds_densAnt_val.pt --device cuda
"""
import argparse, torch
HALF, BOID_MAX, MAXSPEED, MAXFORCE = 840.0, 6.0, 2.5, 0.05
CR, DENS_POW, REACH, SHARP, NBHD, LEADS, LEADM = 178.09, 2.373, 1515.0, 9.25, 0.461, 0.454, 230.6

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('--device', default='cuda'); ap.add_argument('--bs', type=int, default=2048)
    a = ap.parse_args()
    d = torch.load(a.inp, map_location='cpu')
    F = d['feats']; M = d['mask']; P = d['pvel']; Y = d['force']; D = d['d1']
    B, N, _ = F.shape; dev = a.device; R2 = CR * CR
    seek_all = torch.empty(B, 2); seek_c = torch.empty(B, 2); seek_cn = torch.empty(B, 2)
    eye = torch.eye(N, dtype=torch.bool, device=dev)[None]
    for s in range(0, B, a.bs):
        f = F[s:s+a.bs].float().to(dev); m = (M[s:s+a.bs] > 0.5).to(dev)
        pv = P[s:s+a.bs].float().to(dev)
        pv = pv * MAXSPEED if pv.abs().max() < 2.0 else pv          # predator abs velocity
        pos = f[..., :2] * HALF                                     # rel to predator, non-toroidal
        vel = f[..., 2:4] * BOID_MAX + pv[:, None, :]               # ABSOLUTE boid velocity
        off = pos[:, :, None, :] - pos[:, None, :, :]               # i->j, raw (matches production)
        d2 = (off ** 2).sum(-1)
        within = (d2 < R2) & m[:, None, :]                          # j alive within CR of i (incl self)
        cnt = within.sum(-1).float()
        dpred = pos.norm(dim=-1)
        attract = torch.pow(cnt + 1.0, DENS_POW) * torch.exp(-dpred / REACH)
        attract = attract.masked_fill(~m, -1.0)
        amax = attract.clamp(min=1e-12).max(dim=1, keepdim=True).values
        w = torch.pow((attract / amax).clamp(min=0), SHARP) * m.float()
        wsum = w.sum(1, keepdim=True).clamp(min=1e-12)
        cx = (w[..., None] * pos).sum(1) / wsum
        vx = (w[..., None] * vel).sum(1) / wsum
        cx_only = cx.clone()
        best = attract.argmax(dim=1)
        bi = torch.arange(f.shape[0], device=dev)
        bpos = pos[bi, best]
        g2 = ((pos - bpos[:, None, :]) ** 2).sum(-1)
        nbr = (g2 < R2) & m
        ns = nbr.sum(1, keepdim=True).float().clamp(min=1e-12)
        ncx = (nbr[..., None] * pos).sum(1) / ns
        nvx = (nbr[..., None] * vel).sum(1) / ns
        cx = (1 - NBHD) * cx + NBHD * ncx
        vx = (1 - NBHD) * vx + NBHD * nvx
        dcent = cx.norm(dim=1, keepdim=True)
        lead = (dcent / MAXSPEED * LEADS).clamp(0, LEADM)
        tgt = cx + lead * vx
        seek_all[s:s+a.bs] = seek_step(tgt[:, 0], tgt[:, 1], pv[:, 0], pv[:, 1]).cpu()
        seek_c[s:s+a.bs] = seek_step(cx_only[:, 0], cx_only[:, 1], pv[:, 0], pv[:, 1]).cpu()
        seek_cn[s:s+a.bs] = seek_step(cx[:, 0], cx[:, 1], pv[:, 0], pv[:, 1]).cpu()
    yn = Y / (Y.norm(dim=1, keepdim=True) + 1e-9)
    chase = (torch.isfinite(D) & (D < 80.0)); patrol = ~chase
    print(f"{a.inp}  patrol={int(patrol.sum())}")
    for name, seek in [('centroid_only', seek_c), ('+nbhd', seek_cn), ('+nbhd+lead', seek_all)]:
        pn = seek / (seek.norm(dim=1, keepdim=True) + 1e-9)
        cos = (pn * yn).sum(1).clamp(-1, 1)[patrol]
        print(f"  {name:14s} patrol cos_med={cos.median():.4f} cos_mean={cos.mean():.4f} "
              f"%>.99={(cos>0.99).float().mean()*100:5.1f} %>.999={(cos>0.999).float().mean()*100:5.1f}")
    # cross-check: exact stored autoTarget offset (if present)
    if 'auto' in d and 'ppos' in d:
        off = (d['auto'] - d['ppos']).to(dev)
        pvg = (P * MAXSPEED if P.abs().max() < 2.0 else P).to(dev)
        sk = seek_step(off[:,0], off[:,1], pvg[:,0], pvg[:,1]).cpu()
        pn = sk / (sk.norm(dim=1, keepdim=True)+1e-9)
        cos = (pn*yn).sum(1).clamp(-1,1)[patrol]
        print(f"  [stored autoTarget] patrol cos_med={cos.median():.4f} %>.999={(cos>0.999).float().mean()*100:5.1f}")

if __name__ == '__main__':
    main()
