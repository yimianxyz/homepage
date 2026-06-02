"""Does a CHEAP linear-motion lookahead predict the teacher's chosen candidate?

For each candidate c we crudely simulate: predator drives toward c at max speed
(optionally snapping to chase a nearby boid), boids drift at constant velocity,
count boids swept within catch radius over T frames. If argmax of this estimate
matches the teacher's argmax much better than raw geometry (~0.19 on decisive
frames), then this is a learnable feature and supervised distillation is fixable
by feature-augmentation (computable from the cache, no sim regen, browser-cheap).
"""
import sys, torch

PS, VS = 200.0, 6.0
MAXSPD = 2.5
BASE, MAXS = 12.0, 12.0 * 1.8

path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/ds1024_dense08.pt'
T = int(sys.argv[2]) if len(sys.argv) > 2 else 120
CHASE_R = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0  # snap-to-chase radius (0=off)
blob = torch.load(path, map_location='cpu')
BF, MK, PSt, CF, GN = [t.float() for t in blob['data']]
N, K = GN.shape

lab = GN.argmax(1)
srt = GN.sort(1, descending=True).values
margin12 = srt[:, 0] - srt[:, 1]
dec = margin12 > 0.5

# sample: all decisive + equal number of random, capped
torch.manual_seed(0)
dec_idx = dec.nonzero(as_tuple=True)[0]
M = min(6000, len(dec_idx))
sel = dec_idx[torch.randperm(len(dec_idx))[:M]]
print(f"rows={N} sampling {M} decisive (margin>0.5) rows; T={T} chase_R={CHASE_R}")


@torch.no_grad()
def lookahead_estimate(idx, batch=128):
    out = torch.empty(len(idx), K)
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        nb = len(b)
        # world frame: predator at origin, boids relative to predator at t=0
        p = torch.zeros(nb, K, 2)
        v = (PSt[b, :2] * VS)[:, None, :].repeat(1, K, 1)      # (nb,K,2)
        tgt = CF[b, :, :2] * PS                                # (nb,K,2)
        bp = (BF[b, :, :2] * PS)[:, None, :, :].repeat(1, K, 1, 1)   # (nb,K,120,2)
        bv = (BF[b, :, 2:4] * VS)[:, None, :, :].repeat(1, K, 1, 1)
        alive = MK[b].bool()[:, None, :].repeat(1, K, 1)            # (nb,K,120)
        szfrac = PSt[b, 4]
        crad = (BASE + szfrac * (MAXS - BASE)) * 0.7
        crad = crad[:, None, None]                                  # (nb,1,1)
        cnt = torch.zeros(nb, K)
        for t in range(T):
            # steering target: nearest live boid if within CHASE_R else candidate
            d2 = ((bp - p[:, :, None, :]) ** 2).sum(-1)             # (nb,K,120)
            d2 = torch.where(alive, d2, torch.full_like(d2, 1e18))
            mind, mi = d2.min(-1)
            steer_to = tgt.clone()
            if CHASE_R > 0:
                near = mind < CHASE_R ** 2
                nb_pos = torch.gather(bp, 2, mi[:, :, None, None].expand(-1, -1, 1, 2)).squeeze(2)
                steer_to = torch.where(near[:, :, None], nb_pos, tgt)
            dirv = steer_to - p
            n = dirv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            v = dirv / n * MAXSPD
            p = p + v
            bp = bp + bv
            d = ((bp - p[:, :, None, :]) ** 2).sum(-1).sqrt()
            caught = (d < crad) & alive
            cnt = cnt + caught.float().sum(-1)
            alive = alive & ~caught
        out[s:s + batch] = cnt
    return out


est = lookahead_estimate(sel)
pred = est.argmax(1)
true = lab[sel]
acc = float((pred == true).float().mean())
# compare to geometry baselines on the same sample
near = torch.ones(len(sel), dtype=torch.long)            # candidate 1 = nearest boid
e3d = torch.zeros(len(sel), dtype=torch.long)
near_acc = float((near == true).float().mean())
e3d_acc = float((e3d == true).float().mean())
argmin_dist = (CF[sel, :, 2]).argmin(1)
amd_acc = float((argmin_dist == true).float().mean())
# top-2 / top-3 accuracy of the lookahead estimate
top3 = est.topk(3, dim=1).indices
in_top3 = (top3 == true[:, None]).any(1).float().mean()
print(f"\n--- accuracy vs teacher argmax on DECISIVE (margin>0.5) frames ---")
print(f"  cheap_lookahead   : {acc:.3f}   (top-3: {float(in_top3):.3f})")
print(f"  argmin_dist       : {amd_acc:.3f}")
print(f"  always_nearest(1) : {near_acc:.3f}")
print(f"  always_E3D(0)     : {e3d_acc:.3f}")
# correlation of estimate ranking with teacher gain ranking (spearman-ish: does
# the lookahead VALUE track the teacher gain?)
gn = GN[sel]
# rank corr per row, averaged
def avg_rank_corr(a, b):
    ar = a.argsort(1).argsort(1).float(); br = b.argsort(1).argsort(1).float()
    ar = ar - ar.mean(1, keepdim=True); br = br - br.mean(1, keepdim=True)
    num = (ar * br).sum(1)
    den = (ar.pow(2).sum(1).sqrt() * br.pow(2).sum(1).sqrt()).clamp(min=1e-6)
    return float((num / den).mean())
print(f"  spearman(lookahead_est, teacher_gain) over candidates: {avg_rank_corr(est, gn):.3f}")
