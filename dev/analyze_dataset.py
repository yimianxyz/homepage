"""Diagnose WHY supervised distillation stalls: is the teacher's chosen candidate
predictable from cheap structure, or is the label near-random (ties)?

Loads a distill_v5 cache {data:[BF,MK,PSt,CF,GN], gmean}. For each row:
  GN (16,) = per-candidate dense gain (integer catches over H + 0.8*bonus)
  CF (16,4)= candidate feats [rel_x/PS, rel_y/PS, dist/PS, is_e3d]
  BF (120,5)=boid feats [dx/PS, dy/PS, vx/VS, vy/VS, alive] (predator-relative)
Computes the decision margin distribution and the accuracy of simple heuristics
vs the teacher's argmax, overall and restricted to high-margin (decisive) frames.
"""
import sys, torch

PS = 200.0
path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/ds1024_dense08.pt'
blob = torch.load(path, map_location='cpu')
BF, MK, PSt, CF, GN = [t.float() for t in blob['data']]
N, K = GN.shape
print(f"rows={N} K={K} gmean={blob.get('gmean')}")

lab = GN.argmax(1)
srt = GN.sort(1, descending=True).values
gmax, g2, gmin = srt[:, 0], srt[:, 1], srt[:, -1]
margin12 = gmax - g2          # top1 - top2 : how clear the decision is
spread = gmax - gmin

def frac(m): return float(m.float().mean())
print("\n--- decision-margin (top1 - top2) distribution ---")
for thr in [0.0, 1e-6, 0.05, 0.1, 0.25, 0.5, 0.9, 1.0]:
    print(f"  margin>{thr:<5}: {frac(margin12>thr):.3f}")
print(f"  exact ties (margin==0): {frac(margin12<=1e-6):.3f}")

print("\n--- label (argmax) distribution over candidate index ---")
cnt = torch.bincount(lab, minlength=K)
for i in range(K):
    print(f"  cand{i:2d}: {cnt[i].item():6d} ({100*cnt[i].item()/N:.1f}%)", end='')
    if i % 4 == 3: print()
print()

# ---- heuristic predictions ----
dist = CF[:, :, 2]                      # (N,K) candidate distance / PS
boid_xy = BF[:, :, :2] * PS             # (N,120,2) rel to predator
alive = MK.bool()
cand_xy = CF[:, :, :2] * PS             # (N,K,2) rel to predator

def density(R):
    # count live boids within R px of each candidate
    d = torch.cdist(cand_xy, boid_xy)   # (N,K,120)
    within = (d < R) & alive[:, None, :]
    return within.sum(-1).float()       # (N,K)

preds = {
    'always_E3D(0)': torch.zeros(N, dtype=torch.long),
    'always_nearest(1)': torch.ones(N, dtype=torch.long),
    'argmin_dist': dist.argmin(1),
    'max_density_R40': density(40).argmax(1),
    'max_density_R80': density(80).argmax(1),
    'random': torch.randint(0, K, (N,)),
}

def acc(pred, mask=None):
    if mask is None: return float((pred == lab).float().mean())
    if mask.sum() == 0: return float('nan')
    return float((pred[mask] == lab[mask]).float().mean())

dec = margin12 > 0.5     # "decisive": top choice worth ~>=0.5 catch over 2nd
dec_hi = margin12 > 0.9  # near a full extra catch
print(f"\n--- heuristic acc vs teacher argmax  (decisive frac>0.5: {frac(dec):.3f}, >0.9: {frac(dec_hi):.3f}) ---")
print(f"{'heuristic':22s} {'all':>7s} {'margin>0.5':>11s} {'margin>0.9':>11s}")
for name, p in preds.items():
    print(f"{name:22s} {acc(p):7.3f} {acc(p,dec):11.3f} {acc(p,dec_hi):11.3f}")

# best-single-heuristic ceiling on decisive frames tells us if structure exists
print("\n--- interpretation hints ---")
print(f"  mean live boids: {alive.float().sum(1).mean():.1f}")
print(f"  E3D is the actual best on decisive>0.5 frames: {acc(torch.zeros(N,dtype=torch.long),dec):.3f}")
print(f"  nearest-boid best on decisive>0.5 frames     : {acc(torch.ones(N,dtype=torch.long),dec):.3f}")
