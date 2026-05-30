"""Scan all setnet_*.pt in cwd, eval each on the matching setds_*_val.pt with the sharp
steer-match metric, print a table ranked by patrol cos_med (the hard regime). One forward
pass per net; whole sweep is seconds. Reveals where prior set-net art actually stands vs the
>99% steering goal, and which regime (patrol/chase) is the bottleneck.

  python3 sweep_match.py --device cuda
"""
import glob, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/workspace/dev')
from set_e2e import load_setnet
from steer_match import cos_stats

dev = 'cuda' if (len(sys.argv) > 1 and 'cuda' in ' '.join(sys.argv)) or torch.cuda.is_available() else 'cpu'

# index val sets by (feat_dim, density_radii)
vals = {}
for vp in sorted(glob.glob('setds_*_val.pt')) + sorted(glob.glob('setds_val*.pt')):
    try:
        va = torch.load(vp, map_location='cpu')
        fd = va['feats'].shape[-1]
        rr = tuple(va.get('meta', {}).get('density_radii') or ())
        vals.setdefault((fd, rr), []).append((vp, va))
    except Exception as e:
        print(f"# skip {vp}: {e}")

rows = []
for np_ in sorted(glob.glob('setnet_*.pt')):
    try:
        net, ck = load_setnet(np_, dev)
        fd = net.fmean.numel()
        rr = tuple(ck.get('density_radii') or ())
        cand = vals.get((fd, rr))
        if not cand:
            cand = [v for (k, vlist) in vals.items() if k[0] == fd for v in vlist]
        if not cand:
            print(f"# no val match for {np_} (fd={fd} rr={rr})"); continue
        vp, va = cand[0]
        F_, M_, P_, Y_, D_ = va['feats'], va['mask'], va['pvel'], va['force'], va['d1']
        preds = []
        for j in range(0, F_.shape[0], 16384):
            with torch.no_grad():
                preds.append(net(F_[j:j+16384].float().to(dev), M_[j:j+16384].float().to(dev),
                                 P_[j:j+16384].float().to(dev)).float().cpu())
        pred = torch.cat(preds, 0); tgt = Y_.float()
        chase = (torch.isfinite(D_) & (D_ < 80.0)); patrol = ~chase
        sa = cos_stats(pred, tgt, None)
        sp = cos_stats(pred, tgt, patrol)
        sc = cos_stats(pred, tgt, chase)
        rows.append((sp['cos_med'] if sp else -1, np_, ck, sa, sp, sc, os.path.basename(vp)))
    except Exception as e:
        print(f"# err {np_}: {e}")

rows.sort(reverse=True)
print(f"{'net':28s} {'mode/pool':14s} {'par':>6s} {'all_cosM':>8s} {'pat_cosM':>8s} {'pat%>99':>7s} {'chs_cosM':>8s} {'pat_ang':>7s}")
for _, np_, ck, sa, sp, sc, vp in rows:
    mp = f"{ck['mode']}/{ck.get('pool')}"
    print(f"{os.path.basename(np_):28s} {mp:14s} {ck['meta']['params']:6d} "
          f"{sa['cos_med']:8.4f} {sp['cos_med']:8.4f} {sp['pct_gt99']:7.1f} "
          f"{sc['cos_med']:8.4f} {sp['ang_med']:7.2f}  [{vp}]")
