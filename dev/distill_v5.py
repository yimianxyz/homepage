"""distill_v5 — scaled-capacity distillation of the best planner (K16/H120/D8).

WHY v5 (the un-tested hypothesis):
  v4's "near-tie precision wall" was measured on a *tiny* net (crossattn,
  406k params) on *tiny* data (24-256 seeds). The overfit run drove value
  SmoothL1 only to 0.07 (NOT ~0) and decisive TRAIN acc to ~0.55 — that is an
  UNDER-fit, i.e. a capacity/optimisation limit, not a proof the function is
  unlearnable. The planner's action is a deterministic map of the FULL current
  state (it rolls exact dynamics FROM that state — no hidden info), so by
  universal approximation a big enough net on enough data CAN fit it. v5 tests
  that directly: big deep transformer + large data + losses that put capacity
  where catches are won.

What v5 adds over v4:
  - Chunked planner data-gen → scale to thousands of seeds without OOM.
  - Big deep ISAB transformer (d up to 256, 4-6 layers, hid up to 512).
  - Train/VAL split → report HELD-OUT decisive acc, separating memorisation
    from generalisation (the number v4 never isolated).
  - Loss modes:
      value   : SmoothL1(score, gain)  (v4 baseline)
      margin  : per-frame value SmoothL1 weighted by (gmax-gmin) + a pairwise
                hinge so the best beats the rest by their true value gap —
                puts gradient on the decisive frames that buy the catches.
      cls     : class-balanced cross-entropy on the planner's COMMITTED choice
                (argmax gain), masked to DECISIVE frames so ties don't pull the
                net to the dominant E3D class. Pure behaviour-cloning of the
                expert's discrete action.
      listnet : KL(softmax(score) || softmax(gain/tau)).
    combos via '+': e.g. 'margin+cls', 'value+listnet'.

Usage (VM):
  python3 distill_v5.py --enc transformer --d 192 --layers 4 --hid 384 \
      --loss margin+cls --gen_seeds 1024 --gen_chunk 256 --iters 3 \
      --epochs 80 --eval_n 512 --out v5.json
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sim_torch as st
from sim_torch import Sim
import planner_probe as pp
from distill_v4 import (full_obs, cand_features, SetScorer, num_params,
                        gen_rollout, eval_closed_loop)

BASE = 8.3447


# --------------------------- chunked data gen ---------------------------

def gen_chunked(net, seeds, frames, device, K, H, D, chunk):
    """gen_rollout over seeds split into <=chunk batches; concat the 5 tensors
    (BF, MK, PSt, CF, GN) and report the closed-loop catches mean of the gen
    policy (planner if net is None)."""
    parts = [[] for _ in range(5)]
    catches = []
    for i in range(0, len(seeds), chunk):
        sub = seeds[i:i + chunk]
        out = gen_rollout(net, device, sub, frames, device, K, H, D)
        for j in range(5):
            parts[j].append(out[j])
        catches.append(out[5])
        print(f"    chunk {i//chunk}: seeds={len(sub)} rows={out[0].shape[0]} "
              f"gen_mean={out[5].mean():.2f}", flush=True)
    data = tuple(torch.cat(parts[j]) for j in range(5))
    return data, float(np.concatenate(catches).mean())


# --------------------------- losses ---------------------------

def compute_loss(s, gain, modes, tau, margin_w, class_w):
    """s,gain: (B,K). modes: set of strings. Returns scalar loss."""
    loss = s.new_zeros(())
    gmax = gain.max(1).values
    gmin = gain.min(1).values
    decisive = gmax > gmin
    lab = gain.argmax(1)
    if 'value' in modes:
        loss = loss + F.smooth_l1_loss(s, gain)
    if 'cval' in modes:
        # per-frame CENTERED value regression. Raw gain is ~88% scene-difficulty
        # variance (nuisance) and only ~12% candidate-to-candidate signal; the
        # uncentered SmoothL1 spends its capacity fitting the scene mean and the
        # scores collapse to ~flat. Centering removes the scene mean so the net
        # only learns the within-frame ranking (the thing argmax actually uses).
        sc = s - s.mean(1, keepdim=True)
        gc = gain - gain.mean(1, keepdim=True)
        w = (1.0 + margin_w * (gmax - gmin)).unsqueeze(1)
        loss = loss + (F.smooth_l1_loss(sc, gc, reduction='none') * w).mean()
    if 'margin' in modes:
        # value regression weighted per-frame by the decision margin
        w = (1.0 + margin_w * (gmax - gmin)).unsqueeze(1)        # (B,1)
        vl = F.smooth_l1_loss(s, gain, reduction='none') * w
        loss = loss + vl.mean()
        # pairwise hinge: best score should exceed every other by their gain gap
        if decisive.any():
            sb = s[decisive]; gb = gain[decisive]; lb = lab[decisive]
            rows = torch.arange(sb.shape[0], device=s.device)
            s_best = sb[rows, lb].unsqueeze(1)                   # (b,1)
            g_best = gb[rows, lb].unsqueeze(1)
            gap = (g_best - gb).clamp(min=0)                     # desired margin
            viol = (gap - (s_best - sb)).clamp(min=0)            # hinge
            loss = loss + viol.mean()
    if 'cls' in modes:
        if decisive.any():
            sb = s[decisive]; lb = lab[decisive]
            loss = loss + F.cross_entropy(sb, lb, weight=class_w)
    if 'listnet' in modes:
        tgt = F.softmax(gain / tau, dim=1)
        loss = loss + F.kl_div(F.log_softmax(s, dim=1), tgt, reduction='batchmean')
    return loss


def decisive_acc(net, data, device, bs=2048):
    BF, MK, PSt, CF, GN = data
    n = BF.shape[0]
    lab = GN.argmax(1)
    gmax = GN.max(1).values; gmin = GN.min(1).values
    dec = gmax > gmin
    correct = dn = e3d_correct = 0
    net.eval()
    with torch.no_grad():
        for j in range(0, n, bs):
            sl = slice(j, j + bs)
            s = net(BF[sl].to(device), MK[sl].to(device),
                    PSt[sl].to(device), CF[sl].to(device))
            pred = s.argmax(1).cpu()
            db = dec[sl]
            if db.any():
                correct += int((pred[db] == lab[sl][db]).sum())
                e3d_correct += int((lab[sl][db] == 0).sum())
                dn += int(db.sum())
    return correct / max(dn, 1), e3d_correct / max(dn, 1)


def class_weights(GN, K, device):
    lab = GN.argmax(1)
    dec = GN.max(1).values > GN.min(1).values
    lab = lab[dec]
    cnt = torch.bincount(lab, minlength=K).float()
    w = 1.0 / torch.sqrt(cnt.clamp(min=1))
    w = w / w.mean()
    return w.to(device)


# --------------------------- train ---------------------------

def train(tr, va, enc, device, epochs, modes, tau, margin_w, d, layers, hid,
          heads, lr, bs, out_json=None):
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    BF, MK, PSt, CF, GN = [t.to(device) for t in tr]
    K = CF.shape[1]
    cw = class_weights(GN, K, device) if 'cls' in modes else None
    net = SetScorer(enc, d, layers, heads=heads, hid=hid).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = BF.shape[0]
    print(f"  params={num_params(net)} rows={n} modes={sorted(modes)}", flush=True)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for j in range(0, n, bs):
            b = perm[j:j + bs]
            s = net(BF[b], MK[b], PSt[b], CF[b])
            loss = compute_loss(s, GN[b], modes, tau, margin_w, cw)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            tr_acc, tr_e3d = decisive_acc(net, tr, device)
            va_acc, va_e3d = decisive_acc(net, va, device) if va else (0, 0)
            print(f"    ep{ep}: loss={tot/n:.4f} dec_acc tr={tr_acc:.3f} "
                  f"va={va_acc:.3f} (E3D-const va={va_e3d:.3f})", flush=True)
    return net


def split_tr_va(data, va_frac=0.1, seed=0):
    n = data[0].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    nv = int(n * va_frac)
    vi, ti = perm[:nv], perm[nv:]
    tr = tuple(t[ti] for t in data)
    va = tuple(t[vi] for t in data)
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', choices=['deepsets', 'transformer', 'crossattn'], default='transformer')
    ap.add_argument('--loss', default='margin+cls', help="+ -joined: value,margin,cls,listnet")
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--margin_w', type=float, default=4.0)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--H', type=int, default=120)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--gen_seeds', type=int, default=1024)
    ap.add_argument('--gen_chunk', type=int, default=256)
    ap.add_argument('--gen_seedStart', type=int, default=400000)
    ap.add_argument('--gen_frames', type=int, default=1500)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--retain', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--d', type=int, default=192)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--hid', type=int, default=384)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--hold_D', type=int, default=8)
    ap.add_argument('--eval_seedStart', type=int, default=200000)
    ap.add_argument('--eval_n', type=int, default=512)
    ap.add_argument('--eval_frames', type=int, default=1500)
    ap.add_argument('--weights', default='predator_weights.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None)
    ap.add_argument('--save_data', default=None, help='cache iter-0 planner dataset to this .pt')
    ap.add_argument('--load_data', default=None, help='load cached iter-0 dataset, skip gen')
    ap.add_argument('--dense', type=float, default=0.0,
                    help='DENSE_LAMBDA proximity tie-breaker on teacher gain (0=pure integer)')
    ap.add_argument('--save_net', default=None, help='save net state_dict+arch to this .pt')
    args = ap.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    pp.WEIGHTS = st.load_weights(args.weights, device=device)
    pp.DENSE_LAMBDA = args.dense
    modes = set(args.loss.split('+'))
    gen_seeds = list(range(args.gen_seedStart, args.gen_seedStart + args.gen_seeds))
    eval_seeds = list(range(args.eval_seedStart, args.eval_seedStart + args.eval_n))
    t0 = time.time()
    print(f"[v5] enc={args.enc} d={args.d} L={args.layers} hid={args.hid} "
          f"loss={args.loss} margin_w={args.margin_w} K={args.K} H={args.H} D={args.D} "
          f"gen_seeds={args.gen_seeds}", flush=True)

    if args.load_data:
        print(f"[v5] iter 0: load cached dataset {args.load_data}", flush=True)
        blob = torch.load(args.load_data, map_location='cpu')
        data, gmean = tuple(blob['data']), blob['gmean']
    else:
        print("[v5] iter 0: planner data gen (chunked)", flush=True)
        data, gmean = gen_chunked(None, gen_seeds, args.gen_frames, device,
                                  args.K, args.H, args.D, args.gen_chunk)
        if args.save_data:
            torch.save({'data': list(data), 'gmean': gmean}, args.save_data)
            print(f"  saved dataset -> {args.save_data}", flush=True)
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    print(f"  planner_mean={gmean:.2f} rows={data[0].shape[0]} {time.time()-t0:.0f}s", flush=True)
    bufs = [data]
    history = []
    net = None
    for it in range(args.iters):
        keep = bufs[-args.retain:]
        alld = tuple(torch.cat([b[i] for b in keep]) for i in range(5))
        tr, va = split_tr_va(alld)
        print(f"[v5] iter {it} train rows={alld[0].shape[0]}", flush=True)
        net = train(tr, va, args.enc, device, args.epochs, modes, args.tau,
                    args.margin_w, args.d, args.layers, args.hid, args.heads,
                    args.lr, args.bs)
        catches, pick0 = eval_closed_loop(net, eval_seeds, args.eval_frames,
                                          device, args.K, args.hold_D)
        mean = float(catches.mean()); se = float(catches.std(ddof=1) / np.sqrt(len(catches)))
        pct_base = 100.0 * (mean - BASE) / BASE
        pct_plan = 100.0 * mean / 21.40
        print(f"[v5] iter {it} EVAL mean={mean:.3f}±{se:.3f} vs_base={pct_base:+.1f}% "
              f"of_planner={pct_plan:.1f}% pick0={pick0:.3f} {time.time()-t0:.0f}s", flush=True)
        history.append(dict(iter=it, mean=mean, se=se, pct_base=pct_base,
                            of_planner=pct_plan, pick0=pick0))
        if args.out:
            with open(args.out, 'w') as fh:
                json.dump(dict(args=vars(args), params=num_params(net),
                               planner_mean=gmean, history=history,
                               elapsed=time.time() - t0), fh)
        if args.save_net:
            torch.save(dict(state_dict=net.state_dict(),
                            arch=dict(enc=args.enc, d=args.d, layers=args.layers,
                                      heads=args.heads, hid=args.hid),
                            iter=it, mean=mean, of_planner=pct_plan),
                       args.save_net)
            print(f"  saved net -> {args.save_net} (mean={mean:.3f})", flush=True)
        if it < args.iters - 1:
            print(f"[v5] iter {it} DAgger relabel", flush=True)
            dr, drm = gen_chunked(net, gen_seeds, args.gen_frames, device,
                                  args.K, args.H, args.D, args.gen_chunk)
            print(f"  net_rollout_mean={drm:.2f}", flush=True)
            bufs.append(dr)
    print("DONE", json.dumps(history), flush=True)


if __name__ == '__main__':
    main()
