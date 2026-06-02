"""distill_v6_pg — closed-loop policy-gradient fine-tuning of the candidate scorer.

WHY (root cause this fixes):
  Supervised distillation of the K16/H120/D8 planner stalls because the target
  (argmax over near-tied H-step catch counts) forces the student to reproduce
  ~120 frames of CHAOTIC flocking to sub-1-catch precision. dec_acc sticks near
  the trivial always-E3D rate even with a dense tie-free teacher. But the 99%
  goal is about CLOSED-LOOP CATCHES, not per-frame argmax: the student only needs
  the HIGH-IMPACT decisions right. So optimise the real objective directly.

WHAT:
  The scorer sees the same K candidates the planner branches (E3D + nearest live
  boids, lead-adjusted) and picks one each D frames. We treat that pick as a
  stochastic policy pi(a|s)=softmax(score/temp) and run REINFORCE with
  return-to-go on closed-loop catches. The scorer's ceiling IS the planner (same
  candidate set), with zero distillation gap. Warm-start from a dense-supervised
  net (--init) so PG fine-tunes from a sane policy instead of exploring from
  scratch; that is the whole pipeline: supervised pretrain -> closed-loop PG.

  Deployable: still reactive/memoryless — score K cheaply-computed candidate
  offsets, argmax, no H-step rollout in the browser.

Usage (VM):
  python3 distill_v6_pg.py --init net_deepsets_d08.pt --B 256 --frames 1500 \
      --steps 400 --lr 3e-4 --temp 1.0 --gamma 0.95 --ent 0.01 \
      --eval_every 20 --eval_n 512 --save_net pg_deepsets.pt --out v6_pg.json
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sim_torch as st
from sim_torch import Sim
import planner_probe as pp
from distill_v4 import full_obs, cand_features, SetScorer, num_params, eval_closed_loop

BASE = 8.3447
ORIG_PLANNER = 21.40


@torch.no_grad()
def rollout_collect(net, seeds, frames, device, K, D, temp):
    """One closed-loop rollout of the stochastic policy. Returns per-decision
    stored inputs (detached), sampled actions, and per-decision rewards (catches
    gained in that decision's D-frame window). env-major batch B."""
    sim = Sim(seeds=seeds, weights=pp.WEIGHTS, device=device,
              auto_target='evolved', auto_target_opts=dict(pp.E3D))
    B = sim.B
    rows = torch.arange(B, device=device)
    BF, MK, PSt, CF, ACT, REW = [], [], [], [], [], []
    held = None
    f = 0
    while f < frames:
        if f % D == 0:
            cand = pp._candidate_targets(sim, K)
            bf, mk, ps = full_obs(sim)
            cf = cand_features(sim, cand)
            logits = net(bf.float(), mk, ps.float(), cf.float()) / temp
            a = torch.distributions.Categorical(logits=logits).sample()   # (B,)
            held = cand[rows, a]
            c_before = sim.catches.clone()
            BF.append(bf.float()); MK.append(mk); PSt.append(ps.float())
            CF.append(cf.float()); ACT.append(a)
            # placeholder reward filled after the window
            REW.append(c_before)
        pp._step_with_target(sim, held)
        f += 1
        # close out a window's reward when we reach the next decision boundary
        if f % D == 0 or f == frames:
            REW[-1] = (sim.catches - REW[-1]).float()
    # any unclosed last window already handled by f==frames branch above
    T = len(ACT)
    return (torch.stack(BF), torch.stack(MK), torch.stack(PSt), torch.stack(CF),
            torch.stack(ACT), torch.stack(REW), float(sim.catches.float().mean()))


def returns_to_go(rew, gamma):
    """rew (T,B) -> discounted return-to-go (T,B)."""
    T, B = rew.shape
    G = torch.zeros_like(rew)
    acc = torch.zeros(B, device=rew.device)
    for t in range(T - 1, -1, -1):
        acc = rew[t] + gamma * acc
        G[t] = acc
    return G


def pg_update(net, opt, store, temp, gamma, ent_coef, mb, clip):
    BF, MK, PSt, CF, ACT, REW, _ = store
    T, B = ACT.shape
    G = returns_to_go(REW, gamma)                       # (T,B)
    # advantage: whiten per decision-index across the batch (variance reduction)
    adv = (G - G.mean(dim=1, keepdim=True)) / (G.std(dim=1, keepdim=True) + 1e-6)
    # flatten (T*B, ...)
    N = T * B
    bf = BF.reshape(N, *BF.shape[2:]); mk = MK.reshape(N, *MK.shape[2:])
    ps = PSt.reshape(N, *PSt.shape[2:]); cf = CF.reshape(N, *CF.shape[2:])
    act = ACT.reshape(N); advf = adv.reshape(N)
    perm = torch.randperm(N, device=bf.device)
    net.train()
    tot_loss = tot_ent = 0.0
    for j in range(0, N, mb):
        idx = perm[j:j + mb]
        logits = net(bf[idx], mk[idx], ps[idx], cf[idx]) / temp
        logp_all = F.log_softmax(logits, dim=1)
        logp = logp_all.gather(1, act[idx, None]).squeeze(1)
        p = logp_all.exp()
        ent = -(p * logp_all).sum(1)
        loss = -(logp * advf[idx]).mean() - ent_coef * ent.mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()
        tot_loss += float(loss.detach()) * len(idx); tot_ent += float(ent.mean().detach()) * len(idx)
    return tot_loss / N, tot_ent / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default=None, help='warm-start net blob (state_dict+arch)')
    ap.add_argument('--enc', default='deepsets')      # used only if --init absent
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--hid', type=int, default=256)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--frames', type=int, default=1500)
    ap.add_argument('--B', type=int, default=256, help='rollout seeds per PG step')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--seedStart', type=int, default=900000)
    ap.add_argument('--reseed_every', type=int, default=1, help='resample rollout seeds every N steps')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--temp_min', type=float, default=0.5)
    ap.add_argument('--gamma', type=float, default=0.95)
    ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--mb', type=int, default=8192)
    ap.add_argument('--clip', type=float, default=2.0)
    ap.add_argument('--eval_every', type=int, default=20)
    ap.add_argument('--eval_n', type=int, default=512)
    ap.add_argument('--eval_seedStart', type=int, default=200000)
    ap.add_argument('--eval_frames', type=int, default=1500)
    ap.add_argument('--hold_D', type=int, default=8)
    ap.add_argument('--weights', default='predator_weights.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save_net', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    pp.WEIGHTS = st.load_weights(args.weights, device=device)
    eval_seeds = list(range(args.eval_seedStart, args.eval_seedStart + args.eval_n))

    if args.init:
        blob = torch.load(args.init, map_location=device)
        a = blob['arch']
        net = SetScorer(a['enc'], a['d'], a['layers'], heads=a.get('heads', 4),
                        hid=a['hid']).to(device)
        net.load_state_dict(blob['state_dict'])
        arch = a
        print(f"[v6] warm-start {args.init} enc={a['enc']} d={a['d']} "
              f"L={a['layers']} hid={a['hid']} params={num_params(net)}", flush=True)
    else:
        net = SetScorer(args.enc, args.d, args.layers, heads=args.heads, hid=args.hid).to(device)
        arch = dict(enc=args.enc, d=args.d, layers=args.layers, heads=args.heads, hid=args.hid)
        print(f"[v6] fresh net enc={args.enc} params={num_params(net)}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    t0 = time.time()
    history = []
    best_mean = -1.0

    # baseline eval of the init policy (greedy)
    c0, pick0 = eval_closed_loop(net, eval_seeds, args.eval_frames, device, args.K, args.hold_D)
    print(f"[v6] init greedy eval mean={c0.mean():.3f} pick0={pick0:.3f} "
          f"({100*c0.mean()/ORIG_PLANNER:.1f}% of planner)", flush=True)

    for step in range(args.steps):
        frac = step / max(args.steps - 1, 1)
        temp = args.temp + (args.temp_min - args.temp) * frac
        s0 = args.seedStart + (step // args.reseed_every) * args.B
        seeds = list(range(s0, s0 + args.B))
        store = rollout_collect(net, seeds, args.frames, device, args.K, args.D, temp)
        loss, ent = pg_update(net, opt, store, temp, args.gamma, args.ent, args.mb, args.clip)
        train_catches = store[6]
        if step % 5 == 0:
            print(f"  step{step}: temp={temp:.2f} rollout_mean={train_catches:.2f} "
                  f"loss={loss:.4f} ent={ent:.3f} {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            c, pick = eval_closed_loop(net, eval_seeds, args.eval_frames, device, args.K, args.hold_D)
            mean = float(c.mean()); se = float(c.std(ddof=1) / np.sqrt(len(c)))
            pofp = 100 * mean / ORIG_PLANNER
            print(f"[v6] step{step+1} EVAL mean={mean:.3f}±{se:.3f} "
                  f"of_planner={pofp:.1f}% pick0={pick:.3f} {time.time()-t0:.0f}s", flush=True)
            history.append(dict(step=step + 1, mean=mean, se=se, of_planner=pofp,
                                pick0=pick, rollout_mean=train_catches))
            if mean > best_mean:
                best_mean = mean
                if args.save_net:
                    torch.save(dict(state_dict=net.state_dict(), arch=arch,
                                    step=step + 1, mean=mean,
                                    of_planner=pofp), args.save_net)
                    print(f"  saved best -> {args.save_net} (mean={mean:.3f})", flush=True)
            if args.out:
                with open(args.out, 'w') as fh:
                    json.dump(dict(args=vars(args), arch=arch, params=num_params(net),
                                   best_mean=best_mean, history=history,
                                   elapsed=time.time() - t0), fh)
    print(f"[v6] DONE best_mean={best_mean:.3f} ({100*best_mean/ORIG_PLANNER:.1f}% of planner) "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
