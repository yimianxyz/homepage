#!/usr/bin/env python3
# THROUGHPUT GATE — batched torch-float64 skeleton of prod's planCheap.
#
# Purpose (SPEC.md section 6, "Torch float64 bitwise replica - throughput gate
# first"): measure whether a GPU replica of the production planning rollout can
# beat the node CPU farm by >= ~10x in plan-decisions/sec. This is a TIMING
# skeleton of the faithful STRUCTURE, not a bitwise replica: stock torch
# exp/pow/erf are used, and the spatial-hash grid is replaced by all-pairs
# neighbor math (see caveats in REPORT.md).
#
# Structure replicated (file:line refer to js/predator_cheap.js @ rl/teacher):
#   planCheap (:260-291), one "plan" =
#     phase A: candidates (:233-255, incl. computeEvolvedTarget predator.js:29-89)
#              + cp_features (cheap_planner.js:53-100, incl. the 90-step
#              sequential 2-body cp_ballistic :29-48 per candidate)
#              + cp_value (cheap_planner.js:104-121, 23->48->48->1 GELU MLP)
#              + top-K_roll=4 by ballistic pscore (:271-272)
#     4x rolloutFlatState(Hs=90) (:204-229). Each step:
#       pass 1 (:215): accumulateFlock for every alive boid against FROZEN
#               state (only _ax/_ay written) -> genuinely parallel across boids
#               => batched here (chunked all-pairs einsum).
#       pass 2 (:216): SEQUENTIAL over boids: accumulateFlock AGAIN +
#               updateBoid + gridMove in place, so boid i reads already-updated
#               state for j<i. NOT parallelizable across boids => emulated as a
#               Python loop over boid index with tensors batched across the
#               R = 4*P independent rollouts only.
#       predatorStepFlat (:217, :168-191), then catch scan (:218-225): lowest-
#               index alive boid within size*CATCH_FACTOR, at most ONE catch
#               per step ('break'), alive->0 (+ grid removal in prod).
#     terminal: candidates + cp_features + cp_value on each rollout's terminal
#               state (:277-284), boot = max V (NaN-safe -> -inf), score =
#               catches + boot; argmax over 16 (:286-287).
#
# Usage:
#   python3 skeleton_torch.py --device cuda --Ns 120,30 --Bs 256,1024,4096,16384
#   python3 skeleton_torch.py --parity parity_N120_H10.json --device cpu
import argparse, json, math, time

import torch

DT = torch.float64

# ---- constants (boid.js:1-21, predator.js:2-21, predator_cheap.js:27-31,
# ---- cheap_planner.js:15-16). PREDATOR_RANGE bakes to 80 on desktop (see
# ---- dev/fasteval.js:81-87); W/Hc from the 1512x982 desktop device cell.
MAX_SPEED = 6.0
MAX_FORCE = 0.1
DESIRED_SEPARATION = 40.0
NEIGHBOR_DISTANCE = 60.0
BORDER_OFFSET = 10.0
EPSILON = 1e-7
PREDATOR_RANGE = 80.0
PREDATOR_TURN_FACTOR = 0.3
PRED_MAX_SPEED = 2.5
PRED_MAX_FORCE = 0.05
SEP_MULT, COH_MULT, ALIGN_MULT = 2.0, 1.0, 1.0
BASE_SIZE = 12.0
MAX_SIZE = BASE_SIZE * 1.8
GROWTH = 1.2
CATCH_FACTOR = 0.7
POLICY_R = 80.0
K, K_ROLL, HS = 16, 4, 90
CP_PS, CP_VS, CP_RHO, CP_HB, CP_INIT_N = 200.0, 6.0, 70.0, 90, 120
E3D = dict(cluster_r=178.09, dens_pow=2.373, reach_scale=1515.0, sharp=9.25,
           lead_scale=0.454, lead_max=230.6, nbhd=0.461)
CFG_W, CFG_H = 1512.0, 982.0   # overridden by --W/--H


# ---- vector helpers on (..., 2) tensors --------------------------------------
def fmag(v):
    # fastMag (predator_cheap.js:33-36): max*0.96 + min*0.398 of |x|,|y|
    a = v.abs()
    return torch.maximum(a[..., 0], a[..., 1]) * 0.96 + \
           torch.minimum(a[..., 0], a[..., 1]) * 0.398


def vset_mag(v, mag):
    # the `fm = fastMag(...); if (fm > 0) scale to mag` pattern
    fm = fmag(v)
    pos = fm > 0
    s = mag / torch.where(pos, fm, 1.0)
    return torch.where(pos.unsqueeze(-1), v * s.unsqueeze(-1), v)


def vlimit(v, lim):
    # the `if (fm > lim) scale to lim` pattern (iFastLimit)
    fm = fmag(v)
    over = fm > lim
    s = lim / torch.where(over, fm, 1.0)
    return torch.where(over.unsqueeze(-1), v * s.unsqueeze(-1), v)


# ---- the flock rollout (rolloutFlatState) ------------------------------------
class Rollout:
    """R independent rollouts x N boids, static buffers (CUDA-graph friendly)."""

    def __init__(self, R, N, dev, W=CFG_W, Hc=CFG_H, chunk_elems=16_000_000):
        self.R, self.N, self.dev = R, N, dev
        self.W, self.Hc = W, Hc
        z = lambda *s: torch.zeros(*s, dtype=DT, device=dev)
        self.pos = z(R, N, 2); self.vel = z(R, N, 2); self.acc = z(R, N, 2)
        self.alive = z(R, N)                      # 1.0 alive / 0.0 dead
        self.ppos = z(R, 2); self.pvel = z(R, 2)
        self.psize = z(R); self.catches = z(R)
        self.tgt = z(R, 2)
        ns = torch.ones(N, N, dtype=DT, device=dev)
        ns.fill_diagonal_(0.0)
        self.notself = ns                          # (N,N): j != i
        self.rankw = torch.arange(N, 0, -1, dtype=DT, device=dev)  # N..1
        self.arangeR = torch.arange(R, device=dev)
        self.bounds = torch.tensor([W, Hc], dtype=DT, device=dev)
        self.C = max(1, int(chunk_elems // max(1, R * N)))  # pass-1 boid chunk
        self.graph = None

    def reset(self, pos, vel, ppos, pvel, psize, tgt):
        self.pos.copy_(pos); self.vel.copy_(vel); self.acc.zero_()
        self.alive.fill_(1.0)
        self.ppos.copy_(ppos); self.pvel.copy_(pvel)
        self.psize.copy_(psize); self.catches.zero_()
        self.tgt.copy_(tgt)

    def _accumulate(self, c0, c1):
        """accumulateFlock (predator_cheap.js:87-153) for boids [c0,c1) against
        the CURRENT tensors (callers choose frozen vs in-place semantics).
        All-pairs replaces the 3x3-cell grid query (work-inflating; see report).
        Writes self.acc[:, c0:c1] += contribution."""
        posI = self.pos[:, c0:c1]                  # (R,C,2)
        velI = self.vel[:, c0:c1]
        rxy = self.pos.unsqueeze(1) - posI.unsqueeze(2)        # (R,C,N,2)
        d2 = (rxy * rxy).sum(-1)                                # (R,C,N)
        dist = d2.sqrt() + EPSILON                              # :99
        nbf = self.alive.unsqueeze(1) * self.notself[c0:c1].unsqueeze(0)
        in_coh = (dist <= NEIGHBOR_DISTANCE) * nbf              # :100
        in_sep = (dist < DESIRED_SEPARATION) * nbf              # :101
        in_al = (dist < NEIGHBOR_DISTANCE) * nbf                # :108
        cn = in_coh.sum(2)
        csum = torch.einsum('rcn,rnd->rcd', in_coh, self.pos)
        # separation pair term (:102-106): -r normalized then / dist
        m = d2.sqrt()                                           # :103
        inv = torch.where(m > 0, 1.0 / m, 0.0) / dist           # :104-105
        sxy = rxy * (-inv).unsqueeze(-1)
        ssum = torch.einsum('rcn,rcnd->rcd', in_sep, sxy)
        sn = in_sep.sum(2)
        alsum = torch.einsum('rcn,rnd->rcd', in_al, self.vel)
        an = in_al.sum(2)
        # cohesion steering (:113-120)
        has_c = cn > 0
        coh_d = csum / torch.where(has_c, cn, 1.0).unsqueeze(-1) - posI
        coh = vlimit(vset_mag(coh_d, MAX_SPEED) - velI, MAX_FORCE)
        coh = torch.where(has_c.unsqueeze(-1), coh, 0.0)
        # separation steering (:121-128)
        has_s = sn > 0
        sep0 = torch.where(has_s.unsqueeze(-1),
                           ssum / torch.where(has_s, sn, 1.0).unsqueeze(-1), 0.0)
        gs = fmag(sep0) > 0
        sep1 = vlimit(vset_mag(sep0, MAX_SPEED) - velI, MAX_FORCE)
        sep = torch.where(gs.unsqueeze(-1), sep1, sep0)
        # alignment steering (:129-135)
        has_a = an > 0
        al_d = alsum / torch.where(has_a, an, 1.0).unsqueeze(-1)
        al = vlimit(vset_mag(al_d, MAX_SPEED) - velI, MAX_FORCE)
        al = torch.where(has_a.unsqueeze(-1), al, 0.0)
        contrib = coh * COH_MULT + sep * SEP_MULT + al * ALIGN_MULT  # :137-142
        # predator flee (:143-152)
        q = posI - self.ppos.unsqueeze(1)
        pd = (q * q).sum(-1).sqrt() + EPSILON
        fq = fmag(q)
        qp = fq > 0
        qu = torch.where(qp.unsqueeze(-1), q / torch.where(qp, fq, 1.0).unsqueeze(-1), q)
        strg = (PREDATOR_RANGE - pd) / PREDATOR_RANGE * PREDATOR_TURN_FACTOR
        qv = vlimit(qu * strg.unsqueeze(-1), MAX_FORCE * 1.5)
        contrib = contrib + torch.where((pd < PREDATOR_RANGE).unsqueeze(-1), qv, 0.0)
        self.acc[:, c0:c1] += contrib

    def _update_boid(self, i):
        """updateBoid (:155-166) for boid i, in place, masked by alive
        (prod skips dead boids entirely; their state must stay frozen)."""
        ab = (self.alive[:, i] > 0.5).unsqueeze(-1)             # (R,1)
        nv = vlimit(self.vel[:, i] + self.acc[:, i], MAX_SPEED)
        np_ = self.pos[:, i] + nv
        # torus wrap (:160-164): teleport to the opposite edge, NOT modular
        np_ = torch.where(np_ > self.bounds + BORDER_OFFSET, -BORDER_OFFSET, np_)
        np_ = torch.where(np_ < -BORDER_OFFSET, self.bounds + BORDER_OFFSET, np_)
        self.pos[:, i] = torch.where(ab, np_, self.pos[:, i])
        self.vel[:, i] = torch.where(ab, nv, self.vel[:, i])
        self.acc[:, i] = 0.0                                    # :165

    def _predator_step(self):
        """predatorStepFlat (:168-191): nearest-alive chase within POLICY_R,
        else seek the committed target; analytic steer; wrap at +-20."""
        d = self.pos - self.ppos.unsqueeze(1)                   # (R,N,2)
        d2 = (d * d).sum(-1)
        d2m = torch.where(self.alive > 0.5, d2, torch.inf)
        bestD2, bi = d2m.min(1)
        nxy = torch.gather(d, 1, bi.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        chase = (bestD2 < POLICY_R * POLICY_R).unsqueeze(-1)
        des = torch.where(chase, nxy, self.tgt - self.ppos)
        st = vlimit(vset_mag(des, PRED_MAX_SPEED) - self.pvel, PRED_MAX_FORCE)
        self.pvel += st
        self.pvel.copy_(vlimit(self.pvel, PRED_MAX_SPEED))
        self.ppos += self.pvel
        B = 20.0
        p = self.ppos
        p.copy_(torch.where(p > self.bounds + B, -B, p))
        p.copy_(torch.where(p < -B, self.bounds + B, p))

    def _catch(self):
        """catch scan (:218-225): lowest-index alive boid with dist < cr;
        at most one catch per step (the 'break'); size grows, alive->0."""
        e = self.ppos.unsqueeze(1) - self.pos
        dd = (e * e).sum(-1).sqrt()                             # (R,N)
        within = (self.alive > 0.5) & (dd < (self.psize * CATCH_FACTOR).unsqueeze(-1))
        hit = within.any(1)
        first = (within.to(DT) * self.rankw).argmax(1)          # lowest index
        keep = (~hit).to(DT)
        self.alive[self.arangeR, first] = self.alive[self.arangeR, first] * keep
        self.psize.copy_(torch.where(hit, (self.psize + GROWTH).clamp(max=MAX_SIZE),
                                     self.psize))
        self.catches += hit.to(DT)

    def step(self):
        # pass 1 (:215): frozen-state accumulate -> batch boids freely (chunked)
        for c0 in range(0, self.N, self.C):
            self._accumulate(c0, min(self.N, c0 + self.C))
        # pass 2 (:216): SEQUENTIAL over boids, in-place update after each
        for i in range(self.N):
            self._accumulate(i, i + 1)
            self._update_boid(i)
        self._predator_step()                                   # :217
        self._catch()                                           # :218-225

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.step()                                     # warmup
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.step()
        self.graph = g

    def run(self, H):
        if self.graph is not None:
            for _ in range(H):
                self.graph.replay()
        else:
            for _ in range(H):
                self.step()


# ---- planning machinery (cheap_planner.js + candidates/E3D) -------------------
def e3d_target(pos, vel, alive, ppos, chunk_elems=16_000_000):
    """computeEvolvedTarget (predator.js:29-89), batched + alive-masked."""
    M, N, _ = pos.shape
    R2 = E3D['cluster_r'] ** 2
    C = max(1, int(chunk_elems // max(1, M * N)))
    cnt = torch.empty(M, N, dtype=DT, device=pos.device)
    for c0 in range(0, N, C):
        c1 = min(N, c0 + C)
        d2 = ((pos.unsqueeze(1) - pos[:, c0:c1].unsqueeze(2)) ** 2).sum(-1)
        cnt[:, c0:c1] = ((d2 < R2) * alive.unsqueeze(1)).sum(-1)  # self counts
    dpred = ((pos - ppos.unsqueeze(1)) ** 2).sum(-1).sqrt()
    attract = (cnt + 1.0).pow(E3D['dens_pow']) * torch.exp(-dpred / E3D['reach_scale'])
    attract = attract * alive                       # dead excluded (attract>0)
    amax = attract.amax(1).clamp_min(1e-12)
    w = (attract / amax.unsqueeze(1)).pow(E3D['sharp']) * alive
    wsum = w.sum(1).clamp_min(1e-12)
    c = torch.einsum('mn,mnd->md', w, pos) / wsum.unsqueeze(-1)
    v = torch.einsum('mn,mnd->md', w, vel) / wsum.unsqueeze(-1)
    bi = attract.argmax(1)
    bpos = torch.gather(pos, 1, bi.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    nbr = (((pos - bpos.unsqueeze(1)) ** 2).sum(-1) < R2) * alive
    nsum = nbr.sum(1).clamp_min(1e-12)
    nc = torch.einsum('mn,mnd->md', nbr, pos) / nsum.unsqueeze(-1)
    nv = torch.einsum('mn,mnd->md', nbr, vel) / nsum.unsqueeze(-1)
    nb = E3D['nbhd']
    c = (1 - nb) * c + nb * nc
    v = (1 - nb) * v + nb * nv
    dcent = ((c - ppos) ** 2).sum(-1).sqrt()
    lead = (dcent / PRED_MAX_SPEED * E3D['lead_scale']).clamp(0, E3D['lead_max'])
    return c + lead.unsqueeze(-1) * v


def candidates(pos, vel, alive, ppos):
    """candidates (predator_cheap.js:233-255): cand0 = E3D patrol; cand1..15 =
    15 nearest alive boids lead-extrapolated; slots beyond nAlive pad with E3D."""
    M, N, _ = pos.shape
    e3 = e3d_target(pos, vel, alive, ppos)                      # (M,2)
    d2p = ((pos - ppos.unsqueeze(1)) ** 2).sum(-1)
    d2p = torch.where(alive > 0.5, d2p, torch.inf)
    idx = d2p.argsort(1)[:, :K - 1]                              # :240-241
    g = idx.unsqueeze(-1).expand(-1, -1, 2)
    bpos = torch.gather(pos, 1, g)
    bvel = torch.gather(vel, 1, g)
    dcent = ((bpos - ppos.unsqueeze(1)) ** 2).sum(-1).sqrt()
    lead = (dcent / PRED_MAX_SPEED * E3D['lead_scale']).clamp(0, E3D['lead_max'])
    cand = bpos + lead.unsqueeze(-1) * bvel                      # :248-249
    navail = alive.sum(1)
    pad = torch.arange(K - 1, device=pos.device).unsqueeze(0) >= navail.unsqueeze(1)
    cand = torch.where(pad.unsqueeze(-1), e3.unsqueeze(1), cand)  # :251
    return torch.cat([e3.unsqueeze(1), cand], 1)                 # (M,16,2)


def cp_ballistic(ppos, pvel, bpos, bvel, catchD):
    """cp_ballistic (cheap_planner.js:29-48): 90-step SEQUENTIAL 2-body pursuit,
    batched over (M,K)."""
    p = ppos.unsqueeze(1).expand_as(bpos).clone()
    v = pvel.unsqueeze(1).expand_as(bpos).clone()
    b = bpos.clone()
    M, Kk, _ = b.shape
    caught = torch.zeros(M, Kk, dtype=torch.bool, device=b.device)
    tCatch = torch.full((M, Kk), float(CP_HB), dtype=DT, device=b.device)
    mind = torch.full((M, Kk), torch.inf, dtype=DT, device=b.device)
    cd = catchD.unsqueeze(1)
    for t in range(CP_HB):
        d_ = b - p
        dn = (d_ * d_).sum(-1).sqrt().clamp_min(1e-6)
        mind = torch.minimum(mind, dn)
        newly = (~caught) & (dn < cd)
        tCatch = torch.where(newly, float(t), tCatch)
        caught |= newly
        des = d_ / dn.unsqueeze(-1) * PRED_MAX_SPEED - v
        sm = (des * des).sum(-1).sqrt().clamp_min(1e-6)
        v = v + des * (PRED_MAX_FORCE / sm).clamp(max=1.0).unsqueeze(-1)
        spd = (v * v).sum(-1).sqrt().clamp_min(1e-6)
        v = v * (PRED_MAX_SPEED / spd).clamp(max=1.0).unsqueeze(-1)
        p = p + v
        b = b + bvel
    return tCatch / CP_HB, mind, caught.to(DT)


def cp_features(pos, vel, alive, ppos, pvel, psize, cands):
    """cp_features (cheap_planner.js:53-100), batched + alive-masked."""
    M, N, _ = pos.shape
    catchD = psize * 0.7
    r = cands - ppos.unsqueeze(1)                                # (M,16,2)
    dist = (r * r).sum(-1).sqrt().clamp_min(1e-6)
    tgo = dist / PRED_MAX_SPEED
    isE3d = torch.zeros(M, K, dtype=DT, device=pos.device)
    isE3d[:, 0] = 1.0
    # nearest alive boid to each candidate + local density / mean vel (:66-73)
    c2 = ((cands.unsqueeze(2) - pos.unsqueeze(1)) ** 2).sum(-1)  # (M,16,N)
    c2m = torch.where((alive > 0.5).unsqueeze(1), c2, torch.inf)
    best, nb = c2m.min(-1)
    tbDist = best.sqrt()
    g = nb.unsqueeze(-1).expand(-1, -1, 2)
    tb = torch.gather(pos.unsqueeze(1).expand(-1, K, -1, -1), 2, g.unsqueeze(2)).squeeze(2)
    tbv = torch.gather(vel.unsqueeze(1).expand(-1, K, -1, -1), 2, g.unsqueeze(2)).squeeze(2)
    near = (c2m < CP_RHO * CP_RHO).to(DT)
    dens = near.sum(-1)
    mv = torch.einsum('mkn,mnd->mkd', near, vel)
    nsafe = dens.clamp_min(1.0)
    mn = mv / nsafe.unsqueeze(-1)
    tbr = tb - ppos.unsqueeze(1)
    rangepb = (tbr * tbr).sum(-1).sqrt().clamp_min(1e-6)
    relv = tbv - pvel.unsqueeze(1)
    closing = -(tbr * relv).sum(-1) / rangepb
    losRate = (tbr[..., 0] * relv[..., 1] - tbr[..., 1] * relv[..., 0]) / (rangepb * rangepb)
    u = r / dist.unsqueeze(-1)
    fleeAlign = (mn * u).sum(-1)
    tcn, mind, cgt = cp_ballistic(ppos, pvel, tb, tbv, catchD)
    feat = torch.stack([
        r[..., 0] / CP_PS, r[..., 1] / CP_PS, dist / CP_PS, isE3d,
        tgo / 60.0,
        tbr[..., 0] / CP_PS, tbr[..., 1] / CP_PS, tbv[..., 0] / CP_VS, tbv[..., 1] / CP_VS,
        tbDist / CP_PS,
        rangepb / CP_PS, closing / CP_VS, losRate * 50.0,
        dens / 20.0, fleeAlign / CP_VS,
        (rangepb - dist) / CP_PS,
        tcn, mind / CP_PS, cgt,
    ], -1)                                                       # (M,16,19)
    fracAlive = alive.sum(1) / CP_INIT_N
    ctx = torch.stack([pvel[:, 0] / CP_VS, pvel[:, 1] / CP_VS,
                       fracAlive, psize / 20.0], -1)             # (M,4)
    return feat, ctx


def load_net(path, dev):
    with open(path) as f:
        net = json.load(f)
    t = lambda x: torch.tensor(x, dtype=DT, device=dev)
    return dict(W0=t(net['layers'][0]['w']), b0=t(net['layers'][0]['b']),
                W1=t(net['layers'][1]['w']), b1=t(net['layers'][1]['b']),
                W2=t(net['layers'][2]['w']), b2=t(net['layers'][2]['b']),
                fmu=t(net['fmu']), fsd=t(net['fsd']),
                xmu=t(net['xmu']), xsd=t(net['xsd']))


def gelu(x):
    # cp_gelu (cheap_planner.js:19-25): exact erf-GELU; stock torch.erf here
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def cp_value(net, feat, ctx):
    """cp_value (cheap_planner.js:104-121): standardize, 23->48->48->1 GELU."""
    xf = (feat - net['fmu']) / net['fsd']
    xc = ((ctx - net['xmu']) / net['xsd']).unsqueeze(1).expand(-1, feat.shape[1], -1)
    x = torch.cat([xf, xc], -1)                                  # (M,16,23)
    h0 = gelu(x @ net['W0'].T + net['b0'])
    h1 = gelu(h0 @ net['W1'].T + net['b1'])
    return (h1 @ net['W2'].T + net['b2']).squeeze(-1)            # (M,16)


# ---- one full batched plan -----------------------------------------------------
def plan_batch(st, net, ro):
    """st: dict pos (P,N,2) vel ppos pvel psize (all alive). Returns (P,2)."""
    P = st['pos'].shape[0]
    alive = torch.ones(P, st['pos'].shape[1], dtype=DT, device=st['pos'].device)
    # phase A (planCheap :261-272)
    cands = candidates(st['pos'], st['vel'], alive, st['ppos'])
    feat, ctx = cp_features(st['pos'], st['vel'], alive, st['ppos'], st['pvel'],
                            st['psize'], cands)
    vprior = cp_value(net, feat, ctx)
    pscore = feat[..., 18] - feat[..., 16]                       # caught - tCatchNorm
    top4 = pscore.topk(K_ROLL, dim=1).indices                    # (P,4)
    tgts = torch.gather(cands, 1, top4.unsqueeze(-1).expand(-1, -1, 2))
    # 4 independent rollouts per plan, batched as R = 4P (:274-276)
    rep = lambda x: x.repeat_interleave(K_ROLL, dim=0)
    ro.reset(rep(st['pos']), rep(st['vel']), rep(st['ppos']), rep(st['pvel']),
             rep(st['psize']), tgts.reshape(-1, 2))
    ro.run(HS)
    # terminal bootstrap (:277-284) on all 4P terminal states at once
    tcands = candidates(ro.pos, ro.vel, ro.alive, ro.ppos)
    tfeat, tctx = cp_features(ro.pos, ro.vel, ro.alive, ro.ppos, ro.pvel,
                              ro.psize, tcands)
    tv = cp_value(net, tfeat, tctx)
    boot = torch.nan_to_num(tv, nan=-torch.inf).max(-1).values   # NaN -> -inf path
    rolled = (ro.catches + boot).view(P, K_ROLL)
    score = vprior.scatter(1, top4, rolled)                      # :273,284
    bi = score.argmax(1)                                          # :286-287
    return torch.gather(cands, 1, bi.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)


# ---- synthetic flock-like states ----------------------------------------------
def gen_states(P, N, dev, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    rnd = lambda *s: torch.rand(*s, generator=g, dtype=DT)
    ncl = max(1, N // 15)
    centers = rnd(P, ncl, 2) * torch.tensor([CFG_W, CFG_H], dtype=DT)
    a = torch.randint(0, ncl, (P, N), generator=g)
    pos = torch.gather(centers, 1, a.unsqueeze(-1).expand(-1, -1, 2))
    pos = pos + torch.randn(P, N, 2, generator=g, dtype=DT) * 45.0
    pos[..., 0].clamp_(0, CFG_W); pos[..., 1].clamp_(0, CFG_H)
    ang = rnd(P, ncl, 1) * 2 * math.pi
    cvel = torch.cat([torch.cos(ang), torch.sin(ang)], -1) * (2 + 3 * rnd(P, ncl, 1))
    vel = torch.gather(cvel, 1, a.unsqueeze(-1).expand(-1, -1, 2))
    vel = vel + torch.randn(P, N, 2, generator=g, dtype=DT) * 0.8
    ppos = rnd(P, 2) * torch.tensor([CFG_W, CFG_H], dtype=DT)
    pang = rnd(P) * 2 * math.pi
    pvel = torch.stack([torch.cos(pang), torch.sin(pang)], -1) * 2.0
    psize = torch.full((P,), BASE_SIZE, dtype=DT)
    return {k: v.to(dev) for k, v in
            dict(pos=pos, vel=vel, ppos=ppos, pvel=pvel, psize=psize).items()}


# ---- parity vs prod's rolloutFlatState (structure validation) ------------------
def parity(args, dev):
    with open(args.parity) as f:
        d = json.load(f)
    global CFG_W, CFG_H
    CFG_W, CFG_H = float(d['W']), float(d['Hc'])
    s = d['snap']
    n = len(s['bx'])
    t = lambda x: torch.tensor(x, dtype=DT, device=dev)
    ro = Rollout(1, n, dev, W=CFG_W, Hc=CFG_H)
    pos = torch.stack([t(s['bx']), t(s['by'])], -1).unsqueeze(0)
    vel = torch.stack([t(s['bvx']), t(s['bvy'])], -1).unsqueeze(0)
    ro.reset(pos, vel, t([[s['px'], s['py']]]), t([[s['pvx'], s['pvy']]]),
             t([s['psize']]), t([[d['tx'], d['ty']]]))
    ro.run(d['H'])
    cat_t = int(ro.catches.item())
    al = ro.alive[0] > 0.5
    mine_x = ro.pos[0, al, 0].cpu().numpy()
    mine_y = ro.pos[0, al, 1].cpu().numpy()
    term = d['term']
    print(f"catches: js={d['catches']} torch={cat_t}  "
          f"alive: js={len(term['bx'])} torch={int(al.sum())}")
    if len(term['bx']) == int(al.sum()):
        import numpy as np
        dx = np.abs(mine_x - np.array(term['bx']))
        dy = np.abs(mine_y - np.array(term['by']))
        print(f"max |dpos| over alive boids: {max(dx.max(), dy.max()):.3e}")
    pp = ro.ppos[0].cpu().tolist()
    print(f"pred: js=({term['px']:.9f},{term['py']:.9f}) "
          f"torch=({pp[0]:.9f},{pp[1]:.9f}) "
          f"|d|={math.hypot(pp[0]-term['px'], pp[1]-term['py']):.3e}")
    if 'plan' in d:
        net = load_net(args.net, dev)
        ro2 = Rollout(K_ROLL, n, dev, W=CFG_W, Hc=CFG_H)
        st = dict(pos=pos, vel=vel, ppos=t([[s['px'], s['py']]]),
                  pvel=t([[s['pvx'], s['pvy']]]), psize=t([s['psize']]))
        tg = plan_batch(st, net, ro2)[0].cpu().tolist()
        print(f"plan target: js=({d['plan']['x']:.6f},{d['plan']['y']:.6f}) "
              f"torch=({tg[0]:.6f},{tg[1]:.6f}) "
              f"|d|={math.hypot(tg[0]-d['plan']['x'], tg[1]-d['plan']['y']):.3e}")
    if 'cands' in d:
        net = load_net(args.net, dev)
        alive = torch.ones(1, n, dtype=DT, device=dev)
        ppos = t([[s['px'], s['py']]]); pvel = t([[s['pvx'], s['pvy']]])
        psz = t([s['psize']])
        cands_t = candidates(pos, vel, alive, ppos)
        cands_js = t([[c['x'], c['y']] for c in d['cands']]).unsqueeze(0)
        dc = (cands_t - cands_js).abs().max()
        feat_t, ctx_t = cp_features(pos, vel, alive, ppos, pvel, psz, cands_js)
        df = (feat_t[0] - t(d['feat'])).abs().max()
        dctx = (ctx_t[0] - t(d['ctx'])).abs().max()
        v_t = cp_value(net, feat_t, ctx_t)
        dv = (v_t[0] - t(d['vprior'])).abs().max()
        ps = feat_t[0, :, 18] - feat_t[0, :, 16]
        top4 = ps.topk(K_ROLL).indices.cpu().tolist()
        print(f"phase A: max|dcand|={dc:.3e} max|dfeat|={df:.3e} "
              f"max|dctx|={dctx:.3e} max|dV|={dv:.3e}")
        print(f"top4 rolled: js={d['pidx4']} torch={top4}")


# ---- benchmark ------------------------------------------------------------------
def bench(args, dev):
    net = load_net(args.net, dev)
    results = []
    for N in [int(x) for x in args.Ns.split(',')]:
        for B in [int(x) for x in args.Bs.split(',')]:
            P = B // K_ROLL
            row = dict(N=N, B=B, P=P, graphs=bool(args.graphs))
            try:
                st = gen_states(P, N, dev, seed=1234)
                ro = Rollout(B, N, dev)
                if args.graphs and dev.type == 'cuda':
                    # capture needs live buffers; prime with real data first
                    rep = lambda x: x.repeat_interleave(K_ROLL, dim=0)
                    ro.reset(rep(st['pos']), rep(st['vel']), rep(st['ppos']),
                             rep(st['pvel']), rep(st['psize']),
                             torch.zeros(B, 2, dtype=DT, device=dev))
                    t0 = time.perf_counter()
                    ro.capture()
                    row['capture_s'] = round(time.perf_counter() - t0, 2)
                # warmup (also JIT-warms eager kernels)
                plan_batch(st, net, ro)
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
                # calibrate iterations to ~target_sec
                t0 = time.perf_counter()
                plan_batch(st, net, ro)
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
                once = time.perf_counter() - t0
                iters = max(args.min_iters, int(math.ceil(args.target_sec / max(once, 1e-3))))
                t0 = time.perf_counter()
                for _ in range(iters):
                    out = plan_batch(st, net, ro)
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                row.update(iters=iters, sec_per_batch=round(dt / iters, 4),
                           plans_per_sec=round(P * iters / dt, 2),
                           mean_catches=round(float(ro.catches.mean()), 3))
                if dev.type == 'cuda':
                    row['gpu_mem_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    torch.cuda.reset_peak_memory_stats()
            except torch.cuda.OutOfMemoryError:
                row['error'] = 'OOM'
                torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
            results.append(row)
            del ro
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(dict(device=str(dev), torch=torch.__version__,
                           gpu=(torch.cuda.get_device_name(0) if dev.type == 'cuda' else None),
                           results=results), f, indent=1)
    best = {}
    for r in results:
        if 'plans_per_sec' in r:
            n = r['N']
            if n not in best or r['plans_per_sec'] > best[n]['plans_per_sec']:
                best[n] = r
    for n, r in sorted(best.items()):
        print(f"BEST N={n}: {r['plans_per_sec']} plans/s (B={r['B']}, graphs={r['graphs']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--Ns', default='120,30')
    ap.add_argument('--Bs', default='256,1024,4096,16384')
    ap.add_argument('--graphs', type=int, default=1)
    ap.add_argument('--target-sec', type=float, default=8.0)
    ap.add_argument('--min-iters', type=int, default=3)
    ap.add_argument('--net', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--parity', default=None)
    ap.add_argument('--W', type=float, default=None)
    ap.add_argument('--H', type=float, default=None)
    args = ap.parse_args()
    global CFG_W, CFG_H
    if args.W: CFG_W = args.W
    if args.H: CFG_H = args.H
    if args.net is None:
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        for c in [os.path.join(here, 'value_net.json'),
                  os.path.join(here, '..', '..', '..', 'js', 'value_net.json')]:
            if os.path.exists(c):
                args.net = c
                break
    dev = torch.device(args.device)
    torch.set_grad_enabled(False)
    if args.parity:
        parity(args, dev)
    else:
        bench(args, dev)


if __name__ == '__main__':
    main()
