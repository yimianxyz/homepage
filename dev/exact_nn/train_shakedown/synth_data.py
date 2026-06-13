#!/usr/bin/env python3
"""THROWAWAY stand-in dataset generator for the EXACT-NN training-pipeline
shakedown (SPEC.md section 6.4 schema). The real dataset is oracle-logged plan
decisions from prod (instrumented predator_cheap.js fork); this generator makes
random-but-plausible states and computes every derived field with faithful
numpy ports of the prod JS *where the field is an input* (candidates,
cp_features, cp_value/vprior, pidx) so tensor shapes, dtypes, value ranges and
tie structure (E3D padding duplicates!) match reality. The 4 ROLLED SCORES are
JUNK (no rollout is run) -- they are labels and labels are throwaway here.
Every shard header and record carries throwaway=true / oracle_sha=SYNTH-STANDIN.

Schema per record (one plan decision), built from js/predator_cheap.js
planCheap() + js/cheap_planner.js cp_features/cp_value:
  seed         game id (stand-in for the game seed; split key)
  plan_idx     plan index within the game
  cfg          {W, Hc, PREDATOR_RANGE, NUM_BOIDS}      derived-config vector
  pred         {x, y, vx, vy, size}                    size = currentSize
  boids        [[x,y,vx,vy] * n]   n = alive count, 6..120 (plan regime N>5)
  cands        [[x,y] * 16]        cand0 = E3D patrol, 1..15 nearest boids
                                   lead-adjusted, slots k>=n = E3D copies
  cand_src     [[kind, boid_idx] * 16]  provenance: kind 0=e3d 1=boid 2=pad;
                                   boid_idx = index into boids, -1 otherwise
  feat         [[19 floats] * 16]  cp_features per-candidate features
  ctx          [4 floats]          cp_features shared context
  vprior       [16 floats]         cp_value(value_net, feat, ctx)
  pidx         [4 ints]            rolled candidate indices (pscore desc,
                                   tie -> lowest index; pscore=f18-f16)
  rolled       [4 floats]          JUNK rolled scores (catches+bootstrap shape)
  score        [16 floats]         vprior with rolled scores at pidx
  bi           int                 argmax(score), first max (= JS strict >)

Shard = JSONL.gz; line 0 is the shard header (SPEC 6.4): {shard_header, W, Hc,
PREDATOR_RANGE, NUM_BOIDS, seed_lo, seed_hi, frameMs, oracle_sha, cert_run_id}.

Usage:
  python3 synth_data.py --out data/ --n 60000 [--seed0 1000] [--shard 8192]
"""
import argparse, gzip, json, os, sys
import numpy as np

# ---- prod constants (js/boid.js, js/predator.js, js/cheap_planner.js) ----
CP_PS, CP_VS, CP_RHO, CP_HB = 200.0, 6.0, 70.0, 90
CP_INIT_N = 120                  # fracAlive denominator (always 120, even mobile)
MAX_SPEED = 6.0                  # boid
PMAX_S, PMAX_F = 2.5, 0.05      # predator max speed / force
LEAD_SCALE, LEAD_MAX = 0.454, 230.6   # EVOLVED_PATROL
BORDER_OFFSET = 10.0
BASE_SIZE, MAX_SIZE = 12.0, 12.0 * 1.8
K = 16

# device matrix (SPEC section 5) + the pre-configure default cell 1680x1680.
# (W, Hc, PREDATOR_RANGE, NUM_BOIDS); mobile-by-UA cells get 60/60.
CELLS = [
    (390.0,  844.0,  60.0, 60),
    (820.0,  1180.0, 60.0, 60),   # iPad: mobile by UA regex
    (1024.0, 768.0,  80.0, 120),
    (1512.0, 982.0,  80.0, 120),
    (1680.0, 1050.0, 80.0, 120),
    (2560.0, 1440.0, 80.0, 120),
    (1680.0, 1680.0, 80.0, 120),  # cfg default before first configure()
]
FRAME_MS = {60: 18, 120: 12}     # REFRESH_INTERVAL_IN_MS mobile/desktop


def cp_erf(x):
    """Abramowitz & Stegun 7.1.26 erf -- same approx as cheap_planner.js."""
    s = np.sign(x); x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def cp_gelu(x):
    return 0.5 * x * (1.0 + cp_erf(x / np.sqrt(2.0)))


def cp_ballistic_batch(px, py, pvx, pvy, bx, by, bvx, bvy, catchD):
    """Vectorized port of cp_ballistic over (B,K) arrays. Returns
    (tCatchNorm, minDist, caught) each (B,K) float64."""
    px, py = np.broadcast_to(px, bx.shape).copy(), np.broadcast_to(py, bx.shape).copy()
    vx, vy = np.broadcast_to(pvx, bx.shape).copy(), np.broadcast_to(pvy, bx.shape).copy()
    bx, by, bvx, bvy = bx.copy(), by.copy(), bvx.copy(), bvy.copy()
    caught = np.zeros(bx.shape, dtype=bool)
    tCatch = np.full(bx.shape, float(CP_HB))
    mind = np.full(bx.shape, np.inf)
    for t in range(CP_HB):
        dx, dy = bx - px, by - py
        d = np.sqrt(dx * dx + dy * dy)
        np.maximum(d, 1e-6, out=d)
        np.minimum(mind, d, out=mind)
        new = (~caught) & (d < catchD)
        tCatch[new] = float(t)
        caught |= new
        desx, desy = dx / d * PMAX_S - vx, dy / d * PMAX_S - vy
        sm = np.sqrt(desx * desx + desy * desy)
        np.maximum(sm, 1e-6, out=sm)
        sc = np.minimum(PMAX_F / sm, 1.0)
        vx += desx * sc; vy += desy * sc
        spd = np.sqrt(vx * vx + vy * vy)
        np.maximum(spd, 1e-6, out=spd)
        vsc = np.minimum(PMAX_S / spd, 1.0)
        vx *= vsc; vy *= vsc
        px += vx; py += vy
        bx += bvx; by += bvy
    return tCatch / CP_HB, mind, caught.astype(np.float64)


def cp_features_batch(px, py, pvx, pvy, psize, bx, by, bvx, bvy, bmask, cx, cy):
    """Vectorized cp_features. px..psize (B,), b* (B,Nmax) with bmask (B,Nmax)
    alive flags, c* (B,K). Returns feat (B,K,19), ctx (B,4)."""
    B = px.shape[0]
    catchD = (psize * 0.7)[:, None]
    rx, ry = cx - px[:, None], cy - py[:, None]
    dist = np.sqrt(rx * rx + ry * ry)
    np.maximum(dist, 1e-6, out=dist)
    tgo = dist / PMAX_S
    isE3d = np.zeros((B, K)); isE3d[:, 0] = 1.0
    # nearest alive boid to each candidate + local density / mean velocity
    ddx = cx[:, :, None] - bx[:, None, :]
    ddy = cy[:, :, None] - by[:, None, :]
    c2 = ddx * ddx + ddy * ddy                       # (B,K,Nmax)
    c2m = np.where(bmask[:, None, :], c2, np.inf)
    nb = np.argmin(c2m, axis=2)                      # (B,K) first min = JS strict <
    best = np.take_along_axis(c2m, nb[:, :, None], axis=2)[:, :, 0]
    tbDistCand = np.sqrt(np.maximum(best, 0.0))
    near = (c2 < CP_RHO * CP_RHO) & bmask[:, None, :]
    dens = near.sum(axis=2).astype(np.float64)
    mvx = np.where(near, bvx[:, None, :], 0.0).sum(axis=2)
    mvy = np.where(near, bvy[:, None, :], 0.0).sum(axis=2)
    nsafe = np.maximum(dens, 1.0)
    mnx, mny = mvx / nsafe, mvy / nsafe
    bidx = np.arange(B)[:, None]
    tbx, tby = bx[bidx, nb], by[bidx, nb]
    tbvx, tbvy = bvx[bidx, nb], bvy[bidx, nb]
    tbrx, tbry = tbx - px[:, None], tby - py[:, None]
    rangepb = np.sqrt(tbrx * tbrx + tbry * tbry)
    np.maximum(rangepb, 1e-6, out=rangepb)
    relvx, relvy = tbvx - pvx[:, None], tbvy - pvy[:, None]
    closing = -(tbrx * relvx + tbry * relvy) / rangepb
    losRate = (tbrx * relvy - tbry * relvx) / (rangepb * rangepb)
    ux, uy = rx / dist, ry / dist
    fleeAlign = mnx * ux + mny * uy
    tcn, mind, cgt = cp_ballistic_batch(px[:, None], py[:, None], pvx[:, None],
                                        pvy[:, None], tbx, tby, tbvx, tbvy, catchD)
    feat = np.stack([
        rx / CP_PS, ry / CP_PS, dist / CP_PS, isE3d,
        tgo / 60.0,
        tbrx / CP_PS, tbry / CP_PS, tbvx / CP_VS, tbvy / CP_VS,
        tbDistCand / CP_PS,
        rangepb / CP_PS, closing / CP_VS, losRate * 50.0,
        dens / 20.0, fleeAlign / CP_VS,
        (rangepb - dist) / CP_PS,
        tcn, mind / CP_PS, cgt,
    ], axis=2)                                       # (B,K,19)
    nAlive = bmask.sum(axis=1).astype(np.float64)
    ctx = np.stack([pvx / CP_VS, pvy / CP_VS, nAlive / CP_INIT_N, psize / 20.0], axis=1)
    return feat, ctx


def cp_value_batch(net, feat, ctx):
    """cp_value over (B,K,19) feat + (B,4) ctx -> vprior (B,K)."""
    fmu, fsd = np.array(net['fmu']), np.array(net['fsd'])
    xmu, xsd = np.array(net['xmu']), np.array(net['xsd'])
    B = feat.shape[0]
    xf = (feat - fmu) / fsd
    xc = np.broadcast_to(((ctx - xmu) / xsd)[:, None, :], (B, K, 4))
    x = np.concatenate([xf, xc], axis=2).reshape(B * K, 23)
    for li, L in enumerate(net['layers']):
        w, b = np.array(L['w']), np.array(L['b'])
        x = x @ w.T + b
        if li < len(net['layers']) - 1:
            x = cp_gelu(x)
    return x.reshape(B, K)


def gen_states(rng, B, W, Hc, NB):
    """Random plausible plan-regime states. Boids in loose clusters with
    shared headings (so density / fleeAlign features are non-degenerate)."""
    lo, hix, hiy = -BORDER_OFFSET, W + BORDER_OFFSET, Hc + BORDER_OFFSET
    # alive count 6..NB, mixture: uniform + bias to full flock (early game)
    n = np.where(rng.random(B) < 0.35, NB,
                 rng.integers(6, NB + 1, size=B)).astype(np.int64)
    Nmax = int(n.max())
    bmask = np.arange(Nmax)[None, :] < n[:, None]
    # clusters
    ncl = 4
    ccx = rng.uniform(lo, hix, (B, ncl)); ccy = rng.uniform(lo, hiy, (B, ncl))
    chd = rng.uniform(0, 2 * np.pi, (B, ncl))
    asg = rng.integers(0, ncl, (B, Nmax))
    sig = rng.uniform(25.0, 140.0, (B, 1))
    bx = np.take_along_axis(ccx, asg, 1) + rng.normal(0, 1, (B, Nmax)) * sig
    by = np.take_along_axis(ccy, asg, 1) + rng.normal(0, 1, (B, Nmax)) * sig
    # a fraction fully uniform (stragglers)
    uni = rng.random((B, Nmax)) < 0.25
    bx = np.where(uni, rng.uniform(lo, hix, (B, Nmax)), bx)
    by = np.where(uni, rng.uniform(lo, hiy, (B, Nmax)), by)
    np.clip(bx, lo, hix, out=bx); np.clip(by, lo, hiy, out=by)
    hd = np.take_along_axis(chd, asg, 1) + rng.normal(0, 0.6, (B, Nmax))
    spd = MAX_SPEED * rng.uniform(0.45, 1.0, (B, Nmax))
    bvx, bvy = spd * np.cos(hd), spd * np.sin(hd)
    bx[~bmask] = 0.0; by[~bmask] = 0.0; bvx[~bmask] = 0.0; bvy[~bmask] = 0.0
    px = rng.uniform(lo, hix, B); py = rng.uniform(lo, hiy, B)
    phd = rng.uniform(0, 2 * np.pi, B); pspd = PMAX_S * rng.random(B)
    pvx, pvy = pspd * np.cos(phd), pspd * np.sin(phd)
    psize = rng.uniform(BASE_SIZE, MAX_SIZE, B)
    return n, bmask, bx, by, bvx, bvy, px, py, pvx, pvy, psize


def candidates_batch(rng, n, bmask, bx, by, bvx, bvy, px, py, W, Hc):
    """Port of predator_cheap.js candidates(): cand0 = E3D (stand-in: random
    field point -- computeEvolvedTarget is not ported; throwaway data), then
    15 nearest boids lead-adjusted, padding slots k>=n with E3D copies."""
    B, Nmax = bx.shape
    e3x = rng.uniform(-BORDER_OFFSET, W + BORDER_OFFSET, B)
    e3y = rng.uniform(-BORDER_OFFSET, Hc + BORDER_OFFSET, B)
    dx, dy = bx - px[:, None], by - py[:, None]
    d2 = np.where(bmask, dx * dx + dy * dy, np.inf)
    order = np.argsort(d2, axis=1, kind='stable')     # JS sort by d2 (no tiebreak)
    cx = np.empty((B, K)); cy = np.empty((B, K))
    kind = np.full((B, K), 2, dtype=np.int64)         # 2=pad
    bref = np.full((B, K), -1, dtype=np.int64)
    cx[:, 0], cy[:, 0] = e3x, e3y
    kind[:, 0] = 0
    bidx = np.arange(B)
    for k in range(K - 1):
        j = order[:, k] if k < Nmax else np.zeros(B, dtype=np.int64)
        have = (k < n)
        jx, jy = bx[bidx, j], by[bidx, j]
        jvx, jvy = bvx[bidx, j], bvy[bidx, j]
        dcent = np.sqrt((jx - px) ** 2 + (jy - py) ** 2)
        lead = np.clip(dcent / PMAX_S * LEAD_SCALE, 0.0, LEAD_MAX)
        cx[:, k + 1] = np.where(have, jx + lead * jvx, e3x)
        cy[:, k + 1] = np.where(have, jy + lead * jvy, e3y)
        kind[:, k + 1] = np.where(have, 1, 2)
        bref[:, k + 1] = np.where(have, j, -1)
    return cx, cy, kind, bref


def make_batch(rng, net, B, cell, game0, plans_per_game):
    W, Hc, R, NB = cell
    n, bmask, bx, by, bvx, bvy, px, py, pvx, pvy, psize = gen_states(rng, B, W, Hc, NB)
    cx, cy, kind, bref = candidates_batch(rng, n, bmask, bx, by, bvx, bvy, px, py, W, Hc)
    feat, ctx = cp_features_batch(px, py, pvx, pvy, psize, bx, by, bvx, bvy, bmask, cx, cy)
    vprior = cp_value_batch(net, feat, ctx)
    # pidx: pscore desc, tie -> lowest index (stable argsort on -pscore)
    pscore = feat[:, :, 18] - feat[:, :, 16]
    pidx = np.argsort(-pscore, axis=1, kind='stable')[:, :4]
    # ---- JUNK rolled scores (throwaway labels; rollout-shaped, NOT a rollout)
    bidx = np.arange(B)[:, None]
    catches = rng.poisson(0.35, (B, 4)).astype(np.float64)
    boot = vprior[bidx, pidx] + rng.normal(0.0, 0.3, (B, 4))
    rolled = catches + boot
    # Inject the prod "NaN -> -Infinity extermination path" (~0.5% of rolled
    # entries): when a rollout kills ALL boids, the terminal bootstrap is
    # -Infinity, so score[ci] = -Infinity. JS JSON.stringify emits null for it;
    # synth mirrors that (see to_jsonable) so the loader's null handling and
    # the finite-masked losses are exercised before real data lands.
    exterm = rng.random((B, 4)) < 0.005
    rolled[exterm] = -np.inf
    score = vprior.copy()
    score[bidx, pidx] = rolled
    bi = np.argmax(score, axis=1)                     # first max = JS strict >

    def js_floats(a):
        """JS JSON.stringify(x) emits null for non-finite numbers."""
        return [v if np.isfinite(v) else None for v in a]

    games = game0 + np.arange(B) // plans_per_game
    plan_idx = np.arange(B) % plans_per_game
    recs = []
    for r in range(B):
        nn_ = int(n[r])
        recs.append({
            'seed': int(games[r]), 'plan_idx': int(plan_idx[r]), 'throwaway': True,
            'cfg': {'W': W, 'Hc': Hc, 'PREDATOR_RANGE': R, 'NUM_BOIDS': NB},
            'pred': {'x': px[r], 'y': py[r], 'vx': pvx[r], 'vy': pvy[r], 'size': psize[r]},
            'boids': np.stack([bx[r, :nn_], by[r, :nn_], bvx[r, :nn_], bvy[r, :nn_]], 1).tolist(),
            'cands': np.stack([cx[r], cy[r]], 1).tolist(),
            'cand_src': np.stack([kind[r], bref[r]], 1).tolist(),
            'feat': feat[r].tolist(), 'ctx': ctx[r].tolist(),
            'vprior': vprior[r].tolist(), 'pidx': pidx[r].tolist(),
            'rolled': js_floats(rolled[r]), 'score': js_floats(score[r]),
            'bi': int(bi[r]),
        })
    return recs, int(games[-1]) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=60000, help='total records')
    ap.add_argument('--seed0', type=int, default=1000, help='first game id')
    ap.add_argument('--shard', type=int, default=8192, help='records per shard')
    ap.add_argument('--plans-per-game', type=int, default=25)
    ap.add_argument('--rng-seed', type=int, default=7)
    ap.add_argument('--net', default=os.path.join(os.path.dirname(__file__), 'value_net.json'))
    args = ap.parse_args()
    net = json.load(open(args.net))
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)
    game = args.seed0
    done, si = 0, 0
    while done < args.n:
        B = min(args.shard, args.n - done)
        cell = CELLS[si % len(CELLS)]
        recs, game_next = make_batch(rng, net, B, cell, game, args.plans_per_game)
        W, Hc, R, NB = cell
        hdr = {'shard_header': True, 'schema_v': 1, 'throwaway': True,
               'W': W, 'Hc': Hc, 'PREDATOR_RANGE': R, 'NUM_BOIDS': NB,
               'seed_lo': game, 'seed_hi': game_next - 1, 'frameMs': FRAME_MS[NB],
               'oracle_sha': 'SYNTH-STANDIN', 'cert_run_id': 'SYNTH-STANDIN'}
        path = os.path.join(args.out, 'synth_%03d_c%dx%d_s%d-%d.jsonl.gz'
                            % (si, int(W), int(Hc), game, game_next - 1))
        with gzip.open(path, 'wt', compresslevel=4) as f:
            f.write(json.dumps(hdr) + '\n')
            for rec in recs:
                f.write(json.dumps(rec) + '\n')
        game = game_next
        done += B; si += 1
        print('shard %s  (%d/%d)' % (os.path.basename(path), done, args.n), flush=True)
    print('done: %d records, %d shards, games %d..%d' % (done, si, args.seed0, game - 1))


if __name__ == '__main__':
    main()
