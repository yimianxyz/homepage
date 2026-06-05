// Cheap ballistic deploy policy — browser port of feat_planner.run_value_lookahead_cheap
// (prune_by='ball', K_roll=1). Pure functions so the same code runs under Node
// (parity test vs dev/js_fixture.json) and in predator_planner_worker.js.
//
// Decision (every D frames): generate 16 candidates (cand0=E3D patrol, cand1..15
// = nearest boids lead-adjusted); compute 19 pursuit features + 4 ctx per
// candidate; the tiny value net gives a prior V; pick the single most
// ballistically-catchable candidate, roll ONLY it Hs frames, score it
// (rollout catches + terminal value bootstrap); the other 15 keep their prior V;
// argmax -> commit. Ballistic selection makes 1 rollout ~= the full 16-rollout
// planner (sim_torch: 14.97 vs 15.23 ceiling).
'use strict';

// ---- constants (mirror feat_planner.py / sim_torch) ----
var CP_PS = 200.0, CP_VS = 6.0, CP_RHO = 70.0, CP_HB = 90;
var CP_INIT_N = 120;                 // initial boid count (for frac_alive)

// erf via Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7) -> exact-GELU parity.
function cp_erf(x) {
    var s = x < 0 ? -1 : 1; x = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * x);
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
}
function cp_gelu(x) { return 0.5 * x * (1 + cp_erf(x / Math.SQRT2)); }

// 2-body pursuit: predator (max-speed seek, no flock) vs constant-velocity boid.
// Returns {tCatchNorm, minDist, caught}. Mirrors _ballistic_intercept.
function cp_ballistic(px, py, vx, vy, bx, by, bvx, bvy, catchD, pmaxS, pmaxF, Hb) {
    Hb = Hb || CP_HB;
    var caught = false, tCatch = Hb, mind = Infinity;
    for (var t = 0; t < Hb; t++) {
        var dx = bx - px, dy = by - py;
        var d = Math.sqrt(dx * dx + dy * dy); if (d < 1e-6) d = 1e-6;
        if (d < mind) mind = d;
        if (!caught && d < catchD) { caught = true; tCatch = t; }
        var desx = dx / d * pmaxS - vx, desy = dy / d * pmaxS - vy;
        var sm = Math.sqrt(desx * desx + desy * desy); if (sm < 1e-6) sm = 1e-6;
        var sc = pmaxF / sm; if (sc > 1) sc = 1;
        vx += desx * sc; vy += desy * sc;
        var spd = Math.sqrt(vx * vx + vy * vy); if (spd < 1e-6) spd = 1e-6;
        var vsc = pmaxS / spd; if (vsc > 1) vsc = 1;
        vx *= vsc; vy *= vsc;
        px += vx; py += vy;
        bx += bvx; by += bvy;
    }
    return { tCatchNorm: tCatch / Hb, minDist: mind, caught: caught ? 1.0 : 0.0 };
}

// Per-candidate 19 features + shared 4 ctx, from a flat state.
// state: {px,py,pvx,pvy,psize, bx[],by[],bvx[],bvy[], nAlive}; cands: [{x,y}..](16)
// pmaxS/pmaxF: predator max speed/force. Returns {feat:[K][19], ctx:[4]}.
function cp_features(state, cands, pmaxS, pmaxF) {
    var px = state.px, py = state.py, pvx = state.pvx, pvy = state.pvy;
    var bx = state.bx, by = state.by, bvx = state.bvx, bvy = state.bvy;
    var n = bx.length, K = cands.length;
    var catchD = state.psize * 0.7;
    var feat = new Array(K);
    for (var k = 0; k < K; k++) {
        var cx = cands[k].x, cy = cands[k].y;
        var rx = cx - px, ry = cy - py;
        var dist = Math.sqrt(rx * rx + ry * ry); if (dist < 1e-6) dist = 1e-6;
        var tgo = dist / pmaxS;
        var isE3d = k === 0 ? 1.0 : 0.0;
        // nearest alive boid to the candidate point + local density / mean vel
        var best = Infinity, nb = 0, dens = 0, mvx = 0, mvy = 0, nNear = 0;
        var rho2 = CP_RHO * CP_RHO;
        for (var i = 0; i < n; i++) {
            var ddx = cx - bx[i], ddy = cy - by[i];
            var c2 = ddx * ddx + ddy * ddy;
            if (c2 < best) { best = c2; nb = i; }
            if (c2 < rho2) { dens++; mvx += bvx[i]; mvy += bvy[i]; nNear++; }
        }
        var tbDistCand = Math.sqrt(best < 0 ? 0 : best);
        var tbx = bx[nb], tby = by[nb], tbvx = bvx[nb], tbvy = bvy[nb];
        var tbrx = tbx - px, tbry = tby - py;
        var rangepb = Math.sqrt(tbrx * tbrx + tbry * tbry); if (rangepb < 1e-6) rangepb = 1e-6;
        var relvx = tbvx - pvx, relvy = tbvy - pvy;
        var closing = -(tbrx * relvx + tbry * relvy) / rangepb;
        var losRate = (tbrx * relvy - tbry * relvx) / (rangepb * rangepb);
        var nsafe = nNear > 0 ? nNear : 1.0;
        var mnx = mvx / nsafe, mny = mvy / nsafe;
        var ux = rx / dist, uy = ry / dist;
        var fleeAlign = mnx * ux + mny * uy;
        var b = cp_ballistic(px, py, pvx, pvy, tbx, tby, tbvx, tbvy, catchD, pmaxS, pmaxF, CP_HB);
        feat[k] = [
            rx / CP_PS, ry / CP_PS, dist / CP_PS, isE3d,
            tgo / 60.0,
            tbrx / CP_PS, tbry / CP_PS, tbvx / CP_VS, tbvy / CP_VS,
            tbDistCand / CP_PS,
            rangepb / CP_PS, closing / CP_VS, losRate * 50.0,
            dens / 20.0, fleeAlign / CP_VS,
            (rangepb - dist) / CP_PS,
            b.tCatchNorm, b.minDist / CP_PS, b.caught,
        ];
    }
    var fracAlive = state.nAlive / CP_INIT_N;
    var ctx = [pvx / CP_VS, pvy / CP_VS, fracAlive, state.psize / 20.0];
    return { feat: feat, ctx: ctx };
}

// Value net forward: standardize, 23->48->48->1 with GELU. net = value_net.json.
// Returns V[K].
function cp_value(net, feat, ctx) {
    var K = feat.length, V = new Array(K);
    var L0 = net.layers[0], L1 = net.layers[1], L2 = net.layers[2];
    var cs = [(ctx[0] - net.xmu[0]) / net.xsd[0], (ctx[1] - net.xmu[1]) / net.xsd[1],
              (ctx[2] - net.xmu[2]) / net.xsd[2], (ctx[3] - net.xmu[3]) / net.xsd[3]];
    for (var k = 0; k < K; k++) {
        var x = new Array(23);
        for (var i = 0; i < 19; i++) x[i] = (feat[k][i] - net.fmu[i]) / net.fsd[i];
        x[19] = cs[0]; x[20] = cs[1]; x[21] = cs[2]; x[22] = cs[3];
        var h0 = new Array(48);
        for (var j = 0; j < 48; j++) { var a = L0.b[j], wr = L0.w[j]; for (var q = 0; q < 23; q++) a += wr[q] * x[q]; h0[j] = cp_gelu(a); }
        var h1 = new Array(48);
        for (j = 0; j < 48; j++) { a = L1.b[j]; wr = L1.w[j]; for (q = 0; q < 48; q++) a += wr[q] * h0[q]; h1[j] = cp_gelu(a); }
        var o = L2.b[0], w2 = L2.w[0]; for (j = 0; j < 48; j++) o += w2[j] * h1[j];
        V[k] = o;
    }
    return V;
}

// ballistic pscore = caught - tCatchNorm  (feat[18] - feat[16]); argmax = top1.
function cp_top1(feat) {
    var best = -Infinity, bi = 0;
    for (var k = 0; k < feat.length; k++) {
        var ps = feat[k][18] - feat[k][16];
        if (ps > best) { best = ps; bi = k; }
    }
    return bi;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { cp_features: cp_features, cp_value: cp_value, cp_ballistic: cp_ballistic,
        cp_top1: cp_top1, cp_gelu: cp_gelu, cp_erf: cp_erf, CP: { PS: CP_PS, VS: CP_VS, RHO: CP_RHO, HB: CP_HB } };
}
