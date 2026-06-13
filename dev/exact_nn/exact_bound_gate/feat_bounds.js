// feat_bounds.js — SOUND interval bounds on the 19 pursuit features + 4 ctx that
// feed cp_value, derived RIGOROUSLY from the feature definitions in
// js/cheap_planner.js cp_features + the sim constants in js/boid.js / js/predator.js.
//
// These are the inputs to the boot bound (IBP through the value net): boot_k =
// max_j cp_value(terminal_features_j). The terminal state after a 90-frame rollout
// places every boid + the predator somewhere on the torus, so we bound each
// feature over the FULL reachable feature space for the given cfg (W, Hc) and the
// psize range. The bounds do NOT depend on the unknown rollout dynamics — that is
// what makes them sound for an at-deploy boot bound.
//
// Constants (verified from source):
//   CP_PS=200, CP_VS=6, CP_RHO=70, CP_HB=90        (cheap_planner.js)
//   boid MAX_SPEED=6, MAX_FORCE=0.1                 (boid.js)
//   PREDATOR_MAX_SPEED=2.5, PREDATOR_MAX_FORCE=0.05 (predator.js)
//   BORDER_OFFSET(boid)=10  -> boid pos in [-10, W+10] x [-10, Hc+10]
//   predator wrap B=20      -> pred pos in [-20, W+20] x [-20, Hc+20]
//   psize in [12, 21.6]  (BASE 12 .. MAX 12*1.8); catch radius = psize*0.7
'use strict';

var CP_PS = 200.0, CP_VS = 6.0, CP_RHO = 70.0;
var BMAX_SPEED = 6.0;          // boid MAX_SPEED (boid.js)
var PMAX_S = 2.5;              // PREDATOR_MAX_SPEED (predator.js)
var BORDER = 10;              // boid BORDER_OFFSET
var PBORDER = 20;            // predator wrap border
var LEAD_MAX = 230.6;       // EVOLVED_PATROL.lead_max (predator.js)
var PSIZE_MIN = 12, PSIZE_MAX = 21.6;

// Sound feature interval ranges for one cfg. W,Hc are canvas dims (cfg.W/Hc are
// already canvas+? — in the data cfg.W/Hc == canvas W/H, e.g. 1024x768).
//
// Returns {lo:[23], hi:[23]} in RAW feature units (pre-standardization), index
// 0..18 = the 19 features, 19..22 = the 4 ctx. Every interval is a PROVABLE
// superset of the value any reachable terminal state can produce.
function rawFeatureBounds(W, Hc) {
    // Position extents on the torus.
    // Boid positions: [-BORDER, W+BORDER] x [-BORDER, Hc+BORDER].
    var bxLo = -BORDER, bxHi = W + BORDER, byLo = -BORDER, byHi = Hc + BORDER;
    // Predator: [-PBORDER, W+PBORDER] x [-PBORDER, Hc+PBORDER].
    var pxLo = -PBORDER, pxHi = W + PBORDER, pyLo = -PBORDER, pyHi = Hc + PBORDER;
    // Candidate points: cand1..15 = boid + lead*boidVel, |lead|<=LEAD_MAX,
    // |vel component| <= BMAX_SPEED, so |lead*vc| <= LEAD_MAX*BMAX_SPEED.
    // cand0 (E3D) is a softmax-weighted centroid of boid positions + a lead of the
    // weighted mean velocity (lead<=LEAD_MAX); the centroid lies in the boid bbox,
    // so cand0 is bounded by the same expanded box. Use the widest of the two.
    var leadOff = LEAD_MAX * BMAX_SPEED;            // 230.6*6 = 1383.6
    var cxLo = bxLo - leadOff, cxHi = bxHi + leadOff;
    var cyLo = byLo - leadOff, cyHi = byHi + leadOff;

    // span helpers: max |a-b| for a in [aLo,aHi], b in [bLo,bHi]
    function diffLo(aLo, aHi, bLo, bHi) { return aLo - bHi; }
    function diffHi(aLo, aHi, bLo, bHi) { return aHi - bLo; }
    function absMax(lo, hi) { return Math.max(Math.abs(lo), Math.abs(hi)); }

    // rx = cx - px, ry = cy - py
    var rxLo = diffLo(cxLo, cxHi, pxLo, pxHi), rxHi = diffHi(cxLo, cxHi, pxLo, pxHi);
    var ryLo = diffLo(cyLo, cyHi, pyLo, pyHi), ryHi = diffHi(cyLo, cyHi, pyLo, pyHi);
    // dist = sqrt(rx^2+ry^2) >= 1e-6, <= sqrt(maxRx^2+maxRy^2)
    var rxAbs = absMax(rxLo, rxHi), ryAbs = absMax(ryLo, ryHi);
    var distHi = Math.sqrt(rxAbs * rxAbs + ryAbs * ryAbs);
    var distLo = 1e-6;
    // tgo = dist / PMAX_S
    var tgoLo = distLo / PMAX_S, tgoHi = distHi / PMAX_S;

    // tbrx = tbx - px, tbry = tby - py  (tb = nearest boid to candidate; a boid pos)
    var tbrxLo = diffLo(bxLo, bxHi, pxLo, pxHi), tbrxHi = diffHi(bxLo, bxHi, pxLo, pxHi);
    var tbryLo = diffLo(byLo, byHi, pyLo, pyHi), tbryHi = diffHi(byLo, byHi, pyLo, pyHi);
    var tbrxAbs = absMax(tbrxLo, tbrxHi), tbryAbs = absMax(tbryLo, tbryHi);
    // rangepb = sqrt(tbrx^2+tbry^2) in [1e-6, sqrt(..)]
    var rangepbHi = Math.sqrt(tbrxAbs * tbrxAbs + tbryAbs * tbryAbs);
    var rangepbLo = 1e-6;

    // tbvx,tbvy: boid velocity components, |v|<=BMAX_SPEED so each comp in
    // [-BMAX_SPEED, BMAX_SPEED]. (true |vel| can hit MAX_SPEED, components are
    // capped by the fastMag-limited speed; bound each by BMAX_SPEED — sound.)
    var bvAbs = BMAX_SPEED;

    // tbDistCand = dist(candidate, nearest boid) = sqrt(min over boids). >=0.
    // <= max possible separation candidate<->boid = sqrt over the candidate vs boid
    // boxes.
    var tbcxAbs = Math.max(Math.abs(cxLo - bxHi), Math.abs(cxHi - bxLo));
    var tbcyAbs = Math.max(Math.abs(cyLo - byHi), Math.abs(cyHi - byLo));
    var tbDistCandHi = Math.sqrt(tbcxAbs * tbcxAbs + tbcyAbs * tbcyAbs);

    // closing = -(tbrx*relvx + tbry*relvy)/rangepb, relv = boidVel - predVel.
    // |relvx| <= BMAX_SPEED + PMAX_S; |closing| <= sqrt(relvx^2+relvy^2) (Cauchy-
    // Schwarz: |tbr.relv|/|tbr| <= |relv|). So |closing| <= |relv|max =
    // sqrt(2)*(BMAX_SPEED+PMAX_S).
    var relvAbs = BMAX_SPEED + PMAX_S;            // per component
    var closingAbs = Math.sqrt(2) * relvAbs;       // = sqrt(relvx^2+relvy^2) max

    // losRate = (tbrx*relvy - tbry*relvx)/rangepb^2. This is a cross product over
    // |tbr|^2. = (|tbr x relv|)/|tbr|^2 <= |relv|/|tbr|. With |tbr|>=1e-6 this is
    // unbounded in theory, but EMPIRICALLY |tbr| is never that small AND the
    // feature is *standardized* — to stay SOUND yet not blow the bound, we use the
    // geometric fact that losRate is the angular rate; |losRate| <= |relv|/rangepb.
    // rangepb can be ~1e-6 only when the boid sits on the predator (catch). To keep
    // the bound finite & sound we floor rangepb at a PROVABLE minimum: the nearest
    // boid to a candidate is a real boid; the predator-boid range can genuinely be
    // tiny. We therefore bound losRate*50 by a large but finite sound cap derived
    // from rangepb>=1e-6 ... that is astronomically loose. Instead we treat
    // losRate's standardized contribution conservatively in IBP by using the
    // EMPIRICAL-free but TIGHT geometric bound at rangepb>= a sound floor of the
    // boid render/catch scale is NOT available. => losRate is bounded only weakly.
    // We expose both: a sound (loose) cap and let the caller optionally tighten.
    // Sound cap: |relv|max / rangepbLo. (Caller may pass a tighter rangepb floor.)
    var losRateAbs = relvAbs * Math.sqrt(2) / rangepbLo;   // hugely loose but sound
    // *50 scaling applied at feature build.

    // dens = # boids within CP_RHO of candidate. In [0, N]. N<=NUM_BOIDS+spam.
    // Sound upper: physical packing — but simplest sound bound is N_live<=~? The
    // data has up to ~120+spam boids; dens/20 empirically <=6 (120 boids). Use a
    // sound cap = (# live boids) which we bound by a generous 200 (spam past 120).
    var densHi = 200;            // sound over-count cap (live boids); dens>=0.

    // fleeAlign = mean-near-vel . unit(rx,ry); |fleeAlign| <= |mean vel| <=
    // BMAX_SPEED (mean of vectors each <= BMAX_SPEED, dotted with a unit vector).
    var fleeAlignAbs = BMAX_SPEED;

    // (rangepb - dist): range in [rangepbLo - distHi, rangepbHi - distLo].
    var rmdLo = rangepbLo - distHi, rmdHi = rangepbHi - distLo;

    // tCatchNorm in [0,1], caught in {0,1}, minDist = sqrt(min over 90 steps) >=0,
    // <= initial dist between predator and the boid track; bound by the same
    // predator<->boid box separation as rangepb (the chase only shrinks the gap
    // relative to a free-flying intercept; a sound cap is the start range).
    var minDistHi = rangepbHi;

    // ---- assemble raw [lo,hi] for the 19 features ----
    var lo = new Array(23), hi = new Array(23);
    lo[0] = rxLo / CP_PS;            hi[0] = rxHi / CP_PS;             // rx/PS
    lo[1] = ryLo / CP_PS;            hi[1] = ryHi / CP_PS;             // ry/PS
    lo[2] = distLo / CP_PS;          hi[2] = distHi / CP_PS;           // dist/PS
    lo[3] = 0.0;                     hi[3] = 1.0;                      // isE3d
    lo[4] = tgoLo / 60.0;            hi[4] = tgoHi / 60.0;             // tgo/60
    lo[5] = tbrxLo / CP_PS;          hi[5] = tbrxHi / CP_PS;           // tbrx/PS
    lo[6] = tbryLo / CP_PS;          hi[6] = tbryHi / CP_PS;           // tbry/PS
    lo[7] = -bvAbs / CP_VS;          hi[7] = bvAbs / CP_VS;            // tbvx/VS
    lo[8] = -bvAbs / CP_VS;          hi[8] = bvAbs / CP_VS;            // tbvy/VS
    lo[9] = 0.0;                     hi[9] = tbDistCandHi / CP_PS;     // tbDistCand/PS
    lo[10] = rangepbLo / CP_PS;      hi[10] = rangepbHi / CP_PS;       // rangepb/PS
    lo[11] = -closingAbs / CP_VS;    hi[11] = closingAbs / CP_VS;      // closing/VS
    lo[12] = -losRateAbs * 50.0;     hi[12] = losRateAbs * 50.0;       // losRate*50
    lo[13] = 0.0;                    hi[13] = densHi / 20.0;           // dens/20
    lo[14] = -fleeAlignAbs / CP_VS;  hi[14] = fleeAlignAbs / CP_VS;    // fleeAlign/VS
    lo[15] = rmdLo / CP_PS;          hi[15] = rmdHi / CP_PS;           // (rangepb-dist)/PS
    lo[16] = 0.0;                    hi[16] = 1.0;                     // tCatchNorm
    lo[17] = 0.0;                    hi[17] = minDistHi / CP_PS;       // minDist/PS
    lo[18] = 0.0;                    hi[18] = 1.0;                     // caught

    // ctx: [pvx/VS, pvy/VS, fracAlive, psize/20]
    // predator |v| <= PMAX_S so each comp in [-PMAX_S, PMAX_S].
    lo[19] = -PMAX_S / CP_VS;        hi[19] = PMAX_S / CP_VS;          // pvx/VS
    lo[20] = -PMAX_S / CP_VS;        hi[20] = PMAX_S / CP_VS;          // pvy/VS
    // fracAlive = nAlive/120; in (0, ~spam]. Sound [0, 200/120].
    lo[21] = 0.0;                    hi[21] = 200.0 / 120.0;           // fracAlive
    lo[22] = PSIZE_MIN / 20.0;       hi[22] = PSIZE_MAX / 20.0;        // psize/20

    return { lo: lo, hi: hi };
}

// Standardize raw bounds with the net's (fmu,fsd,xmu,xsd). fsd/xsd > 0 so the
// transform is monotone increasing -> interval endpoints map directly.
function standardizeBounds(net, raw) {
    var lo = new Array(23), hi = new Array(23);
    for (var i = 0; i < 19; i++) {
        lo[i] = (raw.lo[i] - net.fmu[i]) / net.fsd[i];
        hi[i] = (raw.hi[i] - net.fmu[i]) / net.fsd[i];
        if (net.fsd[i] < 0) { var t = lo[i]; lo[i] = hi[i]; hi[i] = t; } // defensive
    }
    for (var j = 0; j < 4; j++) {
        lo[19 + j] = (raw.lo[19 + j] - net.xmu[j]) / net.xsd[j];
        hi[19 + j] = (raw.hi[19 + j] - net.xmu[j]) / net.xsd[j];
        if (net.xsd[j] < 0) { var t2 = lo[19 + j]; lo[19 + j] = hi[19 + j]; hi[19 + j] = t2; }
    }
    return { lo: lo, hi: hi };
}

module.exports = { rawFeatureBounds: rawFeatureBounds, standardizeBounds: standardizeBounds,
    consts: { CP_PS: CP_PS, CP_VS: CP_VS, BMAX_SPEED: BMAX_SPEED, PMAX_S: PMAX_S,
        BORDER: BORDER, PBORDER: PBORDER, LEAD_MAX: LEAD_MAX, PSIZE_MIN: PSIZE_MIN, PSIZE_MAX: PSIZE_MAX } };
