// candidates/moe.js — the PHASE-2 pure single MoE-NN policy as a diff_harness/
// verdict candidate. The unification of l1h.js (planner) + l1e.js (endgame),
// with the defining Phase-2 change: **NO FALLBACK.** The NN's argmax IS the
// committed decision in EVERY boid case (N≥1) — we never revert to prod's
// deterministic argmax (planner) or argmin-scan (endgame). The deterministic
// structure (candidates(), the K_roll rollouts, intercept()'s scan geometry) is
// retained ONLY as a feature extractor feeding the NN and as the fixed
// decision→force map (steer/aim) — exactly what SPEC_PHASE2_MOE.md §1.4 allows.
//
// Injection (two anchors in a transformed predator_cheap, loaded into the vm
// context with save/restore so the reference prod closure + its debug hook stay
// untouched — same mechanism as l1h/l1e):
//
//  * PLANNER (N>5): inside planCheap, AFTER the K_roll rollouts have run (so the
//    NN sees the *actual* rollout outputs — catch-count + value-net bootstrap —
//    which is what lifts the Phase-1 ~37% rollout-bound ceiling), BEFORE prod's
//    argmax. The gate returns the committed target {x,y}; prod's argmax is
//    skipped. NO fallback: the gate ALWAYS commits (returning null is a
//    test-only escape hatch that falls through to prod's argmax).
//  * ENDGAME (N≤5): inside intercept()'s `if(!egBoid)` commit block, BEFORE
//    prod's argmin scan. The gate returns the committed boid index; prod's
//    argmin/nearest is skipped. NO fallback (idx<0 is the test-only escape).
//
// EXACTNESS NOTE (for the verdict's reading of agreement): once a target/egBoid
// is committed, prod's downstream map (steer for the planner, scan→aim→steer for
// the endgame) is VERBATIM prod. So committed-coords / egBoid-identity agreement
// with prod ⇒ that frame's force is bitwise-identical. S_dec is therefore the
// primary similarity metric; S_force (texture) and S_traj (free-run) corroborate.
//
// MODES (env EXACTNN_MOE_MODE), so ONE adapter both self-validates the harness
// and verifies side-a's real model:
//   nn        — side-a's unified model: loadMoePolicy(weights)(state,cfg)→{slot}.
//               This is the gated MoE (single forward pass, learned gate routes,
//               shared output head); the harness is weights/feature-agnostic.
//   oracle    — commit prod's EXACT pick (planner: argmax of the visible post-
//               rollout score; endgame: exact scan-t argmin via eg_scan.egPick).
//               This is "perfect argmax of the visible scores": S_dec MUST be
//               100% (proves the no-fallback override path commits and the
//               harness measures agreement). Doubles as the ablation CEILING.
//   raw_prior — the ablation FLOOR: commit argmax of vprior ALONE (no rollout)
//               for the planner; argmin of the wrap-aware analytic intercept time
//               (eg_features[12], no NN) for the endgame. Quantifies how much the
//               rollout/scan FEATURE is worth vs a naive prior argmax.
//   perturb   — oracle pick with a deterministic 1-in-`period` flip to a
//               different valid slot (env EXACTNN_MOE_PERTURB = flip fraction).
//               S_dec must drop to ≈1−fraction → proves the metric is calibrated.
//
// The NN-vs-raw-argmax ablation (SPEC §5, honesty) = S_dec(nn) vs S_dec(oracle)
// [ceiling: a perfect argmax of the SAME visible scores] vs S_dec(raw_prior)
// [floor: no rollout / no NN]. A genuine NN lands at/near the oracle ceiling; the
// raw_prior gap shows the feature stage is load-bearing.
//
// Sources (env-overridable so the SAME composition verifies any checkpoint):
//   EXACTNN_MOE_STUDENT → moePolicy module (default ../student/moePolicy.js)
//   EXACTNN_MOE_WEIGHTS → weights json    (default ../student/moe_weights.json)
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const DEBUG_ANCHOR = '    window.__cheap = {';
// planner anchors (js/predator_cheap.js planCheap)
const SCORE_INIT_ANCHOR = '        var score = vprior.slice();';
const ROLL_CAPTURE_ANCHOR = '            score[ci] = rr.catches + boot;';
const PLANNER_ARGMAX_ANCHOR = '        var bi = 0, bs = -Infinity;';
// endgame anchor (intercept, identical to l1e.js)
const ENDGAME_COMMIT_ANCHOR = '        if (!egBoid) {\n            var bestT = Infinity, i;';

function debugInject(code) {
    return code.replace(DEBUG_ANCHOR,
        '    window.__cheapDebug = {\n'
        + '        get: function () { return { target: target, frame: frame, egBoid: egBoid, configured: configured, cfgW: cfg.W, cfgHc: cfg.Hc }; },\n'
        + '        set: function (s) { target = s.target; frame = s.frame; egBoid = s.egBoid; configured = s.configured; cfg.W = s.cfgW; cfg.Hc = s.cfgHc; }\n'
        + '    };\n' + DEBUG_ANCHOR);
}
function plannerInject(code) {
    // (1) reset the rolled-capture buffer when a fresh score[] is formed
    code = code.replace(SCORE_INIT_ANCHOR, SCORE_INIT_ANCHOR + '\n        window.__moeRolled = [];');
    // (2) capture each rollout's actual outputs (catch-count + bootstrap) as a feature
    code = code.replace(ROLL_CAPTURE_ANCHOR, ROLL_CAPTURE_ANCHOR
        + '\n            window.__moeRolled.push({ ci: ci, catches: rr.catches, boot: boot });');
    // (3) consult the unified NN BEFORE prod's argmax; a committed target skips it.
    // pass pidx (prod's roll order) so the gate can supply side-a's exact planRecord.
    code = code.replace(PLANNER_ARGMAX_ANCHOR,
        '        if (window.__moeGatePlanner) { var __mt = window.__moeGatePlanner(s, cands, fr, vprior, score, window.__moeRolled, pidx); if (__mt) return __mt; }\n'
        + PLANNER_ARGMAX_ANCHOR);
    return code;
}
function endgameInject(code) {
    // pass pred.currentSize so the endgame gate feeds side-a's gateFeat the real psize
    return code.replace(ENDGAME_COMMIT_ANCHOR,
        '        if (!egBoid && window.__moeGateEndgame) { var __egi = window.__moeGateEndgame(px, py, boids, cfg.W, cfg.Hc, pred.currentSize); if (__egi >= 0) egBoid = boids[__egi]; }\n'
        + ENDGAME_COMMIT_ANCHOR);
}

// bitwise float64 key for coord-dedup (matches diff_harness/l1h dedup convention)
const _ck = new DataView(new ArrayBuffer(16));
function ckey(x, y) { _ck.setFloat64(0, x); _ck.setFloat64(8, y); return _ck.getBigUint64(0) + ':' + _ck.getBigUint64(8); }
// deterministic state→[0,1) hash for the perturb self-test: flips a fixed
// FRACTION of commits independent of game length / call order / sharding (a
// per-game counter undershoots in short endgames). FNV-1a over the f64 bits.
function uhash(x, y) {
    _ck.setFloat64(0, x); _ck.setFloat64(8, y);
    let h = 2166136261 >>> 0;
    for (let i = 0; i < 16; i++) { h ^= _ck.getUint8(i); h = Math.imul(h, 16777619) >>> 0; }
    return h / 4294967296;
}

// planner argmax over a score[] with lowest-index strict-greater tiebreak —
// BYTE-EQUIVALENT to prod's `for(k){ if(score[k]>bs){bs=score[k];bi=k;} }`.
function argmaxScore(score) { let bi = 0, bs = -Infinity; for (let k = 0; k < score.length; k++) if (score[k] > bs) { bs = score[k]; bi = k; } return bi; }

module.exports.create = async function (game, helpers) {
    const policyDir = helpers.policyDir;
    const mode = process.env.EXACTNN_MOE_MODE || 'nn';
    const perturbFrac = process.env.EXACTNN_MOE_PERTURB != null ? +process.env.EXACTNN_MOE_PERTURB : 0.1;

    // endgame ablation/oracle helpers (exact scan + analytic prior), loaded lazily
    const egScan = require(path.join(__dirname, '..', 'endgame', 'eg_scan.js'));
    const egFeat = require(path.join(__dirname, '..', 'endgame', 'eg_features.js'));

    // the unified model (nn mode only). side-a's CONFIRMED contract (moePolicy.js):
    //   const P = loadMoePolicy(weights);   // -> { chooseTarget, chooseEgBoid, decide }
    //   PLANNER: P.chooseTarget({s, cands:[[x,y]..16], feat:[16][19], vprior:[16],
    //            pidx:[16], rolled:[4][ci,catches,boot], nAlive}, {W,Hc}) -> {tx,ty,slot}
    //   ENDGAME: P.chooseEgBoid({px,py,bx,by,bvx,bvy,psize}, {W,Hc}) -> {egIdx, margin}
    // Single forward pass internally (learned gate + 2 experts + shared head). The
    // harness feeds prod's exact live structure; side-a's code owns featurization.
    let P = null;
    if (mode === 'nn') {
        const studentMod = process.env.EXACTNN_MOE_STUDENT
            ? path.resolve(process.env.EXACTNN_MOE_STUDENT)
            : path.join(__dirname, '..', 'student', 'moePolicy.js');
        const weightsFp = process.env.EXACTNN_MOE_WEIGHTS
            ? path.resolve(process.env.EXACTNN_MOE_WEIGHTS)
            : path.join(__dirname, '..', 'student', 'moe_weights.json');
        const { loadMoePolicy } = require(studentMod);
        P = loadMoePolicy(weightsFp);
    }

    const stats = { plannerCommits: 0, endgameCommits: 0,
        plannerVsProd: 0, endgameVsProd: 0,   // live NN-vs-prod disagreements (diagnostic)
        flips: 0, malformed: 0 };

    const cfgWH = { W: game.sim.canvasWidth, Hc: game.sim.canvasHeight };
    // ---- PLANNER gate: NO fallback (always returns a committed target) --------
    game.win.__moeGatePlanner = function (s, cands, fr, vprior, score, rolled, pidx) {
        stats.plannerCommits++;
        const prodBi = argmaxScore(score);          // prod's exact pick (for oracle / diagnostics)
        let slot;
        if (mode === 'nn') {
            // side-a's exact planRecord: rolled as [ci,catches,boot] triples (not objects)
            const r = P.chooseTarget({ s: s, cands: cands.map(c => [c.x, c.y]), feat: fr.feat,
                vprior: vprior, pidx: pidx, rolled: rolled.map(o => [o.ci, o.catches, o.boot]),
                nAlive: s.bx.length }, cfgWH);
            slot = (r && typeof r.slot === 'number') ? r.slot : -1;
            // a malformed NN slot is NOT silently corrected to prod (that would hide
            // a model defect / inflate S_dec) — it's PENALIZED as a disagreement.
            if (!(slot >= 0 && slot < cands.length)) { stats.malformed++; slot = altSlotByCoords(cands, prodBi); }
        } else if (mode === 'raw_prior') {
            slot = argmaxScore(vprior);             // prior-only argmax (no rollout) — ablation floor
        } else { // oracle / perturb
            slot = prodBi;
            // hash the COMMITTED candidate coords (diverse across plans; predator
            // start pos is seed-invariant so hashing px,py is degenerate)
            if (mode === 'perturb' && uhash(cands[prodBi].x, cands[prodBi].y) < perturbFrac) { slot = altSlotByCoords(cands, prodBi); if (!sameCoords(cands, slot, prodBi)) stats.flips++; }
        }
        if (slot !== prodBi && !sameCoords(cands, slot, prodBi)) stats.plannerVsProd++;
        return { x: cands[slot].x, y: cands[slot].y };
    };

    // ---- ENDGAME gate: NO fallback (always returns a committed boid index) ----
    game.win.__moeGateEndgame = function (px, py, boids, W, Hc, psize) {
        stats.endgameCommits++;
        const n = boids.length;
        // flat {x,y,vx,vy} — the shape eg_scan.egPick / eg_features expect
        const bs = new Array(n);
        for (let i = 0; i < n; i++) bs[i] = { x: boids[i].position.x, y: boids[i].position.y,
            vx: boids[i].velocity.x, vy: boids[i].velocity.y };
        const prodIdx = egScan.egPick(px, py, bs, W, Hc).egIdx;   // prod's exact egBoid
        let idx;
        if (mode === 'nn') {
            // side-a's chooseEgBoid takes the raw snapshot + computes scan-t itself
            const r = P.chooseEgBoid({ px: px, py: py, bx: bs.map(b => b.x), by: bs.map(b => b.y),
                bvx: bs.map(b => b.vx), bvy: bs.map(b => b.vy), psize: psize }, { W: W, Hc: Hc });
            idx = (r && typeof r.egIdx === 'number') ? r.egIdx : -1;
            // malformed/padded-slot pick (≥n) is PENALIZED as a disagreement, never
            // silently mapped to prod (no hidden fallback; surfaces a real NN failure).
            if (!(idx >= 0 && idx < n)) { stats.malformed++; idx = n >= 2 ? (prodIdx + 1) % n : prodIdx; }
        } else if (mode === 'raw_prior') {
            idx = argminAnalytic(px, py, bs, W, Hc, egFeat);   // wrap-aware analytic argmin (no NN) — ablation floor
        } else { // oracle / perturb
            idx = prodIdx;
            // hash the COMMITTED boid position (diverse across commits, unlike px,py)
            if (mode === 'perturb' && n >= 2 && uhash(bs[prodIdx].x, bs[prodIdx].y) < perturbFrac) { idx = (prodIdx + 1) % n; stats.flips++; }
        }
        if (idx !== prodIdx) stats.endgameVsProd++;
        return idx;
    };

    game.win.__moeStats = stats;
    global.__moeStatsLast = stats;

    // helpers closed over cands
    function sameCoords(cands, a, b) { return cands[a].x === cands[b].x && cands[a].y === cands[b].y; }
    function altSlotByCoords(cands, prodBi) {
        // pick the lowest-index slot whose coords differ from prod's pick (a genuine
        // decision flip; if all 16 coords coincide it's a true no-op → return prodBi)
        const k0 = ckey(cands[prodBi].x, cands[prodBi].y);
        for (let k = 0; k < cands.length; k++) if (ckey(cands[k].x, cands[k].y) !== k0) return k;
        return prodBi;
    }

    // load the transformed predator_cheap (save/restore globals)
    const prevCheap = game.win.__cheap, prevReady = game.win.__predatorReady,
          prevModel = game.win.__predatorModel, prevDbg = game.win.__cheapDebug;
    let code = fs.readFileSync(path.join(policyDir, 'predator_cheap.js'), 'utf8');
    for (const [name, anchor] of [['debug', DEBUG_ANCHOR], ['score-init', SCORE_INIT_ANCHOR],
        ['roll-capture', ROLL_CAPTURE_ANCHOR], ['planner-argmax', PLANNER_ARGMAX_ANCHOR],
        ['endgame-commit', ENDGAME_COMMIT_ANCHOR]]) {
        if (code.indexOf(anchor) < 0) throw new Error('moe: anchor not found: ' + name);
    }
    code = endgameInject(plannerInject(debugInject(code)));
    vm.runInThisContext(code, { filename: 'predator_cheap.js#moe' });
    const moeForce = game.win.__cheap.force, moeDbg = game.win.__cheapDebug, moeReady = game.win.__predatorReady;
    game.win.__cheap = prevCheap; game.win.__predatorReady = prevReady;
    game.win.__predatorModel = prevModel; game.win.__cheapDebug = prevDbg;
    if (moeReady && typeof moeReady.then === 'function') await moeReady;

    return { name: 'MoE(' + mode + (mode === 'perturb' ? ',p=' + perturbFrac : '') + ')',
        configure() {}, force: moeForce, reset() {}, _debug: moeDbg,
        stats: () => Object.assign({}, stats) };
};

// wrap-aware analytic intercept-time argmin (the no-NN endgame prior, ~99%).
// feature index 12 of egBoidFeatures = wa0/100 (wrap-aware intercept time).
function argminAnalytic(px, py, bs, W, Hc, egFeat) {
    let idx = 0, best = Infinity, second = Infinity;
    for (let i = 0; i < bs.length; i++) {
        const f = egFeat.egBoidFeatures(px, py, bs[i].x, bs[i].y, bs[i].vx, bs[i].vy, W, Hc);
        const t = f[12];
        if (t < best) { second = best; best = t; idx = i; } else if (t < second) second = t;
    }
    return idx;
}
