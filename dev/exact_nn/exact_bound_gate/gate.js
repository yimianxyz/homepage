// gate.js — THE exact cheap-bound gate + all four deliverables, single pass.
//
// Per plan, at DEPLOY we know exactly: the 16 vprior (bitwise reused), the 4
// rolled indices pidx[0..3], the 16 candidate coords, the live boid/predator
// state. We do NOT know the rolled scores. We bound each candidate's true prod
// score and certify the deduped argmax iff it is PROVABLY independent of the
// unknown rolled scores.
//
//   score_k (prod) = vprior_k                      (k NOT rolled)  -> EXACT point
//                  = catches_k + boot_k            (k rolled)      -> interval
//   rolled interval: [0 + Bmin, Cmax_k + Bmax]
//     Cmax_k = catchBound(state)  (same for all rolled k; per-plan)
//     [Bmin,Bmax] = IBP envelope of cp_value over the reachable feature box (cfg)
//
// Dedup (SPEC §3): group candidates by BITWISE-equal (x,y); canonical = lowest
// index; group score interval = [max slo, max shi] over the group (prod takes the
// max score in a tie, lowest index as the coord). Certify the unique winner W iff
//   slo[W] > shi[c]  for every other canonical group c.
// Certified coord = cands[canonical(W)] ; cross-check == prod lab.(tx,ty).
//
// We also compute, with the SAME machinery:
//  (A) sound gate (full IBP boot bound)            -> Deliverable 1
//  (B) losRate-tightened IBP variant               -> Deliverable 3 secondary
//  (C) oracle ceiling (TRUE scores)                -> Deliverable 2
//  (D) tightness breakdown: true-boot / true-catch -> Deliverable 3
'use strict';
var fs = require('fs'), zlib = require('zlib'), readline = require('readline'), struct = null;
var fb = require('./feat_bounds.js');
var ibp = require('./ibp.js');
var cbnd = require('./catch_bound.js');
var net = JSON.parse(fs.readFileSync(__dirname + '/../../../js/value_net.json', 'utf8'));

function hexToF64(h) {
    var b = Buffer.from(h, 'hex');
    return b.readDoubleBE(0);
}

// ---- IBP envelopes (cached per cfg key) ----
var _envCache = {};
function envelope(W, Hc, tightenLosRate) {
    var key = W + 'x' + Hc + (tightenLosRate ? 'T' : '');
    if (_envCache[key]) return _envCache[key];
    var raw = fb.rawFeatureBounds(W, Hc);
    var sb = fb.standardizeBounds(net, raw);
    if (tightenLosRate != null && tightenLosRate !== false) {
        // tightenLosRate is a raw losRate*50 half-range cap (sound IF justified);
        // here used only as an ANALYSIS variant, NOT the sound gate. Replace the
        // standardized interval of feature 12 with one derived from a capped raw
        // range [-cap, cap].
        var cap = tightenLosRate;
        var lo12 = (-cap - net.fmu[12]) / net.fsd[12];
        var hi12 = (cap - net.fmu[12]) / net.fsd[12];
        sb.lo[12] = Math.min(lo12, hi12); sb.hi[12] = Math.max(lo12, hi12);
    }
    var e = ibp.ibpValue(net, sb.lo, sb.hi);
    _envCache[key] = e;
    return e;
}

// ---- certification core ----
// cands: [16][x,y]; scoreLo/scoreHi: per-candidate interval. Returns
// {certified:bool, coord:[x,y]|null}. Dedup by bitwise-equal coords.
function certifyArgmax(cands, scoreLo, scoreHi) {
    var K = cands.length;
    // group by bitwise-equal coords; canonical = lowest index
    var groups = [];           // {x,y,slo,shi,canon}
    var assigned = new Array(K).fill(-1);
    for (var i = 0; i < K; i++) {
        if (assigned[i] >= 0) continue;
        var gx = cands[i][0], gy = cands[i][1];
        var slo = scoreLo[i], shi = scoreHi[i];
        for (var j = i + 1; j < K; j++) {
            if (assigned[j] >= 0) continue;
            if (cands[j][0] === gx && cands[j][1] === gy) {   // bitwise (===) equal
                assigned[j] = groups.length;
                if (scoreLo[j] > slo) slo = scoreLo[j];
                if (scoreHi[j] > shi) shi = scoreHi[j];
            }
        }
        assigned[i] = groups.length;
        groups.push({ x: gx, y: gy, slo: slo, shi: shi, canon: i });
    }
    // winner = group with highest slo; certify if its slo > shi of all others
    var w = 0;
    for (var g = 1; g < groups.length; g++) if (groups[g].slo > groups[w].slo) w = g;
    var ok = true;
    for (g = 0; g < groups.length; g++) {
        if (g === w) continue;
        if (!(groups[w].slo > groups[g].shi)) { ok = false; break; }
    }
    if (!ok) return { certified: false, coord: null };
    return { certified: true, coord: [groups[w].x, groups[w].y] };
}

// Build per-candidate score intervals for a given mode.
// Non-rolled candidate -> exact point [vprior, vprior].
// Rolled candidate -> [catchLo+bootLo, catchHi+bootHi] per mode:
//   'sound'    : catch [0,Cmax],        boot [env.lo, env.hi]         (IBP)
//   'ceiling'  : catch [0,Cmax],        boot [BSTAR_LO, BSTAR_HI]     (global true-boot range; the achievable ceiling)
//   'trueBoot' : catch [0,Cmax],        boot point [tb,tb]            (boot perfect)
//   'trueCatch': catch point [tc,tc],   boot [env.lo, env.hi]         (catch perfect)
//   'perfect'  : score point [trueScore,trueScore]                    (soundness self-test)
function buildIntervals(d, mode, env, Cmax, trueCatch, trueBoot, BSTAR_LO, BSTAR_HI) {
    var K = d.cands.length;
    var slo = new Array(K), shi = new Array(K);
    var rolledIdx = {};
    for (var r = 0; r < d.rolled.length; r++) rolledIdx[d.rolled[r][0]] = d.rolled[r];
    for (var k = 0; k < K; k++) {
        if (rolledIdx[k] === undefined) { slo[k] = d.vprior[k]; shi[k] = d.vprior[k]; continue; }
        var cLo = 0, cHi = Cmax, bLo = env.lo, bHi = env.hi;
        if (mode === 'ceiling') { bLo = BSTAR_LO; bHi = BSTAR_HI; }
        else if (mode === 'trueBoot') { bLo = trueBoot[k]; bHi = trueBoot[k]; }
        else if (mode === 'trueCatch') { cLo = trueCatch[k]; cHi = trueCatch[k]; }
        else if (mode === 'perfect') { cLo = cHi = trueCatch[k]; bLo = bHi = trueBoot[k]; }
        slo[k] = cLo + bLo; shi[k] = cHi + bHi;
    }
    return { lo: slo, hi: shi };
}

async function run(globPaths, label, sampleEvery, BSTAR_LO, BSTAR_HI) {
    sampleEvery = sampleEvery || 1;
    if (BSTAR_LO == null) BSTAR_LO = 0.0;
    if (BSTAR_HI == null) BSTAR_HI = 4.92;   // data-wide true-boot ceiling (near-sound)
    var stats = {
        n: 0, sound: 0, soundT: 0, ceiling: 0, noRolledTop2: 0, perfect: 0,
        falseCert_sound: 0, falseCert_soundT: 0, falseCert_ceiling: 0, falseCert_perfect: 0,
        // tightness: among NON-sound-certified plans
        nFail: 0, failBoot: 0, failCatch: 0, failBoth: 0, failNeither: 0,
        certTrueBoot: 0, certTrueCatch: 0
    };
    for (var fp = 0; fp < globPaths.length; fp++) {
        var rl = readline.createInterface({ input: fs.createReadStream(globPaths[fp]).pipe(zlib.createGunzip()) });
        var lineNo = 0;
        for await (var line of rl) {
            if (!line) continue;
            lineNo++;
            if (sampleEvery > 1 && (lineNo % sampleEvery) !== 0) continue;
            var d = JSON.parse(line);
            stats.n++;
            var W = d.cfg.W, Hc = d.cfg.Hc;
            var env = envelope(W, Hc, false);
            var envT = envelope(W, Hc, 110.0);   // losRate*50 capped at +-110 (analysis variant)
            var Cmax = cbnd.catchBound(d.s, d.cfg);
            // true catch/boot per rolled index
            var trueCatch = {}, trueBoot = {};
            for (var r = 0; r < d.rolled.length; r++) { trueCatch[d.rolled[r][0]] = d.rolled[r][1]; trueBoot[d.rolled[r][0]] = d.rolled[r][2]; }
            var tx = hexToF64(d.lab.tx), ty = hexToF64(d.lab.ty);

            // ---- SOUND gate (full IBP boot bound) ----
            var iv = buildIntervals(d, 'sound', env, Cmax, trueCatch, trueBoot, BSTAR_LO, BSTAR_HI);
            var cert = certifyArgmax(d.cands, iv.lo, iv.hi);
            if (cert.certified) {
                stats.sound++;
                if (!(cert.coord[0] === tx && cert.coord[1] === ty)) stats.falseCert_sound++;
            }
            // ---- losRate-tightened variant (analysis only) ----
            var ivT = buildIntervals(d, 'sound', envT, Cmax, trueCatch, trueBoot, BSTAR_LO, BSTAR_HI);
            var certT = certifyArgmax(d.cands, ivT.lo, ivT.hi);
            if (certT.certified) {
                stats.soundT++;
                if (!(certT.coord[0] === tx && certT.coord[1] === ty)) stats.falseCert_soundT++;
            }
            // ---- ORACLE CEILING: rolled = [0, Cmax + Bstar] (true catch ceiling +
            //      data-wide boot ceiling). The fraction certifiable here is the
            //      ceiling the sound gate approaches as the boot bound tightens to
            //      the true reachable range. ----
            var ivO = buildIntervals(d, 'ceiling', env, Cmax, trueCatch, trueBoot, BSTAR_LO, BSTAR_HI);
            var certO = certifyArgmax(d.cands, ivO.lo, ivO.hi);
            if (certO.certified) {
                stats.ceiling++;
                if (!(certO.coord[0] === tx && certO.coord[1] === ty)) stats.falseCert_ceiling++;
            }
            // ---- PERFECT (true scores) soundness self-test: MUST be 100% / 0-false ----
            var ivP = buildIntervals(d, 'perfect', env, Cmax, trueCatch, trueBoot, BSTAR_LO, BSTAR_HI);
            var certP = certifyArgmax(d.cands, ivP.lo, ivP.hi);
            if (certP.certified) {
                stats.perfect++;
                if (!(certP.coord[0] === tx && certP.coord[1] === ty)) stats.falseCert_perfect++;
            }
            // ---- Deliverable 2: oracle ceiling, two precise quantities ----
            // (i) "no rolled in deduped top-2 (by TRUE score)": the decision is then
            //     a pure-vprior call -> the rolled scores never enter the top-2, so
            //     the committed target is decided entirely by the exact vprior.
            var grp = dedupTrue(d);
            var sortedG = grp.slice().sort(function (a, b) { return b.score - a.score; });
            var top2rolled = (sortedG.length >= 1 && sortedG[0].rolled) || (sortedG.length >= 2 && sortedG[1].rolled);
            if (!top2rolled) stats.noRolledTop2++;
            // (ii) "robust to ANY rolled perturbation": the committed target is
            //     unchanged for EVERY assignment of rolled scores in their FULL
            //     possible range. A rolled candidate's score can range over
            //     [0, Cmax + Bmax_oracle] where Bmax_oracle is the data-wide true
            //     boot ceiling (the largest boot any plan can produce). The decision
            //     is robust iff the deduped winner W still wins when every rolled
            //     group is pushed to its max-possible and W (if rolled) to its min.
            //     This is the SOUND gate with the TRUE Cmax and the TIGHTEST sound
            //     boot ceiling -> the ceiling the sound gate approaches.

            // ---- tightness breakdown (only for plans the SOUND gate FAILS) ----
            if (!cert.certified) {
                stats.nFail++;
                var ivTB = buildIntervals(d, 'trueBoot', env, Cmax, trueCatch, trueBoot, BSTAR_LO, BSTAR_HI);
                var cTB = certifyArgmax(d.cands, ivTB.lo, ivTB.hi).certified;   // boot perfect, catch bounded [0,Cmax]
                var ivTC = buildIntervals(d, 'trueCatch', env, Cmax, trueCatch, trueBoot, BSTAR_LO, BSTAR_HI);
                var cTC = certifyArgmax(d.cands, ivTC.lo, ivTC.hi).certified;   // catch perfect, boot bounded (IBP)
                if (cTB) stats.certTrueBoot++;   // would certify if boot were perfect
                if (cTC) stats.certTrueCatch++;  // would certify if catch were perfect
                if (cTB && !cTC) stats.failBoot++;       // boot is the sole blocker
                else if (cTC && !cTB) stats.failCatch++; // catch is the sole blocker
                else if (cTB && cTC) stats.failBoth++;   // either fix alone suffices
                else stats.failNeither++;                // needs BOTH bounds tightened
            }
        }
    }
    return stats;
}

// deduped TRUE-score groups with rolled flag
function dedupTrue(d) {
    var K = d.cands.length, rolledIdx = {};
    for (var r = 0; r < d.rolled.length; r++) rolledIdx[d.rolled[r][0]] = true;
    var groups = [], assigned = new Array(K).fill(-1);
    for (var i = 0; i < K; i++) {
        if (assigned[i] >= 0) continue;
        var gx = d.cands[i][0], gy = d.cands[i][1], sc = d.score[i], rolled = !!rolledIdx[i];
        for (var j = i + 1; j < K; j++) {
            if (assigned[j] >= 0) continue;
            if (d.cands[j][0] === gx && d.cands[j][1] === gy) {
                assigned[j] = groups.length;
                if (d.score[j] > sc) sc = d.score[j];
                if (rolledIdx[j]) rolled = true;
            }
        }
        assigned[i] = groups.length;
        groups.push({ score: sc, rolled: rolled, canon: i });
    }
    return groups;
}

module.exports = { run: run, certifyArgmax: certifyArgmax, envelope: envelope, dedupTrue: dedupTrue, hexToF64: hexToF64 };
