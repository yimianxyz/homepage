// studentScores.js — the DETERMINISTIC JS student scorer for L1h (#5 → side-b #6).
//
// Contract:  studentScores(snapshot, cands, cfg) -> number[16]
//   snapshot : { px, py, pvx, pvy, psize, bx[], by[], bvx[], bvy[], nAlive }
//              (prod's planCheap snapshot; lastFeed/nowMs are causally dead and
//               not required)
//   cands    : [{x,y} × 16]  the 16 candidate targets (prod's candidates())
//   cfg      : { W, Hc, PREDATOR_RANGE, NUM_BOIDS }  derived-config vector
//   returns  : number[16]  the student's final scores (argmax over the DEDUPED
//              groups = the committed target). L1h: trust this argmax when its
//              deduped top-2 margin ≥ τ, else fall back to prod's exact rollout.
//
// This is the L1r/l1rs student: it reuses prod's EXACT cp_features + cp_value
// (js/cheap_planner.js) and the EXACT ballistic pidx sort — 12 of 16 scores are
// prod's value-net prior, bitwise. It replaces ONLY the rollout: the 4 rolled
// candidates (pidx[0:4]) get the NN's predicted score = argmax(catches over
// 0..23 classes) + boot, from a deep-set forward in **float64** (the canonical
// deploy precision; the torch trunk was float32 — SPEC §4c calibrates the JS
// student, never the torch copy).
//
// All arithmetic JS float64 (V8/node). Deterministic; no wall-clock, no RNG.
'use strict';
const fs = require('fs');
const path = require('path');
// Locate prod js/ robustly: walk up from here (works in dev/exact_nn/student/
// AND when this file is copied into /workspace/.team/...). Override with
// EXACTNN_JS_DIR if prod lives elsewhere.
function findJsDir() {
    if (process.env.EXACTNN_JS_DIR) return process.env.EXACTNN_JS_DIR;
    let d = __dirname;
    for (let i = 0; i < 8; i++) {
        if (fs.existsSync(path.join(d, 'js', 'cheap_planner.js'))) return path.join(d, 'js');
        const up = path.dirname(d);
        if (up === d) break;
        d = up;
    }
    throw new Error('studentScores: cannot find js/cheap_planner.js — set EXACTNN_JS_DIR');
}
const JS_DIR = findJsDir();
const planner = require(path.join(JS_DIR, 'cheap_planner.js'));   // cp_features, cp_value
const NET = require(path.join(JS_DIR, 'value_net.json'));

const PMAX_S = 2.5, PMAX_F = 0.05;   // js/predator.js PREDATOR_MAX_SPEED/FORCE
const CP_PS = planner.CP.PS, CP_VS = planner.CP.VS;   // 200, 6

// ---- deep-set forward (matches train_shakedown/models.py DeepSet, head_out=25) ----
function gelu(x) { return 0.5 * x * (1 + erf(x / Math.SQRT2)); }
// erf == prod's cp_erf (Abramowitz-Stegun 7.1.26, bit-identical coefficients), so
// this GELU matches prod's cp_gelu exactly. NOTE: the torch trunk trained with
// torch.nn.GELU (EXACT erf), so the deploy JS student differs from the torch copy
// by the A-S erf error (~1e-7 in the activation → ~1e-6 in the output) — negligible
// for argmax(catches)+boot, and the JS student is the canonical calibrated artifact
// (SPEC §4c: τ is calibrated against the deployed JS student, never the torch copy).
function erf(x) {
    var t = 1 / (1 + 0.3275911 * Math.abs(x));
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return x >= 0 ? y : -y;
}
function lin(W, b, x) {                 // W (out×in), b (out), x (in) -> (out)
    const out = new Array(W.length);
    for (let o = 0; o < W.length; o++) { let a = b[o]; const wr = W[o]; for (let i = 0; i < x.length; i++) a += wr[i] * x[i]; out[o] = a; }
    return out;
}
function geluVec(v) { const o = new Array(v.length); for (let i = 0; i < v.length; i++) o[i] = gelu(v[i]); return o; }

function loadStudent(weightsPath) {
    const J = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));
    const W = J.weights;
    const TASK = (J.args && J.args.task) || 'l1rs';   // l1rs: argmax(catch)+boot; l1rs2: E[catch]+boot
    const CATCH_CLASSES = 24;

    // catch reduction → the deploy combined score per rolled candidate is
    // catchReduce(logits) + boot. l1rs uses the integer argmax; l1rs2 (side-a v2a)
    // uses the EXPECTED catch (Σ c·softmax_c, float64) — smooth, matches the
    // ranking-loss training target, and makes the score-margin a cleaner gate.
    function catchReduce(row) {
        if (TASK === 'l1rs2') {
            let mx = -Infinity;
            for (let c = 0; c < CATCH_CLASSES; c++) if (row[c] > mx) mx = row[c];
            let Z = 0; const e = new Array(CATCH_CLASSES);
            for (let c = 0; c < CATCH_CLASSES; c++) { e[c] = Math.exp(row[c] - mx); Z += e[c]; }
            let ec = 0; for (let c = 0; c < CATCH_CLASSES; c++) ec += c * e[c];
            return ec / Z;
        }
        let bc = 0, bb = -Infinity;
        for (let c = 0; c < CATCH_CLASSES; c++) if (row[c] > bb) { bb = row[c]; bc = c; }
        return bc;
    }

    function deepset(boidTok, glob, candTok) {
        // boidTok: [n][4]; glob: [11]; candTok: [16][25]
        const n = boidTok.length;
        // per-boid phi -> (n,64), then masked mean+max pool
        const mean = new Array(64).fill(0);
        const mx = new Array(64).fill(-Infinity);
        for (let i = 0; i < n; i++) {
            let h = geluVec(lin(W['enc.phi.0.weight'], W['enc.phi.0.bias'], boidTok[i]));
            h = geluVec(lin(W['enc.phi.2.weight'], W['enc.phi.2.bias'], h));
            for (let d = 0; d < 64; d++) { mean[d] += h[d]; if (h[d] > mx[d]) mx[d] = h[d]; }
        }
        const cnt = n > 0 ? n : 1;
        for (let d = 0; d < 64; d++) mean[d] /= cnt;
        if (n === 0) for (let d = 0; d < 64; d++) mx[d] = 0;   // amax of empty → matches masked_pool clamp
        // post: [mean(64), max(64), glob(11)] -> 64
        const pooled = mean.concat(mx, glob);                  // 139
        let g = geluVec(lin(W['enc.post.0.weight'], W['enc.post.0.bias'], pooled));
        g = lin(W['enc.post.2.weight'], W['enc.post.2.bias'], g);   // (64), no final act
        // per-candidate head -> (16,25)
        const out = new Array(candTok.length);
        for (let k = 0; k < candTok.length; k++) {
            const ci = candTok[k].concat(g);                   // 25 + 64 = 89
            let c = geluVec(lin(W['cand.0.weight'], W['cand.0.bias'], ci));
            c = geluVec(lin(W['cand.2.weight'], W['cand.2.bias'], c));
            c = lin(W['cand.4.weight'], W['cand.4.bias'], c);  // (64), no final act
            out[k] = lin(W['head.lin.weight'], W['head.lin.bias'], c);   // 25
        }
        return out;
    }

    // featurization — mirrors train_shakedown/ds.py build_inputs exactly.
    function featurize(s, cands, cfg, feat, vprior) {
        const n = s.nAlive;
        const boidTok = new Array(n);
        for (let i = 0; i < n; i++)
            boidTok[i] = [(s.bx[i] - s.px) / CP_PS, (s.by[i] - s.py) / CP_PS, s.bvx[i] / CP_VS, s.bvy[i] / CP_VS];
        const glob = [s.pvx / CP_VS, s.pvy / CP_VS, s.psize / 20.0,
                      n / 120.0, n / cfg.NUM_BOIDS,
                      cfg.W / 1680.0, cfg.Hc / 1680.0, cfg.PREDATOR_RANGE / 80.0, cfg.NUM_BOIDS / 120.0,
                      s.px / cfg.W, s.py / cfg.Hc];
        const candTok = new Array(cands.length);
        for (let k = 0; k < cands.length; k++) {
            const kind = k === 0 ? 0 : (k <= n ? 1 : 2);       // e3d / boid / E3D-pad (candidates())
            const oneh = [kind === 0 ? 1 : 0, kind === 1 ? 1 : 0, kind === 2 ? 1 : 0];
            candTok[k] = [(cands[k].x - s.px) / CP_PS, (cands[k].y - s.py) / CP_PS]
                .concat(feat[k], [vprior[k] / 3.0], oneh);
        }
        return { boidTok, glob, candTok };
    }

    // THE SCORER.
    function studentScores(snapshot, cands, cfg) {
        const st = { px: snapshot.px, py: snapshot.py, pvx: snapshot.pvx, pvy: snapshot.pvy,
                     psize: snapshot.psize, bx: snapshot.bx, by: snapshot.by, bvx: snapshot.bvx,
                     bvy: snapshot.bvy, nAlive: snapshot.nAlive };
        const fr = planner.cp_features(st, cands, PMAX_S, PMAX_F);     // EXACT prod features
        const vprior = planner.cp_value(NET, fr.feat, fr.ctx);         // EXACT prod value net
        // ballistic pidx: pscore = feat[18]-feat[16], desc, tie -> lowest index
        const ps = fr.feat.map(r => r[18] - r[16]);
        const pidx = fr.feat.map((_, k) => k).sort((a, b) => (ps[b] - ps[a]) || (a - b));
        const rolled = pidx.slice(0, 4);
        // NN forward
        const inp = featurize(st, cands, cfg, fr.feat, vprior);
        const out = deepset(inp.boidTok, inp.glob, inp.candTok);
        // final scores: vprior everywhere, NN-predicted on the 4 rolled
        const score = vprior.slice();
        for (const k of rolled) {
            score[k] = catchReduce(out[k]) + out[k][CATCH_CLASSES];    // catch + boot
        }
        return score;
    }
    studentScores.deepset = deepset;          // exposed for the parity check
    studentScores.featurize = featurize;
    return studentScores;
}

module.exports = { loadStudent, PMAX_S, PMAX_F };
