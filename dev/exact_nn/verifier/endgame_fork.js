// endgame_fork.js — FAST T*-search runner. The planner/endgame split gate only
// changes behaviour at N<=T (T<=12), so the entire natural game from N0 down to
// N=FORK_N(=13) is IDENTICAL across all T. We run that expensive high-N planner
// PREFIX once per (screen,seed), snapshot the full sim state, then replay the cheap
// low-N ENDGAME for each T from the snapshot. ~5-10x vs re-running full games per T.
//
// Correctness is proven empirically: validate_fork.js compares this runner's
// {frames,eaten,cleared,trajDigest} against the trusted diff_harness.runGame full
// game for the SAME (screen,seed,T) — must be bitwise-identical. Only then is it used.
//
// State captured at the fork: (a) deep clone of sim (boids/predator/boidsEaten —
// the whole object graph), (b) RNG state (mulberry32 call-count + seed → exact
// restore), (c) virtual clock _frame (feed-cooldown ms), (d) policy persistent
// closure {target,frame,egBoid,configured,cfg.W/Hc} via the __cheapDebug seam.
'use strict';
const path = require('path');
const { createGame } = require('../stepper.js');

const POLICY_DIR = path.join(__dirname, '..', '..', '..', 'js');
const K_MUL = 0x6D2B79F5;
const AREF = (1024 + 20) * (768 + 20);

// ---- transforms (applied to the pristine prod JS as it loads) ----------------
// 1) rng.js: expose exact RNG state (call-count + seed) and a _frame setter, so we
//    can restore the virtual clock + mulberry32 stream at the fork. Arithmetic is
//    unchanged — only getters/setters added (validated bitwise downstream).
function rngTransform(code) {
    code = code.replace('var _rng = null;', 'var _rng = null, _rngCalls = 0, _rngSeed = 0;');
    code = code.replace('_rng = mulberry32(seed >>> 0);\n        _frame = 0;',
                        '_rng = mulberry32(seed >>> 0);\n        _rngSeed = seed >>> 0; _rngCalls = 0;\n        _frame = 0;');
    code = code.replace('        return _rng();\n    }',
                        '        _rngCalls++;\n        return _rng();\n    }');
    code = code.replace('        mulberry32: mulberry32,\n    };',
        '        mulberry32: mulberry32,\n'
      + '        __getRngState: function () { return { calls: _rngCalls, seed: _rngSeed, frame: _frame, frameMs: _frameMs }; },\n'
      + '        __setRngState: function (s) { _rngSeed = s.seed >>> 0; _rngCalls = s.calls >>> 0; _frame = s.frame; _frameMs = s.frameMs;\n'
      + '            _rng = mulberry32(((s.seed >>> 0) + Math.imul(s.calls, ' + K_MUL + ')) >>> 0); },\n    };');
    code = code.replace('        g.mulberry32 = mulberry32;',
        '        g.mulberry32 = mulberry32;\n'
      + '        g.__getRngState = api.__getRngState;\n'
      + '        g.__setRngState = api.__setRngState;');
    return code;
}

// 2) predator_cheap.js: replace the fixed `boids.length <= 5` endgame gate with a
//    configurable split gate (reads mutable window.__split), define that gate + a
//    safe default at LOAD time (createGame ticks once during setup, before any
//    external config), and expose the persistent closure for snapshot/restore.
const GATE_LINE = '            if (boids.length <= 5) return intercept(pred, boids);   // ENDGAME: torus head-on intercept';
const DEBUG_ANCHOR = '    window.__cheap = {';
const GATE_DEF =
    '    if (!window.__split) window.__split = { rule: "count", T: 0, Tref: 5, P: 0.3, theta: 0.09, N0: 120, inEndgame: false };\n'
  + '    window.__splitGate = function (N, cfgW, cfgHc) {\n'
  + '        var s = window.__split, enter, exit, Td;\n'
  + '        if (s.rule === "count") { enter = N <= s.T; exit = N >= s.T + 2; }\n'
  + '        else if (s.rule === "density") { var A = (cfgW + 20) * (cfgHc + 20); Td = Math.max(1, Math.round(s.Tref * Math.pow(A / ' + AREF + ', s.P))); enter = N <= Td; exit = N >= Td + 2; }\n'
  + '        else if (s.rule === "n0") { Td = Math.max(1, Math.round(s.theta * s.N0)); enter = N <= Td; exit = N >= Td + 2; }\n'
  + '        else if (s.rule === "lindim") { Td = Math.max(1, Math.round(s.a + s.b * Math.sqrt((cfgW + 20) * (cfgHc + 20)))); enter = N <= Td; exit = N >= Td + 2; }\n'
  + '        else if (s.rule === "mindim") { Td = Math.max(1, Math.round(s.a + s.b * Math.min(cfgW, cfgHc))); enter = N <= Td; exit = N >= Td + 2; }\n'
  + '        else throw new Error("fork: unknown rule " + s.rule);\n'
  + '        if (!s.inEndgame && enter) s.inEndgame = true; else if (s.inEndgame && exit) s.inEndgame = false;\n'
  + '        return s.inEndgame;\n'
  + '    };\n';
function cheapTransform(code) {
    if (code.indexOf(GATE_LINE) < 0) throw new Error('fork: gate line not found');
    code = code.replace(GATE_LINE,
        '            if (window.__splitGate(boids.length, cfg.W, cfg.Hc)) return intercept(pred, boids);   // CONFIGURABLE SPLIT');
    code = code.replace(DEBUG_ANCHOR,
        '    window.__cheapDebug = {\n'
      + '        get: function () { return { target: target, frame: frame, egBoid: egBoid, configured: configured, cfgW: cfg.W, cfgHc: cfg.Hc }; },\n'
      + '        set: function (s) { target = s.target; frame = s.frame; egBoid = s.egBoid; configured = s.configured; cfg.W = s.cfgW; cfg.Hc = s.cfgHc; }\n'
      + '    };\n' + GATE_DEF + DEBUG_ANCHOR);
    return code;
}

function transform(file, code) {
    if (file === 'rng.js') return rngTransform(code);
    if (file === 'predator_cheap.js') return cheapTransform(code);
    return null;
}

// ---- deep clone of the sim object graph (Vectors + back-refs) -----------------
function cloneEntity(proto, src, Vector, clonedSim) {
    const e = Object.create(proto);
    for (const k in src) if (Object.prototype.hasOwnProperty.call(src, k)) {
        const val = src[k];
        e[k] = (val && val instanceof Vector) ? new Vector(val.x, val.y) : val;
    }
    e.simulation = clonedSim;
    return e;
}
function cloneSimParts(sim, Vector, targetSim) {
    targetSim.boidsEaten = sim.boidsEaten;
    targetSim.boids = sim.boids.map(b => cloneEntity(Object.getPrototypeOf(b), b, Vector, targetSim));
    targetSim.predator = cloneEntity(Object.getPrototypeOf(sim.predator), sim.predator, Vector, targetSim);
}
function snapshotSim(sim, Vector) {
    const cs = Object.create(Object.getPrototypeOf(sim));
    for (const k in sim) if (Object.prototype.hasOwnProperty.call(sim, k)) {
        if (k === 'boids' || k === 'predator' || k === 'obstacles') continue;
        cs[k] = sim[k];
    }
    cs.obstacles = [];
    cloneSimParts(sim, Vector, cs);
    return cs;
}
// restore the LIVE sim (the one stepFrame closes over) from the pristine snapshot
function restoreSim(liveSim, snapSim, Vector) {
    cloneSimParts(snapSim, Vector, liveSim);   // fresh clones each call → snapshot stays pristine
    liveSim.predator.simulation = liveSim;
    for (const b of liveSim.boids) b.simulation = liveSim;
}

// FNV-1a 64-bit (BigInt) over 32-bit words — IDENTICAL to diff_harness.js so the
// forked full-game trajDigest (prefix ++ continuation) is bitwise-comparable.
const FNV_OFF = 0xcbf29ce484222325n, FNV_PRIME = 0x100000001b3n, MASK64 = (1n << 64n) - 1n;
function fnvInit() { return FNV_OFF; }
function fnvWord(h, w) { return ((h ^ BigInt(w >>> 0)) * FNV_PRIME) & MASK64; }
const _ab = new ArrayBuffer(32), _f64 = new Float64Array(_ab), _u32 = new Uint32Array(_ab);  // 4 doubles / 8 words

// ---- the fork runner ---------------------------------------------------------
// Returns per-T {T, frames, eaten, cleared, thru, trajDigest} for one (screen,seed).
async function forkRun(opt) {
    const { W, H, uaMobile, N0, seed, Ts, rule, ruleParams, forkN, maxFrames, digest } = opt;
    const startBoids = N0 || 0;   // force NUM_BOIDS = N0 (60/120); 0 => natural (uaMobile decides)
    const game = await createGame({ policyDir: POLICY_DIR, W, H, seed,
        startBoids, scatter: false, uaMobile, fastRender: true, transform });
    const win = game.win;
    const Vector = global.Vector;
    const dbg = win.__cheapDebug;
    win.__split.N0 = N0 || (uaMobile ? 60 : 120);

    // ---- PREFIX: pure planner (T=0 → gate never enters) until N <= forkN ----
    win.__split.rule = 'count'; win.__split.T = 0; win.__split.inEndgame = false;
    let prefixCapped = false, prefixDigest = fnvInit();
    {   // digest the prefix identically to diff_harness so a forked full-game digest
        // (prefix ++ continuation) is bitwise-comparable to runGame's trajDigest.
        const p = game.sim.predator;
        while (game.boidCount() > forkN) {
            if (game.frame() >= maxFrames) { prefixCapped = true; break; }
            game.stepFrame();
            if (digest) {
                _f64[0] = p.position.x; _f64[1] = p.position.y; _f64[2] = p.velocity.x; _f64[3] = p.velocity.y;
                for (let w = 0; w < 8; w++) prefixDigest = fnvWord(prefixDigest, _u32[w]);
                prefixDigest = fnvWord(prefixDigest, game.boidCount());
            }
        }
    }
    const snapSim = snapshotSim(game.sim, Vector);
    const snapRng = global.__getRngState();
    const snapPolicy = dbg.get();
    const prefixFrames = game.frame();
    const prefixEaten = game.eaten();
    const forkNcount = game.boidCount();
    const liveSim = game.sim;

    const out = [];
    for (const T of Ts) {
        restoreSim(liveSim, snapSim, Vector);
        global.__setRngState(snapRng);
        dbg.set({ target: snapPolicy.target, frame: snapPolicy.frame, egBoid: null,
                  configured: snapPolicy.configured, cfgW: snapPolicy.cfgW, cfgHc: snapPolicy.cfgHc });
        win.__split.rule = rule; win.__split.T = T; win.__split.inEndgame = false;
        if (ruleParams) Object.assign(win.__split, ruleParams);
        let f = prefixFrames, cleared = false, tDigest = prefixDigest;
        const p = liveSim.predator;
        while (f < maxFrames) {
            if (liveSim.boids.length === 0) { cleared = true; break; }
            game.stepFrame(); f++;
            if (digest) {
                _f64[0] = p.position.x; _f64[1] = p.position.y; _f64[2] = p.velocity.x; _f64[3] = p.velocity.y;
                for (let w = 0; w < 8; w++) tDigest = fnvWord(tDigest, _u32[w]);
                tDigest = fnvWord(tDigest, liveSim.boids.length);
            }
        }
        out.push({ T, frames: f, eaten: game.eaten(), cleared, thru: game.eaten() / f,
                   trajDigest: digest ? tDigest.toString(16) : null });
    }
    return { W, H, uaMobile: !!uaMobile, N0: win.__split.N0, seed, rule, forkN, forkNcount,
             prefixFrames, prefixEaten, prefixCapped, results: out };
}

module.exports = { forkRun, transform };
