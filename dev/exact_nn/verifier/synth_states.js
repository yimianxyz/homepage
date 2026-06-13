// synth_states.js — SPEC §5 boundary/synthetic-state differential for L0.
//
// The 84-cell matrix verifies L0≡prod over 12.9M REACHABLE game frames. "All
// situation" is read broadly (§5): arbitrary LEGAL states, not just reachable
// ones, with the persistent policy state sampled JOINTLY with the boid config.
// This fuzzes exactly that and checks L0's force == prod's force BITWISE, with
// identical injected {target, frame, egBoid, configured, cfg} on both — which
// also exercises the state-injection convention (§2a) end-to-end (the path the
// student sealed-verdict will use).
//
// Both policies are loaded with the __cheapDebug transform (digest-inert, proven
// by diff_harness --selftest) so state can be injected identically. Prod = a
// fresh predator_cheap closure (loadPolicyAgain); L0 = the in-memory debug build
// of policy_unified.js (candidates/l0.js with the transform). Same context →
// same Vector / globals.
//
//   node synth_states.js --n 20000 --seed 1
'use strict';
const path = require('path');
const { createGame } = require('../stepper.js');

// the same anchored transform diff_harness uses (kept local to avoid import cycle)
const DEBUG_ANCHOR = '    window.__cheap = {';
function debugTransform(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(DEBUG_ANCHOR) < 0) throw new Error('anchor not found');
    return code.replace(DEBUG_ANCHOR,
        '    window.__cheapDebug = {\n'
        + '        get: function () { return { target: target, frame: frame, egBoid: egBoid, configured: configured, cfgW: cfg.W, cfgHc: cfg.Hc }; },\n'
        + '        set: function (s) { target = s.target; frame = s.frame; egBoid = s.egBoid; configured = s.configured; cfg.W = s.cfgW; cfg.Hc = s.cfgHc; }\n'
        + '    };\n' + DEBUG_ANCHOR);
}

// tiny deterministic RNG (mulberry32) — own, so synthetic sampling is replayable
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

const _f64 = new Float64Array(4), _u32 = new Uint32Array(_f64.buffer);
function bitsEq(ax, ay, bx, by) { _f64[0] = ax; _f64[1] = ay; _f64[2] = bx; _f64[3] = by; return _u32[0] === _u32[4] && _u32[1] === _u32[5] && _u32[2] === _u32[6] && _u32[3] === _u32[7]; }
function hex(x) { const d = new DataView(new ArrayBuffer(8)); d.setFloat64(0, x); return d.getBigUint64(0).toString(16).padStart(16, '0'); }

async function main() {
    const args = { n: 20000, seed: 1, W: 1512, H: 982, mismatchLimit: 20 };
    for (let i = 2; i < process.argv.length; i++) {
        const k = process.argv[i];
        if (k === '--n') args.n = +process.argv[++i];
        else if (k === '--seed') args.seed = +process.argv[++i];
        else if (k === '--W') args.W = +process.argv[++i];
        else if (k === '--H') args.H = +process.argv[++i];
    }
    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    // one game just to get a loaded context (+ a sim object for pred.simulation)
    const game = await createGame({ policyDir, W: args.W, H: args.H, seed: 1, fastRender: true, transform: debugTransform });
    const Vector = game.win.Vector || global.Vector;
    const prod = await game.loadPolicyAgain();                 // fresh prod closure + its __cheapDebug
    const prodForce = prod.cheap.force, prodDbg = prod.debug;
    const l0mod = require(path.join(__dirname, '..', 'candidates', 'l0.js'));
    const l0 = await l0mod.create(game, { policyDir, transform: debugTransform });
    const l0Force = l0.force, l0Dbg = l0._debug;
    if (!prodDbg || !l0Dbg) throw new Error('debug hooks missing (prod=' + !!prodDbg + ' l0=' + !!l0Dbg + ')');

    const rnd = mulberry32(args.seed >>> 0);
    const B = 20, Wd = args.W, Hd = args.H;
    const pos = () => ({ x: rnd() * (Wd + 4 * B) - 2 * B, y: rnd() * (Hd + 4 * B) - 2 * B });   // incl. out-of-domain band
    const vel = (max) => { const a = rnd() * 2 * Math.PI, m = rnd() * max; return { x: Math.cos(a) * m, y: Math.sin(a) * m }; };
    // N distribution: stress every regime incl. extremes
    const Ns = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 15, 16, 30, 60, 120, 130, 200];

    let checked = 0, mism = 0; const samples = [];
    const regimeCount = {};
    for (let it = 0; it < args.n; it++) {
        const N = Ns[(rnd() * Ns.length) | 0];
        regimeCount[N <= 5 ? 'N<=5' : 'N>5'] = (regimeCount[N <= 5 ? 'N<=5' : 'N>5'] || 0) + 1;
        const boids = new Array(N);
        for (let i = 0; i < N; i++) boids[i] = { position: new Vector(pos().x, pos().y), velocity: new Vector(vel(6).x, vel(6).y) };
        const p = pos(), pv = vel(2.5);
        const predStub = { position: new Vector(p.x, p.y), velocity: new Vector(pv.x, pv.y),
            currentSize: 12 + rnd() * (21.6 - 12), simulation: { canvasWidth: Wd, canvasHeight: Hd } };
        // adversarial persistent state, sampled JOINTLY (§5)
        const egIdx = N > 0 ? ((rnd() * (N + 1) | 0) - 1) : -1;   // -1..N-1 (-1 = null)
        const inj = {
            target: { x: rnd() * Wd, y: rnd() * Hd },
            frame: (rnd() * 17) | 0,                               // 0..16 (gate timing)
            egBoid: (egIdx >= 0 && egIdx < N) ? boids[egIdx] : null,
            configured: rnd() < 0.5,                               // also test pre-configure path
            cfgW: Wd + 2 * B, cfgHc: Hd + 2 * B,
        };
        // inject identical state into both (egBoid by index since the boid OBJECTS
        // differ per policy only by reference — both get the SAME boids array, so
        // the same object reference is valid for both)
        prodDbg.set(inj); l0Dbg.set(inj);
        const rf = prodForce(predStub, boids);
        const cf = l0Force(predStub, boids);
        checked++;
        if (!bitsEq(rf.x, rf.y, cf.x, cf.y)) {
            mism++;
            if (samples.length < args.mismatchLimit) samples.push({ it, N, frame: inj.frame, egIdx, configured: inj.configured,
                prod: { x: rf.x, y: rf.y, xb: hex(rf.x), yb: hex(rf.y) }, l0: { x: cf.x, y: cf.y, xb: hex(cf.x), yb: hex(cf.y) } });
        }
    }
    const out = { spec: 'SPEC §5 synthetic/adversarial-state L0≡prod differential (state-injected)',
        W: args.W, H: args.H, seed: args.seed, checked, mismatches: mism,
        Ns_sampled: Ns, regimeCount, samples };
    console.log(JSON.stringify(out, null, 1));
    process.exit(mism > 0 ? 2 : 0);
}

if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
