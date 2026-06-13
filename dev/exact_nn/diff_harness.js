// diff_harness.js — lockstep bitwise differential harness (EXACT-NN north star).
//
// Steps ONE faithful simulation (stepper.js); every frame evaluates BOTH the
// frozen reference policy (prod js/predator_cheap.js closure) and a candidate
// policy on the same live state; compares the returned force (fx,fy) BITWISE
// (Float64 bit patterns via a shared Float64Array/Uint32Array view — catches
// -0 vs +0 and NaN-payload cases that === would hide); applies the REFERENCE
// force (lockstep) and logs every mismatch with the full force-input state.
//
// Modes:
//   lockstep (default) — apply reference force; candidate runs open-loop on
//     the reference trajectory. Mismatch count is the metric.
//   fork — apply the CANDIDATE force; report the first-divergence frame (the
//     first bitwise force mismatch — after it the trajectories part).
//
// Candidate interface (a node module):
//   module.exports.create = async function (game, helpers) {
//       return { name, configure(sim), force(pred, boids) -> {x,y}, reset() };
//   };
//   - force() must NOT mutate pred/boids/sim — the harness self-test proves the
//     trajectory is undisturbed vs an unhooked run, every time.
//   - Persistent state across frames (target / frame counter / egBoid
//     equivalents) is the candidate's own; reset() must clear it. The harness
//     builds a fresh engine+candidate per game anyway.
//   Builtins: 'identity' (a second independent prod closure — must produce 0
//   mismatches) and 'broken1ulp' (identity with fx XOR 1 ulp every --ulpEvery
//   calls — every perturbation must be caught at exactly that frame).
//
// Usage:
//   node dev/exact_nn/diff_harness.js --candidate identity --W 1512 --H 982 \
//        --seedStart 270000 --seeds 8 --maxFrames 20000 --json
//   node dev/exact_nn/diff_harness.js --selftest          # full proof battery
//   Exit codes: 0 = ran, zero mismatches; 2 = mismatches found; 1 = error/selftest fail.
//
// Throughput: printed per run (framesPerMinPerCore). Shard across cores/VMs by
// splitting --seedStart/--seeds (see shard_runner.js).
'use strict';
const fs = require('fs');
const path = require('path');
const { createGame } = require('./stepper.js');

// ---- the __cheapDebug transform -------------------------------------------
// Anchored in-memory insert into predator_cheap.js (and into the same bytes
// embedded in an in-memory L0 build): exposes get/set over the closure's
// persistent state {target, frame, egBoid, configured, cfg.W/Hc}. This powers
// the DECISION-LEVEL metric (primary per the #6 spec revision), the resync
// convention, and the state-injection convention. The insert adds no
// arithmetic and runs nothing per-frame by itself; the selftest proves a
// transformed game is digest-identical to a pristine one. Committed files are
// never touched.
const DEBUG_ANCHOR = '    window.__cheap = {';
function debugTransform(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(DEBUG_ANCHOR) < 0) throw new Error('debugTransform: anchor not found');
    return code.replace(DEBUG_ANCHOR,
        '    window.__cheapDebug = {\n' +
        '        get: function () { return { target: target, frame: frame, egBoid: egBoid, configured: configured, cfgW: cfg.W, cfgHc: cfg.Hc }; },\n' +
        '        set: function (s) { target = s.target; frame = s.frame; egBoid = s.egBoid; configured = s.configured; cfg.W = s.cfgW; cfg.Hc = s.cfgHc; }\n' +
        '    };\n' +
        DEBUG_ANCHOR);
}

function parseArgs(argv) {
    const a = { policyDir: path.join(__dirname, '..', '..', 'js'),
        candidate: 'identity', mode: 'lockstep',
        W: 390, H: 844, seedStart: 270000, seeds: 4,
        startBoids: 0, scatter: false, uaMobile: false, maxFrames: 20000,
        spawnScript: null, ulpEvery: 997, mismatchLimit: 50,
        postExtinct: 60, decisions: true, resync: true,
        out: null, json: false, perseed: false, selftest: false,
        fastRender: true };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--policyDir') a.policyDir = argv[++i];
        else if (k === '--candidate') a.candidate = argv[++i];
        else if (k === '--mode') a.mode = argv[++i];
        else if (k === '--W') a.W = +argv[++i];
        else if (k === '--H') a.H = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--startBoids') a.startBoids = +argv[++i];
        else if (k === '--scatter') a.scatter = true;
        else if (k === '--uaMobile') a.uaMobile = true;
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--spawnScript') a.spawnScript = JSON.parse(argv[++i]);
        else if (k === '--ulpEvery') a.ulpEvery = +argv[++i];
        else if (k === '--mismatchLimit') a.mismatchLimit = +argv[++i];
        else if (k === '--postExtinct') a.postExtinct = +argv[++i];
        else if (k === '--noDecisions') a.decisions = false;
        else if (k === '--noResync') a.resync = false;
        else if (k === '--out') a.out = argv[++i];
        else if (k === '--json') a.json = true;
        else if (k === '--perseed') a.perseed = true;
        else if (k === '--selftest') a.selftest = true;
        else if (k === '--noFastRender') a.fastRender = false;
        else throw new Error('unknown arg: ' + k);
    }
    return a;
}

// ---- float64 bit helpers ----
const _f64 = new Float64Array(4);
const _u32 = new Uint32Array(_f64.buffer);
function bitsHex(x) {
    const dv = new DataView(new ArrayBuffer(8));
    dv.setFloat64(0, x);
    return dv.getBigUint64(0).toString(16).padStart(16, '0');
}
function xorUlp(x) {   // flip the low mantissa bit: a guaranteed 1-ulp change
    const dv = new DataView(new ArrayBuffer(8));
    dv.setFloat64(0, x);
    dv.setBigUint64(0, dv.getBigUint64(0) ^ 1n);
    return dv.getFloat64(0);
}

// FNV-1a 64-bit over 32-bit words (BigInt; called once per frame — cheap enough)
const FNV_OFF = 0xcbf29ce484222325n, FNV_PRIME = 0x100000001b3n, MASK64 = (1n << 64n) - 1n;
function fnvInit() { return FNV_OFF; }
function fnvWord(h, w) { return ((h ^ BigInt(w >>> 0)) * FNV_PRIME) & MASK64; }

// ---- candidates ----
async function makeCandidate(spec, game, opt) {
    if (spec === 'identity') {
        const h = await game.loadPolicyAgain();
        return { name: 'identity-prod', configure() {}, force: h.cheap.force, reset() {}, _debug: h.debug };
    }
    if (spec === 'broken1ulp') {
        const h = await game.loadPolicyAgain();
        let calls = 0;
        const perturbed = [];
        return {
            name: 'broken-1ulp(every ' + opt.ulpEvery + ')',
            perturbedCalls: perturbed,
            configure() {}, reset() { calls = 0; perturbed.length = 0; },
            force(pred, boids) {
                const f = h.cheap.force(pred, boids);
                calls++;
                if (calls % opt.ulpEvery === 0) {
                    perturbed.push(calls - 1);          // 0-based call (== frame) index
                    return { x: xorUlp(f.x), y: f.y };
                }
                return f;
            },
        };
    }
    if (spec === 'brokencoarse') {   // selftest-only: a fault big enough to move the trajectory
        const h = await game.loadPolicyAgain();
        let calls = 0;
        const perturbed = [];
        return {
            name: 'broken-coarse(+1e-2 every ' + opt.ulpEvery + ')',
            perturbedCalls: perturbed,
            configure() {}, reset() { calls = 0; perturbed.length = 0; },
            force(pred, boids) {
                const f = h.cheap.force(pred, boids);
                calls++;
                if (calls % opt.ulpEvery === 0) { perturbed.push(calls - 1); return { x: f.x + 1e-2, y: f.y }; }
                return f;
            },
        };
    }
    const mod = require(path.resolve(spec));
    if (typeof mod.create !== 'function') throw new Error('candidate must export create()');
    return await mod.create(game, { policyDir: opt.policyDir,
        transform: opt.decisions ? debugTransform : null });
}

// ---- single game under the differential hook ----
// Returns { frames, cleared, clearedAt, mismatches:[...], counts, forceDigest,
//           trajDigest, framesByRegime }
async function runGame(opt, seed, candidateSpec) {
    const wantDbg = opt.decisions !== false;
    const game = await createGame({
        policyDir: opt.policyDir, W: opt.W, H: opt.H, seed,
        startBoids: opt.startBoids, scatter: opt.scatter, uaMobile: opt.uaMobile,
        fastRender: opt.fastRender, spawnScript: opt.spawnScript,
        transform: wantDbg ? debugTransform : null,
    });
    // Capture the reference closure BEFORE any candidate loads or hooks land.
    const refForce = game.win.__cheap.force;
    const refDbg = wantDbg ? game.win.__cheapDebug : null;
    const cand = candidateSpec ? await makeCandidate(candidateSpec, game, opt) : null;
    if (cand) cand.configure(game.sim);
    const candDbg = cand ? (cand._debug || null) : null;
    const fork = opt.mode === 'fork';
    const decisions = !!(refDbg && candDbg);

    let mmCount = 0, firstMM = -1, resyncs = 0;
    const mismatches = [];
    let fDigest = fnvInit();
    const regime = { planner: 0, intercept: 0, zero: 0 };
    // decision-level (PRIMARY metric per the #6 spec revision)
    const dec = { plans: 0, planDisagree: 0, egCommits: 0, egDisagree: 0 };
    let prevRefTarget, prevCandTarget, prevRefEg, prevCandEg;
    if (decisions) {
        const rs0 = refDbg.get(), cs0 = candDbg.get();
        prevRefTarget = rs0.target; prevCandTarget = cs0.target;
        prevRefEg = rs0.egBoid; prevCandEg = cs0.egBoid;
    }

    game.setForce(function (pred, boids) {
        const rf = refForce(pred, boids);
        let eq = true, cf = null;
        if (cand) {
            cf = cand.force(pred, boids);
            _f64[0] = rf.x; _f64[1] = rf.y; _f64[2] = cf.x; _f64[3] = cf.y;
            eq = _u32[0] === _u32[4] && _u32[1] === _u32[5] &&
                 _u32[2] === _u32[6] && _u32[3] === _u32[7];
        } else {
            _f64[0] = rf.x; _f64[1] = rf.y;
        }
        fDigest = fnvWord(fnvWord(fDigest, _u32[0]), _u32[1]);
        fDigest = fnvWord(fnvWord(fDigest, _u32[2]), _u32[3]);
        if (boids.length === 0) regime.zero++;
        else if (boids.length <= 5) regime.intercept++; else regime.planner++;

        let decBad = false;
        if (decisions) {
            const rs = refDbg.get(), cs = candDbg.get();
            // plan commit = planCheap produced a fresh target object this frame
            if (rs.target !== prevRefTarget) {
                dec.plans++;
                // dedup-by-coords: agreement iff the COMMITTED COORDINATES are
                // bitwise equal (duplicate candidates with identical coords
                // count as the same decision — indices are not the metric)
                const candPlanned = cs.target !== prevCandTarget;
                let agree = candPlanned && !!cs.target;
                if (agree) {
                    _f64[0] = rs.target.x; _f64[1] = rs.target.y;
                    _f64[2] = cs.target.x; _f64[3] = cs.target.y;
                    agree = _u32[0] === _u32[4] && _u32[1] === _u32[5] &&
                            _u32[2] === _u32[6] && _u32[3] === _u32[7];
                }
                if (!agree) { dec.planDisagree++; decBad = true; }
            }
            // endgame commit = egBoid changed (incl. null -> boid re-commit)
            if (rs.egBoid !== prevRefEg) {
                dec.egCommits++;
                // identity agreement: the SAME live boid object (both policies
                // receive the same array, so identity is directly comparable)
                if (cs.egBoid !== rs.egBoid) { dec.egDisagree++; decBad = true; }
            }
            prevRefTarget = rs.target; prevCandTarget = cs.target;
            prevRefEg = rs.egBoid; prevCandEg = cs.egBoid;
        }

        if (!eq || decBad) {
            if (!eq) {
                mmCount++;
                if (firstMM < 0) firstMM = game.frame();
            }
            if (mismatches.length < opt.mismatchLimit) {
                mismatches.push({
                    seed, W: opt.W, H: opt.H, frame: game.frame(),
                    n: boids.length, kind: !eq ? (decBad ? 'force+decision' : 'force') : 'decision',
                    ref:  { x: rf.x, y: rf.y, xb: bitsHex(rf.x), yb: bitsHex(rf.y) },
                    cand: cf ? { x: cf.x, y: cf.y, xb: bitsHex(cf.x), yb: bitsHex(cf.y) } : null,
                    state: game.snapshotState(),   // exact force-input state
                });
            }
            // resync convention: copy prod's persistent state into the candidate
            // so disagreement counts are per-decision, not cascade-inflated.
            // (target/egBoid are shared-safe: planCheap always allocates a fresh
            // target object and never mutates in place; egBoid is a live boid.)
            if (opt.resync !== false && decisions) { candDbg.set(refDbg.get()); resyncs++; }
        }
        return (fork && cf) ? cf : rf;
    });

    let tDigest = fnvInit();
    const p = game.sim.predator;
    let clearedAt = -1, post = 0;
    // Step to extinction (or maxFrames), then keep stepping opt.postExtinct
    // frames so the N==0 path (force(pred, []) -> Vector(0,0)) is exercised
    // differentially every game — the live page keeps ticking after the last
    // catch, so the corpus must too.
    while (game.frame() < opt.maxFrames) {
        if (game.boidCount() === 0) {
            if (clearedAt < 0) clearedAt = game.frame();
            if (post >= (opt.postExtinct || 0)) break;
            post++;
        }
        game.stepFrame();
        _f64[0] = p.position.x; _f64[1] = p.position.y;
        _f64[2] = p.velocity.x; _f64[3] = p.velocity.y;
        for (let w = 0; w < 8; w++) tDigest = fnvWord(tDigest, _u32[w]);
        tDigest = fnvWord(tDigest, game.boidCount());
    }
    const frames = game.frame();
    return {
        seed, frames, cleared: clearedAt >= 0,
        clearedAt: clearedAt >= 0 ? clearedAt : null,
        mismatchCount: mmCount, firstMismatchFrame: firstMM,
        decisions: decisions ? dec : null, resyncs,
        mismatches, framesByRegime: regime,
        forceDigest: fDigest.toString(16), trajDigest: tDigest.toString(16),
        candName: cand ? cand.name : null,
        perturbedCalls: cand && cand.perturbedCalls ? cand.perturbedCalls.slice() : undefined,
    };
}

// ---- self-test battery: proves the harness itself ----
async function selftest(opt) {
    const cfgs = [
        // ulpEvery is sized to each game's expected length so every config
        // exercises >=1 perturbation (short endgame games are ~10^2 frames).
        { tag: 'fullgame', W: 1024, H: 768, seed: 271001, startBoids: 0, scatter: false, maxFrames: 9000, ulpEvery: 997 },
        { tag: 'endgame',  W: 390,  H: 844, seed: 271002, startBoids: 4, scatter: true,  maxFrames: 4000, ulpEvery: 97 },
        { tag: 'gate',     W: 820,  H: 1180, seed: 271003, startBoids: 8, scatter: true, maxFrames: 8000, ulpEvery: 211 },
        // spawn-schedule game (#6 revision): endgame commit, then taps push
        // N 5->7 (incl. two SAME-COORDINATE spawns -> duplicate boids ->
        // duplicate candidates), then back down through the gate with the
        // committed egBoid possibly still alive.
        { tag: 'spawn', W: 1024, H: 768, seed: 271004, startBoids: 5, scatter: true, maxFrames: 8000, ulpEvery: 211,
          spawnScript: [{ frame: 120, x: 512, y: 384 }, { frame: 120, x: 512, y: 384 }, { frame: 300, x: 100, y: 100 }] },
    ];
    let fail = 0;
    const check = (name, ok, extra) => {
        console.log((ok ? 'PASS' : 'FAIL') + '  ' + name + (extra ? '  ' + extra : ''));
        if (!ok) fail++;
    };
    for (const c of cfgs) {
        const base = Object.assign({}, opt, c, { mode: 'lockstep' });
        const A  = await runGame(base, c.seed, null);                    // hooked, no candidate
        const A2 = await runGame(base, c.seed, null);                    // determinism
        const P  = await runGame(Object.assign({}, base, { decisions: false }), c.seed, null);  // PRISTINE load (no transform)
        const NR = await runGame(Object.assign({}, base, { fastRender: false }), c.seed, null);
        const I  = await runGame(base, c.seed, 'identity');              // identity candidate
        const B  = await runGame(base, c.seed, 'broken1ulp');            // lockstep, perturbed
        const F  = await runGame(Object.assign({}, base, { mode: 'fork' }), c.seed, 'broken1ulp');

        check(c.tag + ': determinism (traj+force digests equal)',
            A.trajDigest === A2.trajDigest && A.forceDigest === A2.forceDigest);
        check(c.tag + ': __cheapDebug transform is inert (pristine == transformed)',
            P.trajDigest === A.trajDigest && P.forceDigest === A.forceDigest);
        check(c.tag + ': fastRender bitwise-identical', NR.trajDigest === A.trajDigest);
        check(c.tag + ': candidate eval does not disturb sim', I.trajDigest === A.trajDigest);
        check(c.tag + ': identity candidate -> 0 mismatches',
            I.mismatchCount === 0, '(' + I.frames + ' frames)');
        check(c.tag + ': N==0 post-extinction frames exercised',
            !I.cleared || I.framesByRegime.zero > 0, '(' + I.framesByRegime.zero + ' zero-N frames)');
        check(c.tag + ': decision metric live, identity agrees on all commits',
            !!I.decisions && (I.decisions.plans + I.decisions.egCommits) > 0 &&
            I.decisions.planDisagree === 0 && I.decisions.egDisagree === 0,
            I.decisions ? '(' + I.decisions.plans + ' plans, ' + I.decisions.egCommits + ' egCommits)' : '(no decisions!)');
        const expected = Math.floor(B.frames / base.ulpEvery);
        check(c.tag + ': broken1ulp caught every perturbation',
            B.mismatchCount === expected && expected > 0,
            '(' + B.mismatchCount + '/' + expected + ')');
        check(c.tag + ': broken1ulp mismatch at exact perturbed frame',
            B.mismatches.length > 0 && B.perturbedCalls[0] === B.mismatches[0].frame);
        check(c.tag + ': lockstep contains the fault (trajectory unchanged)',
            B.trajDigest === A.trajDigest);
        // NOTE: a 1-ulp force fault need NOT move the trajectory — adding a
        // ~1e-17 force delta into a ~2.5-magnitude velocity rounds away below
        // velocity ulp (~4.4e-16). That asymmetry is WHY lockstep bitwise
        // force-compare is the north star: trajectory-level checks are
        // structurally blind to sub-velocity-ulp faults. Fork mode must still
        // DETECT the fault bitwise at the exact frame; trajectory divergence
        // is asserted with a coarse (1e-2) fault instead.
        check(c.tag + ': fork mode detects 1-ulp fault at exact frame',
            F.firstMismatchFrame === F.perturbedCalls[0],
            '(first @' + F.firstMismatchFrame + ')');
        const FC = await runGame(Object.assign({}, base, { mode: 'fork' }), c.seed, 'brokencoarse');
        check(c.tag + ': fork mode coarse fault diverges trajectory',
            FC.firstMismatchFrame === FC.perturbedCalls[0] && FC.trajDigest !== A.trajDigest,
            '(first @' + FC.firstMismatchFrame + ')');
        if (c.tag === 'spawn') {
            check('spawn: gate re-crossed upward (planner frames present)',
                I.framesByRegime.planner > 0 && I.framesByRegime.intercept > 0,
                '(' + I.framesByRegime.planner + ' planner / ' + I.framesByRegime.intercept + ' intercept frames)');
        }
    }
    console.log(fail === 0 ? 'SELFTEST: ALL PASS' : 'SELFTEST: ' + fail + ' FAILURES');
    return fail === 0 ? 0 : 1;
}

async function main() {
    const opt = parseArgs(process.argv);
    if (opt.selftest) process.exit(await selftest(opt));

    const t0 = process.hrtime.bigint();
    let totalFrames = 0, totalMM = 0, games = 0, cleared = 0, resyncs = 0;
    const regime = { planner: 0, intercept: 0, zero: 0 };
    const dec = { plans: 0, planDisagree: 0, egCommits: 0, egDisagree: 0 };
    let decLive = false;
    const perseed = [];
    const outStream = opt.out ? fs.createWriteStream(opt.out, { flags: 'a' }) : null;
    let firstMMs = [];

    for (let i = 0; i < opt.seeds; i++) {
        const seed = opt.seedStart + i;
        const r = await runGame(opt, seed, opt.candidate);
        games++; totalFrames += r.frames; totalMM += r.mismatchCount;
        if (r.cleared) cleared++;
        resyncs += r.resyncs;
        regime.planner += r.framesByRegime.planner;
        regime.intercept += r.framesByRegime.intercept;
        regime.zero += r.framesByRegime.zero;
        if (r.decisions) {
            decLive = true;
            dec.plans += r.decisions.plans; dec.planDisagree += r.decisions.planDisagree;
            dec.egCommits += r.decisions.egCommits; dec.egDisagree += r.decisions.egDisagree;
        }
        if (r.firstMismatchFrame >= 0) firstMMs.push({ seed, frame: r.firstMismatchFrame });
        if (outStream) for (const m of r.mismatches) outStream.write(JSON.stringify(m) + '\n');
        if (opt.perseed) perseed.push({ seed, frames: r.frames, cleared: r.cleared,
            mismatches: r.mismatchCount, firstMismatchFrame: r.firstMismatchFrame });
    }
    if (outStream) outStream.end();

    const elapsedMin = Number(process.hrtime.bigint() - t0) / 1e9 / 60;
    const out = {
        candidate: opt.candidate, mode: opt.mode,
        W: opt.W, H: opt.H, startBoids: opt.startBoids, scatter: opt.scatter,
        uaMobile: opt.uaMobile || undefined,
        spawnScript: opt.spawnScript ? opt.spawnScript.length + ' spawns' : null,
        seedStart: opt.seedStart, seeds: opt.seeds, maxFrames: opt.maxFrames,
        games, cleared, frames: totalFrames,
        framesByRegime: regime,
        mismatches: totalMM,
        decisions: decLive ? dec : null,
        resyncs,
        firstMismatches: firstMMs.slice(0, 10),
        framesPerMinPerCore: Math.round(totalFrames / elapsedMin),
        elapsedSec: +(elapsedMin * 60).toFixed(1),
    };
    if (opt.perseed) out.perseed = perseed;
    console.log(JSON.stringify(out));
    const decBad = decLive && (dec.planDisagree > 0 || dec.egDisagree > 0);
    process.exit((totalMM > 0 || decBad) ? 2 : 0);
}

if (require.main === module) {
    main().catch(e => { console.error(e); process.exit(1); });
}
module.exports = { runGame, selftest, parseArgs, xorUlp, bitsHex };
