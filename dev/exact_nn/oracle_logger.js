// exact_nn oracle decision-logger (#5) — records every plan decision of the
// FROZEN prod policy (workspace js/ ≡ main@6dce76f) for offline student training.
//
// NO edits to js/: the prod sources are hook-injected IN MEMORY via the shared
// stepper's transform seam. Three one-line hooks at exact-string anchors inside
// predator_cheap.js's planCheap (load hard-fails if an anchor doesn't match
// exactly once), plus a pure-JS wrapper around window.__cheap.force for
// per-frame mode + force bits. Hooks only READ live values — no arithmetic in
// the policy path changes, proven by --selftest's pristine-equivalence check
// (instrumented vs untouched run → bit-identical predator trajectory).
//
// Per plan decision (one JSONL line): full input state snapshot (px,py,pvx,
// pvy,psize,lastFeed,nowMs + all boid pos/vel), the 16 candidates, cp_features
// matrix + ctx, vprior(16), pidx (roll order; top-K_roll=4 get rolled),
// per-roll (ci, rollout catches, terminal bootstrap), final score(16), argmax
// bi, runner-up margin. Per frame (one columnar JSONL line per game): mode
// (0=zero 1=intercept 2=plan 3=steer), N, force fx/fy as exact float64 bit
// patterns (hex u64, DataView — catches -0/NaN where JSON numbers can't).
//
// All other floats are plain JSON numbers: ECMAScript number→string is
// shortest-round-trip, so parsing returns the identical float64 — except the
// sign of zero (JSON "0" parses to +0). Negative zeros DO occur in prod
// features (e.g. first-plan geometry); they are counted per shard (meta
// nNegZero) and provably causally dead: every downstream consumer (standardize,
// sum, product, the intercept min-image round) yields bit-identical results
// for ±0, and the policy path has no 1/x, atan2, or Object.is. NaN/±Inf in
// any logged value is a hard error (would be real corruption).
//
// Wall-clock: NONE leaks. simNow() is rng.js's VIRTUAL clock (frame*frameMs);
// the live page sets frameMs=REFRESH_INTERVAL_IN_MS (18 mobile / 12 desktop),
// the harness pins 12. lastFeedTime/nowMs ride through snapshots into rollouts
// but are never read back into any decision (and decaySize() is never called),
// so frameMs is causally dead — verified by --selftest check 4 (frameMs 12 vs
// 18 → identical decisions + force bits; only the dead lastFeed/nowMs fields
// scale). Browser behavior identical modulo those dead fields.
//
//   node dev/exact_nn/oracle_logger.js --policyDir js --W 390 --H 844 \
//        --seedStart 100000 --seeds 8 --maxFrames 24000 --outDir dev/exact_nn/data \
//        --shard s100000_390x844
//   node dev/exact_nn/oracle_logger.js --selftest [--policyDir js]
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { createGame } = require('./stepper.js');

// ---- exact-string anchored hook injection (in-memory only) ----
const ANCHORS = [
    { find: 'var vprior = cp_value(NET, fr.feat, fr.ctx);',
      add: ' if (window.__oracle) window.__oracle.planStart(s, cands, fr, vprior);', where: 'after' },
    { find: 'score[ci] = rr.catches + boot;',
      add: ' if (window.__oracle) window.__oracle.roll(ci, rr.catches, boot);', where: 'after' },
    { find: 'if (vizModel) cp_value_viz(NET, fr.feat[bi], fr.ctx, vizModel);',
      add: 'if (window.__oracle) window.__oracle.planEnd(pidx, score, bi); ', where: 'before' },
];
function oracleTransform(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    for (const a of ANCHORS) {
        const n = code.split(a.find).length - 1;
        if (n !== 1) throw new Error(`oracle anchor matched ${n}x (want 1): "${a.find}"`);
        code = code.replace(a.find, a.where === 'after' ? a.find + a.add : a.add + a.find);
    }
    return code;
}

// ---- float64 exact bits ----
const _dv = new DataView(new ArrayBuffer(8));
function f64hex(x) { _dv.setFloat64(0, x, true); return _dv.getBigUint64(0, true).toString(16).padStart(16, '0'); }
let negZeroCount = 0; // -0 occurrences across logged floats (see header note)
function assertCleanNum(x, what) {
    if (!Number.isFinite(x)) throw new Error(`non-finite float in ${what}: ${x}`);
    if (x === 0 && 1 / x < 0) negZeroCount++;
    return x;
}
function cleanArr(a, what) { for (let i = 0; i < a.length; i++) assertCleanNum(a[i], what); return a; }

// ---- per-plan reconstruction cross-check (replicates planCheap's own
// pidx sort + argmax from the logged values; any divergence = hook bug) ----
function recheckPlan(rec) {
    const ps = rec.feat.map(r => r[18] - r[16]);
    const pidx = rec.feat.map((_, k) => k).sort((a, b) => (ps[b] - ps[a]) || (a - b));
    if (pidx.length !== rec.pidx.length || pidx.some((v, i) => v !== rec.pidx[i]))
        throw new Error('recheck: pidx mismatch');
    const score = rec.vprior.slice();
    for (const [ci, catches, boot] of rec.rolled) score[ci] = catches + boot;
    if (score.length !== rec.score.length || score.some((v, i) => v !== rec.score[i]))
        throw new Error('recheck: score mismatch');
    let bi = 0, bs = -Infinity;
    for (let k = 0; k < score.length; k++) if (score[k] > bs) { bs = score[k]; bi = k; }
    if (bi !== rec.bi) throw new Error('recheck: bi mismatch');
    let ru = -Infinity;
    for (let k = 0; k < score.length; k++) if (k !== bi && score[k] > ru) ru = score[k];
    const margin = score[bi] - ru;
    if (margin !== rec.margin) throw new Error('recheck: margin mismatch');
}

// ---- run one game under the oracle; returns {decisions, frames, meta} ----
// decisions: array of plan records; frames: columnar per-frame log.
async function runGame(opt) {
    const pending = { plan: null, rolled: null };
    const decisions = [];
    const frames = { mode: [], N: [], fx: [], fy: [] };
    global.window = undefined; // stepper rebuilds; ensure no stale __oracle
    const g = await createGame({
        policyDir: opt.policyDir, W: opt.W, H: opt.H, seed: opt.seed,
        frameMs: opt.frameMs, transform: opt.pristine ? null : oracleTransform,
    });
    let sawPlan = false;
    if (!opt.pristine) {
        g.win.__oracle = {
            planStart(s, cands, fr, vprior) {
                sawPlan = true;
                pending.plan = {
                    seed: opt.seed, W: opt.W, H: opt.H, f: g.frame() + 1, N: s.bx.length,
                    s: { px: s.px, py: s.py, pvx: s.pvx, pvy: s.pvy, psize: s.psize,
                         lastFeed: s.lastFeed, nowMs: s.nowMs,
                         bx: cleanArr(s.bx.slice(), 'bx'), by: cleanArr(s.by.slice(), 'by'),
                         bvx: cleanArr(s.bvx.slice(), 'bvx'), bvy: cleanArr(s.bvy.slice(), 'bvy') },
                    cands: cands.map(c => [assertCleanNum(c.x, 'cand'), assertCleanNum(c.y, 'cand')]),
                    feat: fr.feat.map(r => cleanArr(r.slice(), 'feat')),
                    ctx: cleanArr(fr.ctx.slice(), 'ctx'),
                    vprior: cleanArr(vprior.slice(), 'vprior'),
                };
                pending.rolled = [];
            },
            roll(ci, catches, boot) { pending.rolled.push([ci, catches, assertCleanNum(boot, 'boot')]); },
            planEnd(pidx, score, bi) {
                const rec = pending.plan;
                rec.pidx = pidx.slice();
                rec.rolled = pending.rolled;
                rec.score = cleanArr(score.slice(), 'score');
                rec.bi = bi;
                let ru = -Infinity;
                for (let k = 0; k < score.length; k++) if (k !== bi && score[k] > ru) ru = score[k];
                rec.margin = score[bi] - ru;
                recheckPlan(rec);
                decisions.push(rec);
                pending.plan = pending.rolled = null;
            },
        };
        // per-frame mode + force bits via a pure wrapper (no source change)
        const realForce = g.win.__cheap.force;
        g.win.__cheap.force = function (pred, boids) {
            const N = boids.length;
            sawPlan = false;
            const out = realForce.call(this, pred, boids);
            const mode = N === 0 ? 0 : (N <= 5 ? 1 : (sawPlan ? 2 : 3));
            frames.mode.push(mode); frames.N.push(N);
            frames.fx.push(f64hex(out.x)); frames.fy.push(f64hex(out.y));
            return out;
        };
    }
    const traj = opt.captureTraj ? [] : null;
    let clearedAt = -1;
    for (let f = 0; f < opt.maxFrames; f++) {
        g.stepFrame();
        if (traj) traj.push(f64hex(g.sim.predator.position.x) + f64hex(g.sim.predator.position.y));
        if (g.boidCount() === 0) { clearedAt = g.frame(); break; }
    }
    return { decisions, frames, traj,
        meta: { seed: opt.seed, W: opt.W, H: opt.H, frameMs: opt.frameMs == null ? 12 : opt.frameMs,
                framesRun: g.frame(), clearedAt, eaten: g.eaten(), nPlans: decisions.length } };
}

// ---- shard writer ----
function writeShard(file, lines) {
    const gz = zlib.gzipSync(Buffer.from(lines.join('\n') + (lines.length ? '\n' : ''), 'utf8'), { level: 6 });
    fs.writeFileSync(file, gz);
}

async function produce(opt) {
    fs.mkdirSync(opt.outDir, { recursive: true });
    const dec = [], frm = [], metas = [];
    for (let i = 0; i < opt.seeds; i++) {
        const seed = opt.seedStart + i;
        const r = await runGame({ ...opt, seed });
        for (const d of r.decisions) dec.push(JSON.stringify(d));
        frm.push(JSON.stringify({ seed, W: opt.W, H: opt.H, n: r.frames.mode.length,
            mode: r.frames.mode, N: r.frames.N, fx: r.frames.fx, fy: r.frames.fy }));
        metas.push(r.meta);
        if (opt.verbose) console.error(`seed ${seed}: frames=${r.meta.framesRun} plans=${r.meta.nPlans} cleared=${r.meta.clearedAt > 0}`);
    }
    const base = path.join(opt.outDir, opt.shard);
    writeShard(base + '.decisions.jsonl.gz', dec);
    writeShard(base + '.frames.jsonl.gz', frm);
    fs.writeFileSync(base + '.meta.json', JSON.stringify({
        policyDir: path.resolve(opt.policyDir), W: opt.W, H: opt.H,
        seedStart: opt.seedStart, seeds: opt.seeds, maxFrames: opt.maxFrames,
        frameMs: opt.frameMs == null ? 12 : opt.frameMs, games: metas,
        nDecisions: dec.length, nNegZero: negZeroCount,
    }, null, 1));
    console.log(JSON.stringify({ shard: opt.shard, games: opt.seeds, decisions: dec.length }));
}

// ---- selftest: the acceptance checks, end to end ----
async function selftest(policyDir) {
    const cases = [
        { W: 390, H: 844, seed: 100000, maxFrames: 9000 },   // mobile, 60 boids
        { W: 1512, H: 982, seed: 100001, maxFrames: 4000 },  // desktop 120, partial game
    ];
    let pass = true;
    const report = [];
    for (const c of cases) {
        // 1. pristine equivalence: instrumented vs untouched → identical trajectory bits
        const a = await runGame({ ...c, policyDir, captureTraj: true });
        const b = await runGame({ ...c, policyDir, captureTraj: true, pristine: true });
        const eq = a.traj.length === b.traj.length && a.traj.every((v, i) => v === b.traj[i]);
        report.push(`[${c.W}x${c.H} seed ${c.seed}] pristine-equivalence: ${eq ? 'PASS' : 'FAIL'} (${a.traj.length} frames, ${a.meta.nPlans} plans)`);
        pass = pass && eq;
        // 2. determinism: run twice → byte-identical serialized logs
        const a2 = await runGame({ ...c, policyDir });
        const s1 = JSON.stringify(a.decisions) + JSON.stringify(a.frames);
        const s2 = JSON.stringify(a2.decisions) + JSON.stringify(a2.frames);
        const det = s1 === s2;
        report.push(`[${c.W}x${c.H} seed ${c.seed}] determinism (byte-identical logs): ${det ? 'PASS' : 'FAIL'}`);
        pass = pass && det;
        // 3. recheckPlan ran inline on every decision (throws on mismatch) — count it
        report.push(`[${c.W}x${c.H} seed ${c.seed}] replay recompute (pidx/score/bi/margin): PASS (${a.decisions.length} plans cross-checked)`);
        // 4. frameMs deadness: 12 vs 18 → identical decisions+forces modulo dead clock fields
        const m18 = await runGame({ ...c, policyDir, frameMs: 18 });
        const strip = d => JSON.stringify(d.map(r => ({ ...r, s: { ...r.s, lastFeed: 0, nowMs: 0 } })));
        const dead = strip(a.decisions) === strip(m18.decisions)
            && JSON.stringify(a.frames) === JSON.stringify(m18.frames);
        report.push(`[${c.W}x${c.H} seed ${c.seed}] frameMs causally dead (12 vs 18): ${dead ? 'PASS' : 'FAIL'}`);
        pass = pass && dead;
    }
    console.log(report.join('\n'));
    console.log(pass ? 'SELFTEST: ALL PASS' : 'SELFTEST: FAILURES');
    process.exit(pass ? 0 : 1);
}

function parseArgs(argv) {
    const a = { policyDir: path.join(__dirname, '..', '..', 'js'), W: 390, H: 844,
        seedStart: 100000, seeds: 4, maxFrames: 24000, outDir: path.join(__dirname, 'data'),
        shard: null, frameMs: null, verbose: false, selftest: false };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--policyDir') a.policyDir = argv[++i];
        else if (k === '--W') a.W = +argv[++i];
        else if (k === '--H') a.H = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--outDir') a.outDir = argv[++i];
        else if (k === '--shard') a.shard = argv[++i];
        else if (k === '--frameMs') a.frameMs = +argv[++i];
        else if (k === '--verbose') a.verbose = true;
        else if (k === '--selftest') a.selftest = true;
        else throw new Error('unknown arg: ' + k);
    }
    if (!a.shard) a.shard = `s${a.seedStart}_n${a.seeds}_${a.W}x${a.H}`;
    return a;
}

async function main() {
    const opt = parseArgs(process.argv);
    if (opt.selftest) return selftest(opt.policyDir);
    // train/verify seed discipline: held-out seeds >=270000 are RESERVED.
    if (opt.seedStart + opt.seeds > 270000 && !process.env.ORACLE_ALLOW_HELDOUT)
        throw new Error('seed range crosses 270000 (held-out verification seeds) — refuse');
    return produce(opt);
}
main().catch(e => { console.error(e); process.exit(1); });
