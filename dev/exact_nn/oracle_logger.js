// exact_nn oracle decision-logger (#5, spec rev 2) — records every plan
// decision of the FROZEN prod policy (workspace js/ ≡ main@6dce76f) for
// offline student training.
//
// Architecture (per the #5 spec revision): the sim is DRIVEN by the CERTIFIED
// INSTRUMENTED FORK dev/exact_nn/oracle_policy.js (generated from
// js/predator_cheap.js by gen_oracle_policy.js — logging lines only, see its
// header). planCheap's internals (feat/vprior/pidx/score/bi) are closure-local
// and unreachable from any wrapper, so the hooks live inside the fork.
// Bit-identity of the hooked fork vs pristine prod is proven by
// certify_oracle.js (lockstep + fork-trajectory, all device cells, gate
// crossings, spawn games); this logger REFUSES to farm unless CERT.json
// matches the current fork sha. The cert run id is embedded in every shard.
//
// Per plan decision (one JSONL line):
//   seed,cell,W,H, f (1-based sim frame), N,
//   s     : full input snapshot {px,py,pvx,pvy,psize,lastFeed,nowMs,bx,by,bvx,bvy}
//   cands : the 16 candidate targets [[x,y]...]
//   feat  : 16x19 cp_features matrix; ctx: 4 context features
//   vprior: 16 value-net priors
//   pidx  : roll order (ballistic pscore sort; top-K_roll=4 get rolled)
//   rolled: [[ci, rollout catches, terminal bootstrap]...]
//   score : final 16 scores (vprior with rolled overrides)
//   bi    : prod's argmax slot;  margin: slot-level runner-up margin
//   lab   : {ti,tx,ty} — THE LABEL: committed target COORDINATES, slot index
//           canonicalized to the LOWEST k with bitwise-equal (x,y) (slots
//           k>=N are E3D copies whenever N<=15, so raw indices alias)
//   dmargin: top1−top2 margin over coordinate-DEDUPED candidate groups
//           (group score = max over its slots; >= margin by construction)
//   nDistinct: distinct coordinate groups among the 16 slots
// Every decision is replay-verified inline (recheckPlan recomputes
// pidx/score/bi/margin/lab/dmargin from the logged pieces; throws on any bit
// of disagreement).
//
// Per frame (one columnar JSONL line per game):
//   mode (0=zero 1=intercept 2=plan 3=steer), N, force fx/fy as exact float64
//   bit patterns (hex u64, DataView — catches -0/NaN where JSON can't),
//   pf (policy's internal plan-cycle frame counter), tx/ty (committed target
//   coords as hex, '' before first commit), eg (egBoid index in the live
//   boids array, -1 when unset).
//
// Floats in decision records are plain JSON numbers: ECMAScript serialization
// is shortest-round-trip, so parsing returns the identical float64 — except
// -0 (parses +0). Negative zeros DO occur (counted per shard, meta nNegZero)
// and are provably causally dead downstream (no 1/x, atan2, Object.is in the
// policy path).
//
// Two fields can be non-finite, both serialized by JSON.stringify as `null`
// with a FIELD-SPECIFIC meaning (each consumer knows its field's domain):
//   * score[k] / rolled[i][2] (boot): may be -Infinity (extermination path,
//     see scoreNum) → null means -Inf. These are argmax-losers, masked in
//     regression, never the label. Counted per shard as nNegInf.
//   * dmargin: may be +Infinity when the committed coordinate has NO competing
//     distinct-coordinate group (single-group plan, or all competitors are
//     -Inf-exterminated) → null means +Inf (maximally safe, never a near-tie).
// dmargin is always ≥0 (bi is the global argmax). NaN and +Inf anywhere else
// remain hard errors.
//
// Wall-clock: NONE leaks. simNow() is rng.js's VIRTUAL clock (frame*frameMs).
// The browser uses frameMs=18 (mobile) / 12 (desktop); cells run their
// faithful value (device_matrix.js). lastFeedTime/nowMs ride through
// snapshots but are never read back into any decision (decaySize is never
// called) — re-proven by --selftest check 4 (frameMs 12 vs 18 → identical
// decisions + force bits modulo the dead clock fields).
//
// Spawn profiles (--spawn, the live page's tap-to-spawn, SPEC corpus):
//   none    (default)
//   mid     scripted taps during the planner phase incl. same-coord double-tap
//   recross REACTIVE: when N first reaches 5, taps at +40 (double, same
//           coord) and +44 force 5->6->7->...->5 gate re-crossings with
//           egBoid/frame-counter state alive across them. The realized
//           schedule {frame,x,y} is recorded in the shard meta (replayable).
//   spam    48 taps in 48 consecutive frames (exceeds the device boid cap)
//
//   node dev/exact_nn/oracle_logger.js --cell iphone_390x844 \
//        --seedStart 100000 --seeds 8 --outDir dev/exact_nn/data --shard <name>
//   node dev/exact_nn/oracle_logger.js --selftest
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const crypto = require('crypto');
const { createGame } = require('./stepper.js');
const { loadFork, FORK } = require('./oracle_candidate.js');
const { CELLS, HELD_OUT_SEED } = require('./device_matrix.js');

// ---- float64 exact bits ----
const _dv = new DataView(new ArrayBuffer(8));
function f64hex(x) { _dv.setFloat64(0, x, true); return _dv.getBigUint64(0, true).toString(16).padStart(16, '0'); }
let negZeroCount = 0; // -0 occurrences across logged floats (see header note)
let negInfCount = 0;  // -Infinity rolled scores (extermination path; see below)
function assertCleanNum(x, what) {
    if (!Number.isFinite(x)) throw new Error(`non-finite float in ${what}: ${x}`);
    if (x === 0 && 1 / x < 0) negZeroCount++;
    return x;
}
function cleanArr(a, what) { for (let i = 0; i < a.length; i++) assertCleanNum(a[i], what); return a; }

// Score-path numbers MAY be -Infinity: prod's bootstrap is
// `boot=-Infinity; for(j) if(tv[j]>boot) boot=tv[j]` over the terminal value
// net, and when a rolled candidate's rollout exterminates every boid the
// terminal features go NaN, every tv[j] is NaN, no `>` fires, and boot stays
// -Infinity (SPEC §4 "NaN→−Infinity extermination path"). Then
// score[ci]=catches+(-Inf)=-Inf. This is a LEGITIMATE, load-bearing value, not
// a bug — so the score path permits -Infinity (counted) but still rejects NaN
// and +Infinity (those would be real corruption). JSON.stringify maps
// -Infinity→null; the packer maps null→-inf (these scores are argmax-losers,
// masked in regression, never the label). The winner's score is always finite
// (≥12 candidates keep their finite vprior), so bi and the coordinate label are
// unaffected.
function scoreNum(x, what) {
    if (x === -Infinity) { negInfCount++; return x; }
    if (!Number.isFinite(x)) throw new Error(`+Inf/NaN in ${what}: ${x}`);
    if (x === 0 && 1 / x < 0) negZeroCount++;
    return x;
}
function scoreArr(a, what) { for (let i = 0; i < a.length; i++) scoreNum(a[i], what); return a; }

// ---- label + dedup margin (coordinate identity = exact f64 bit pairs) ----
function coordKey(c) { return f64hex(c.x) + f64hex(c.y); }
function labelAndDedup(cands, score, bi) {
    const keys = cands.map(coordKey);
    const win = keys[bi];
    let ti = -1;
    const groupMax = new Map();
    for (let k = 0; k < cands.length; k++) {
        if (ti < 0 && keys[k] === win) ti = k;
        const g = groupMax.get(keys[k]);
        if (g === undefined || score[k] > g) groupMax.set(keys[k], score[k]);
    }
    let ru = -Infinity;
    for (const [key, s] of groupMax) if (key !== win && s > ru) ru = s;
    return { ti, dmargin: score[bi] - ru, nDistinct: groupMax.size };
}

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
    if (score[bi] - ru !== rec.margin) throw new Error('recheck: margin mismatch');
    // label invariants: lowest bitwise-equal slot; committed coords match; dedup
    // margin recomputes exactly and can only exceed the slot-level margin.
    const cands = rec.cands.map(c => ({ x: c[0], y: c[1] }));
    const ld = labelAndDedup(cands, score, bi);
    if (ld.ti !== rec.lab.ti || ld.dmargin !== rec.dmargin || ld.nDistinct !== rec.nDistinct)
        throw new Error('recheck: label/dedup mismatch');
    if (rec.lab.ti > rec.bi) throw new Error('recheck: ti above bi');
    if (f64hex(cands[rec.lab.ti].x) !== rec.lab.tx || f64hex(cands[rec.lab.ti].y) !== rec.lab.ty)
        throw new Error('recheck: label coords mismatch');
    if (!(rec.dmargin >= rec.margin)) throw new Error('recheck: dmargin < margin');
}

// ---- certification gate ----
function certInfo() {
    const oracleSha = crypto.createHash('sha256').update(fs.readFileSync(FORK)).digest('hex');
    const certPath = path.join(__dirname, 'CERT.json');
    if (process.env.ORACLE_SKIP_CERT) return { oracleSha, certRunId: 'UNCERTIFIED-DEV' };
    if (!fs.existsSync(certPath)) throw new Error('no CERT.json — run certify_oracle.js first');
    const cert = JSON.parse(fs.readFileSync(certPath, 'utf8'));
    if (!cert.ok) throw new Error('CERT.json says ok=false — fork is NOT certified');
    if (cert.oracleSha !== oracleSha)
        throw new Error('CERT.json is for a different oracle_policy.js — re-certify');
    return { oracleSha, certRunId: cert.certRunId };
}

// ---- spawn profiles (deterministic; coords scale with the cell) ----
function spawnPlan(profile, cell) {
    if (!profile || profile === 'none') return { script: null, reactive: false };
    if (profile === 'mid') {
        const x = cell.W * 0.25, y = cell.H * 0.25;
        return { reactive: false, script: [
            { frame: 400, x, y }, { frame: 400, x, y }, { frame: 401, x, y },
            { frame: 1200, x: cell.W * 0.7, y: cell.H * 0.6 },
            { frame: 2000, x: cell.W * 0.1, y: cell.H * 0.9 },
        ] };
    }
    if (profile === 'spam') {
        const script = [];
        for (let k = 0; k < 48; k++) {
            script.push({ frame: 300 + k, x: cell.W * (0.2 + 0.6 * ((k % 5) / 4)),
                          y: cell.H * (0.2 + 0.6 * (((k * 7) % 9) / 8)) });
        }
        return { script, reactive: false };
    }
    if (profile === 'recross') return { script: null, reactive: true };
    throw new Error('unknown spawn profile: ' + profile);
}

// ---- run one game under the oracle; returns {decisions, frames, meta} ----
async function runGame(opt) {
    const cell = opt.cell;
    const pending = { plan: null, rolled: null };
    const decisions = [];
    const frames = { mode: [], N: [], fx: [], fy: [], pf: [], tx: [], ty: [], eg: [] };
    const sp = spawnPlan(opt.spawn, cell);
    const realizedSpawns = sp.script ? sp.script.slice() : [];
    const g = await createGame({
        policyDir: opt.policyDir, W: cell.W, H: cell.H, seed: opt.seed,
        startBoids: cell.startBoids, frameMs: opt.frameMs != null ? opt.frameMs : cell.frameMs,
        fastRender: true, spawnScript: sp.script,
    });
    let sawPlan = false;
    // derived-config vector (SPEC §1): captured from the policy's OWN cfg object
    // (as-evaluated) + the engine globals. Constant per game; stamped on every
    // record so a shard is self-describing and labels never alias across cells.
    const numBoids = g.api.getNumBoids();
    if (!opt.pristine) {
        const forkCheap = await loadFork(g);
        g.win.__oracle = {
            planStart(s, cands, fr, vprior, cfg) {
                sawPlan = true;
                pending.plan = {
                    seed: opt.seed, cell: cell.id, W: cell.W, H: cell.H,
                    cfg: { W: cfg.W, Hc: cfg.Hc, PREDATOR_RANGE: cfg.POLICY_R, NUM_BOIDS: numBoids },
                    f: g.frame() + 1, N: s.bx.length,
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
            roll(ci, catches, boot) { pending.rolled.push([ci, assertCleanNum(catches, 'catches'), scoreNum(boot, 'boot')]); },
            planEnd(pidx, score, bi) {
                const rec = pending.plan;
                rec.pidx = pidx.slice();
                rec.rolled = pending.rolled;
                rec.score = scoreArr(score.slice(), 'score');   // -Infinity allowed (extermination)
                rec.bi = bi;
                // the winner's score is finite by construction (≥12 candidates
                // keep their finite vprior; only the ≤4 rolled losers can be
                // -Inf). If this ever fires, an exterminated candidate became the
                // label — a real anomaly, not the masked-loser case. Catch it.
                if (!Number.isFinite(score[bi])) throw new Error(`winner score non-finite: ${score[bi]} (bi=${bi})`);
                let ru = -Infinity;
                for (let k = 0; k < score.length; k++) if (k !== bi && score[k] > ru) ru = score[k];
                rec.margin = score[bi] - ru;
                const cands = rec.cands.map(c => ({ x: c[0], y: c[1] }));
                const ld = labelAndDedup(cands, score, bi);
                rec.lab = { ti: ld.ti, tx: f64hex(cands[ld.ti].x), ty: f64hex(cands[ld.ti].y) };
                rec.dmargin = ld.dmargin;
                rec.nDistinct = ld.nDistinct;
                recheckPlan(rec);
                decisions.push(rec);
                pending.plan = pending.rolled = null;
            },
            // per-frame persistent closure state + the returned force
            frameEnd(target, pframe, egBoid, boids, r) {
                const N = boids.length;
                frames.mode.push(N === 0 ? 0 : (N <= 5 ? 1 : (sawPlan ? 2 : 3)));
                frames.N.push(N);
                frames.fx.push(f64hex(r.x)); frames.fy.push(f64hex(r.y));
                frames.pf.push(pframe);
                frames.tx.push(target ? f64hex(target.x) : '');
                frames.ty.push(target ? f64hex(target.y) : '');
                frames.eg.push(egBoid ? boids.indexOf(egBoid) : -1);
                sawPlan = false;
            },
        };
        g.setForce(forkCheap.force);
    }
    const traj = opt.captureTraj ? [] : null;
    let clearedAt = -1, recrossArmed = sp.reactive, recrossSpawned = false;
    for (let f = 0; f < opt.maxFrames; f++) {
        if (recrossArmed && g.boidCount() <= 5) {
            // reactive endgame taps: schedule relative to NOW (frame g.frame())
            const f0 = g.frame();
            const taps = [
                { frame: f0 + 40, x: cell.W * 0.5, y: cell.H * 0.2 },
                { frame: f0 + 40, x: cell.W * 0.5, y: cell.H * 0.2 },
                { frame: f0 + 44, x: cell.W * 0.8, y: cell.H * 0.8 },
            ];
            realizedSpawns.push(...taps);
            recrossArmed = false; recrossSpawned = true;
            // inject via the sim directly at the scheduled frames below
            opt._taps = taps;
        }
        if (opt._taps) {
            for (const t of opt._taps) if (t.frame === g.frame()) g.sim.spawnBoid(t.x, t.y);
        }
        g.step();
        if (traj) traj.push(f64hex(g.sim.predator.position.x) + f64hex(g.sim.predator.position.y));
        if (g.boidCount() === 0) { clearedAt = g.frame(); break; }
    }
    delete opt._taps;
    return { decisions, frames, traj,
        meta: { seed: opt.seed, cell: cell.id, W: cell.W, H: cell.H,
                numBoids: g.api.getNumBoids(), predRange: g.api.getPredRange(),
                frameMs: opt.frameMs != null ? opt.frameMs : cell.frameMs,
                spawn: opt.spawn || 'none', spawns: realizedSpawns,
                recrossSpawned,
                framesRun: g.frame(), clearedAt, eaten: g.sim.boidsEaten, nPlans: decisions.length } };
}

// ---- shard writer ----
function writeShard(file, lines) {
    const gz = zlib.gzipSync(Buffer.from(lines.join('\n') + (lines.length ? '\n' : ''), 'utf8'), { level: 6 });
    fs.writeFileSync(file, gz);
}

async function produce(opt) {
    const ci = certInfo();   // refuses to farm uncertified
    fs.mkdirSync(opt.outDir, { recursive: true });
    const dec = [], frm = [], metas = [];
    for (let i = 0; i < opt.seeds; i++) {
        const seed = opt.seedStart + i;
        const r = await runGame({ ...opt, seed });
        for (const d of r.decisions) dec.push(JSON.stringify(d));
        frm.push(JSON.stringify({ seed, cell: opt.cell.id, n: r.frames.mode.length,
            mode: r.frames.mode, N: r.frames.N, fx: r.frames.fx, fy: r.frames.fy,
            pf: r.frames.pf, tx: r.frames.tx, ty: r.frames.ty, eg: r.frames.eg }));
        metas.push(r.meta);
        if (opt.verbose) console.error(`seed ${seed}: frames=${r.meta.framesRun} plans=${r.meta.nPlans} cleared=${r.meta.clearedAt > 0}`);
    }
    const base = path.join(opt.outDir, opt.shard);
    writeShard(base + '.decisions.jsonl.gz', dec);
    writeShard(base + '.frames.jsonl.gz', frm);
    fs.writeFileSync(base + '.meta.json', JSON.stringify({
        policyDir: path.resolve(opt.policyDir),
        cell: opt.cell.id, W: opt.cell.W, H: opt.cell.H,
        numBoids: metas[0] && metas[0].numBoids, predRange: metas[0] && metas[0].predRange,
        frameMs: opt.frameMs != null ? opt.frameMs : opt.cell.frameMs,
        spawn: opt.spawn || 'none',
        seedStart: opt.seedStart, seeds: opt.seeds, maxFrames: opt.maxFrames,
        oracleSha: ci.oracleSha, certRunId: ci.certRunId, node: process.version,
        games: metas, nDecisions: dec.length, nNegZero: negZeroCount, nNegInf: negInfCount,
    }, null, 1));
    console.log(JSON.stringify({ shard: opt.shard, games: opt.seeds, decisions: dec.length }));
}

// ---- selftest: the logger-level acceptance checks, end to end ----
// (bit-identity of the fork itself is certify_oracle.js's job; here we prove
// the LOGGER: non-perturbation, determinism, replay recompute, frameMs
// deadness, spawn-profile determinism.)
async function selftest() {
    const cases = [
        { cell: CELLS[0], seed: 100000, maxFrames: 9000 },   // iphone, 60 boids
        { cell: CELLS[3], seed: 100001, maxFrames: 4000 },   // desktop 120, partial game
        { cell: CELLS[1], seed: 100002, maxFrames: 6000 },   // ipad: UA-mobile, 60 boids @820x1180
    ];
    const policyDir = path.join(__dirname, '..', '..', 'js');
    let pass = true;
    const report = [];
    for (const c of cases) {
        const tag = `[${c.cell.id} seed ${c.seed}]`;
        // 1. pristine equivalence: fork-driven (hooks on) vs pristine prod-driven
        const a = await runGame({ ...c, policyDir, captureTraj: true });
        const b = await runGame({ ...c, policyDir, captureTraj: true, pristine: true });
        const eq = a.traj.length === b.traj.length && a.traj.every((v, i) => v === b.traj[i]);
        report.push(`${tag} fork-vs-pristine trajectory bits: ${eq ? 'PASS' : 'FAIL'} (${a.traj.length} frames, ${a.meta.nPlans} plans)`);
        pass = pass && eq;
        // 2. determinism: run twice → byte-identical serialized logs
        const a2 = await runGame({ ...c, policyDir });
        const det = JSON.stringify(a.decisions) + JSON.stringify(a.frames)
                === JSON.stringify(a2.decisions) + JSON.stringify(a2.frames);
        report.push(`${tag} determinism (byte-identical logs): ${det ? 'PASS' : 'FAIL'}`);
        pass = pass && det;
        // 3. recheckPlan ran inline on every decision (throws on mismatch)
        report.push(`${tag} replay recompute incl. label+dedup margin: PASS (${a.decisions.length} plans)`);
        // 4. frameMs deadness: cell value vs 12 → identical modulo dead clock fields
        const m12 = await runGame({ ...c, policyDir, frameMs: 12 });
        const strip = d => JSON.stringify(d.map(r => ({ ...r, s: { ...r.s, lastFeed: 0, nowMs: 0 } })));
        const dead = strip(a.decisions) === strip(m12.decisions)
            && JSON.stringify(a.frames) === JSON.stringify(m12.frames);
        report.push(`${tag} frameMs causally dead (${c.cell.frameMs} vs 12): ${dead ? 'PASS' : 'FAIL'}`);
        pass = pass && dead;
    }
    // 5. spawn-profile determinism (reactive recross): identical realized
    // schedule + logs across two runs; and the recross actually happened
    const rc = { cell: CELLS[0], seed: 100003, maxFrames: 24000, spawn: 'recross' };
    const policy = path.join(__dirname, '..', '..', 'js');
    const r1 = await runGame({ ...rc, policyDir: policy });
    const r2 = await runGame({ ...rc, policyDir: policy });
    const sdet = JSON.stringify(r1.meta.spawns) === JSON.stringify(r2.meta.spawns)
        && JSON.stringify(r1.frames) === JSON.stringify(r2.frames)
        && r1.meta.recrossSpawned;
    report.push(`[${rc.cell.id} seed ${rc.seed}] recross spawn determinism + occurred: ${sdet ? 'PASS' : 'FAIL'} (${r1.meta.spawns.length} taps, ${r1.meta.framesRun} frames)`);
    pass = pass && sdet;

    // 6. non-finite score-path contract (extermination path may not trigger in
    // the seeds above, so assert the encode/decode invariants directly):
    //  - scoreNum permits -Infinity (counts it), rejects NaN and +Infinity;
    //  - JSON.stringify maps -Inf and +Inf to null, and the field-specific
    //    decode (score: null→-Inf; dmargin: null→+Inf) round-trips the meaning;
    //  - a winner that is -Inf is a hard error (planEnd guard).
    let nf = true;
    try { scoreNum(-Infinity, 't'); } catch { nf = false; }            // -Inf allowed
    for (const bad of [NaN, Infinity]) { try { scoreNum(bad, 't'); nf = false; } catch {} }  // must throw
    const rt = JSON.parse(JSON.stringify({ score: [-Infinity, 1.5], dmargin: Infinity }));
    if (!(rt.score[0] === null && rt.score[1] === 1.5 && rt.dmargin === null)) nf = false;
    const decScore = rt.score.map(v => v === null ? -Infinity : v);     // packer rule
    const decDm = rt.dmargin === null ? Infinity : rt.dmargin;          // analyze_margin rule
    if (!(decScore[0] === -Infinity && decScore[1] === 1.5 && decDm === Infinity)) nf = false;
    report.push(`non-finite score-path contract (-Inf score / +Inf dmargin round-trip): ${nf ? 'PASS' : 'FAIL'}`);
    pass = pass && nf;
    console.log(report.join('\n'));
    console.log(pass ? 'SELFTEST: ALL PASS' : 'SELFTEST: FAILURES');
    process.exit(pass ? 0 : 1);
}

function parseArgs(argv) {
    const a = { policyDir: path.join(__dirname, '..', '..', 'js'), cell: null,
        seedStart: 100000, seeds: 4, maxFrames: null, outDir: path.join(__dirname, 'data'),
        shard: null, frameMs: null, spawn: 'none', verbose: false, selftest: false };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--policyDir') a.policyDir = argv[++i];
        else if (k === '--cell') a.cell = argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--outDir') a.outDir = argv[++i];
        else if (k === '--shard') a.shard = argv[++i];
        else if (k === '--frameMs') a.frameMs = +argv[++i];
        else if (k === '--spawn') a.spawn = argv[++i];
        else if (k === '--verbose') a.verbose = true;
        else if (k === '--selftest') a.selftest = true;
        else throw new Error('unknown arg: ' + k);
    }
    if (!a.selftest) {
        const cell = CELLS.find(c => c.id === a.cell);
        if (!cell) throw new Error('--cell required, one of: ' + CELLS.map(c => c.id).join(', '));
        a.cell = cell;
        if (a.maxFrames == null) a.maxFrames = cell.maxFrames;
        if (!a.shard) a.shard = `train_${cell.id}_s${a.seedStart}${a.spawn !== 'none' ? '_' + a.spawn : ''}`;
    }
    return a;
}

async function main() {
    const opt = parseArgs(process.argv);
    if (opt.selftest) return selftest();
    // train/verify seed discipline: held-out seeds >=270000 are RESERVED.
    if (opt.seedStart + opt.seeds > HELD_OUT_SEED && !process.env.ORACLE_ALLOW_HELDOUT)
        throw new Error(`seed range crosses ${HELD_OUT_SEED} (held-out verification seeds) — refuse`);
    return produce(opt);
}
main().catch(e => { console.error(e); process.exit(1); });
