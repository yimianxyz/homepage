// student_attack.js — SPEC §4c adversarial search: find legal states that
// MAXIMIZE the student's deduped top-2 margin SUBJECT TO student-argmax ≠
// prod-argmax (deduped, §3). The max such "trusted-but-wrong" margin vs the
// frozen τ is the residual-risk headline: if max-adversarial-margin < τ, the
// L1h fast path provably never commits a wrong target (the sealed 0-mismatch is
// robust beyond the corpus); if ≥ τ, confident-wrong states exist that τ admits
// — a real residual to report. (JS ground truth, per SPEC §5; GPU screening is
// only worth it at scale for the stronger student.)
//
// Attacks the STUDENT, not prod (SPEC §4c). Two phases:
//   1) random legal states (planner regime N∈[6,120]) → collect disagreements
//   2) hill-climb the highest-margin disagreements (perturb pred/boids, keep
//      disagreement, push student margin up) — finds the adversarial ceiling.
// prod's committed target is computed by prod's OWN planCheap (exposed via an
// anchored transform), exact rollout and all.
//
//   node student_attack.js --random 200000 --climb 200 --W 1024 --H 768 [--tau <τ>]
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { createGame } = require('../stepper.js');

// expose planCheap + candidates + configure (logging-only, like parity_dump)
function exposeTransform(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    const A = '    window.__cheap = {';
    if (code.indexOf(A) < 0) throw new Error('expose anchor not found');
    return code.replace(A, '    window.__prod = { plan: planCheap, candidates: candidates, configure: configure };\n' + A);
}

const _ck = new DataView(new ArrayBuffer(16));
function ckey(x, y) { _ck.setFloat64(0, x); _ck.setFloat64(8, y); return _ck.getBigUint64(0) + ':' + _ck.getBigUint64(8); }
function dedupArg(score, cands) {
    const best = new Map();
    for (let k = 0; k < score.length; k++) { const key = ckey(cands[k].x, cands[k].y); const c = best.get(key); if (!c || score[k] > c.s || (score[k] === c.s && k < c.idx)) best.set(key, { s: score[k], idx: k }); }
    const arr = Array.from(best.values()).sort((a, b) => (b.s - a.s) || (a.idx - b.idx));
    return { idx: arr[0].idx, margin: arr.length >= 2 ? arr[0].s - arr[1].s : Infinity, key: ckey(cands[arr[0].idx].x, cands[arr[0].idx].y) };
}
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

async function main() {
    const a = { random: 200000, climb: 200, W: 1024, H: 768, seed: 1, tau: null, out: null };
    for (let i = 2; i < process.argv.length; i++) { const k = process.argv[i];
        if (k === '--random') a.random = +process.argv[++i]; else if (k === '--climb') a.climb = +process.argv[++i];
        else if (k === '--W') a.W = +process.argv[++i]; else if (k === '--H') a.H = +process.argv[++i];
        else if (k === '--seed') a.seed = +process.argv[++i]; else if (k === '--tau') a.tau = +process.argv[++i];
        else if (k === '--out') a.out = process.argv[++i]; }
    if (a.tau == null) { const fp = path.join(__dirname, 'frozen_tau.json'); if (fs.existsSync(fp)) a.tau = JSON.parse(fs.readFileSync(fp, 'utf8')).chosenTau; }
    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    const { loadStudent } = require(path.join(__dirname, '..', 'student', 'studentScores.js'));
    const student = loadStudent(path.join(__dirname, '..', 'student', 'student_weights.json'));
    const cfg = { W: a.W, Hc: a.H, PREDATOR_RANGE: 80, NUM_BOIDS: a.W <= 768 ? 60 : 120 };

    // load prod with exposed internals
    const game = await createGame({ policyDir, W: a.W, H: a.H, seed: 1, fastRender: true });
    let code = fs.readFileSync(path.join(policyDir, 'predator_cheap.js'), 'utf8');
    const prevCheap = game.win.__cheap, prevReady = game.win.__predatorReady;
    vm.runInThisContext(exposeTransform('predator_cheap.js', code), { filename: 'predator_cheap.js#expose' });
    const prod = game.win.__prod; const expReady = game.win.__predatorReady;
    game.win.__cheap = prevCheap; game.win.__predatorReady = prevReady;
    if (expReady && expReady.then) await expReady;
    prod.configure({ canvasWidth: a.W, canvasHeight: a.H });

    const r = mulberry32(a.seed >>> 0), B = 20;
    function randState() {
        const N = 6 + (r() * 115 | 0);            // planner regime 6..120
        const s = { px: r() * a.W, py: r() * a.H, pvx: (r() * 2 - 1) * 2.5, pvy: (r() * 2 - 1) * 2.5,
            psize: 12 + r() * 9.6, bx: [], by: [], bvx: [], bvy: [], lastFeed: -1e9, nowMs: 0 };
        // clustered boids (realistic flock geometry → realistic candidates)
        const nc = Math.max(1, N / 15 | 0), cx = [], cy = [];
        for (let c = 0; c < nc; c++) { cx.push(r() * a.W); cy.push(r() * a.H); }
        for (let i = 0; i < N; i++) { const c = r() * nc | 0; s.bx.push(cx[c] + (r() * 2 - 1) * 60); s.by.push(cy[c] + (r() * 2 - 1) * 60);
            const ang = r() * 2 * Math.PI, sp = r() * 6; s.bvx.push(Math.cos(ang) * sp); s.bvy.push(Math.sin(ang) * sp); }
        return s;
    }
    function evaln(s) {
        const cands = prod.candidates(s);
        const sc = student({ px: s.px, py: s.py, pvx: s.pvx, pvy: s.pvy, psize: s.psize, bx: s.bx, by: s.by, bvx: s.bvx, bvy: s.bvy, nAlive: s.bx.length }, cands, cfg);
        const sd = dedupArg(sc, cands);
        const pt = prod.plan(s);                  // prod's EXACT committed target (rollout)
        const prodKey = ckey(pt.x, pt.y);
        return { disagree: sd.key !== prodKey, margin: sd.margin, cands };
    }
    function perturb(s) {                          // small legal perturbation (hill-climb step)
        const t = JSON.parse(JSON.stringify(s));
        const i = r() * t.bx.length | 0;
        t.bx[i] += (r() * 2 - 1) * 20; t.by[i] += (r() * 2 - 1) * 20;
        t.bvx[i] += (r() * 2 - 1) * 1.0; t.bvy[i] += (r() * 2 - 1) * 1.0;
        if (r() < 0.3) { t.px += (r() * 2 - 1) * 20; t.py += (r() * 2 - 1) * 20; }
        return t;
    }

    // phase 1: random search
    let maxMargin = -Infinity, best = null, disag = 0, n = 0;
    const buckets = { '>=tau': 0, '0.1': 0, '0.05': 0, '0.01': 0 };
    for (let it = 0; it < a.random; it++) {
        const s = randState(); const e = evaln(s); n++;
        if (e.disagree && Number.isFinite(e.margin)) {
            disag++;
            if (a.tau != null && e.margin >= a.tau) buckets['>=tau']++;
            if (e.margin >= 0.1) buckets['0.1']++; if (e.margin >= 0.05) buckets['0.05']++; if (e.margin >= 0.01) buckets['0.01']++;
            if (e.margin > maxMargin) { maxMargin = e.margin; best = s; }
        }
    }
    // phase 2: hill-climb the best disagreement to push margin up
    let climbed = maxMargin, climbState = best;
    if (best) {
        let cur = best, curM = maxMargin;
        for (let it = 0; it < a.climb * 50; it++) {
            const cand = perturb(cur); const e = evaln(cand);
            if (e.disagree && Number.isFinite(e.margin) && e.margin > curM) { cur = cand; curM = e.margin; }
        }
        climbed = curM; climbState = cur;
    }

    const report = {
        spec: 'SPEC §4c — student-attack: max student deduped-margin s.t. student≠prod (JS ground truth)',
        cell: a.W + 'x' + a.H, tau: a.tau,
        randomStates: n, disagreements: disag, disagreeRate: +(disag / n).toFixed(5),
        maxAdversarialMargin_random: maxMargin === -Infinity ? null : maxMargin,
        maxAdversarialMargin_climbed: climbed === -Infinity ? null : climbed,
        disagreementsAtMargin: buckets,
        verdict: (a.tau != null && climbed !== -Infinity)
            ? (climbed < a.tau
                ? `SAFE: max adversarial trusted-margin ${climbed.toExponential(3)} < τ ${a.tau} → L1h fast path never commits a wrong target (robust beyond the sealed corpus).`
                : `RESIDUAL: found student-confident-and-wrong at margin ${climbed.toExponential(3)} ≥ τ ${a.tau} — τ admits confident-wrong states; report this residual + the state.`)
            : 'no τ provided or no disagreements found',
    };
    console.log(JSON.stringify(report, null, 1));
    if (a.out) fs.writeFileSync(a.out, JSON.stringify(report, null, 1));
}

if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
