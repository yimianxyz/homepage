// unify_measure.js — ELEGANT UNIFICATION test (#5): does prod's ROLLOUT-PLANNER
// (planCheap), un-gated to N<=5, reproduce prod's intercept() scan egBoid?
//
// Measurement 1 (decision agreement S_dec(N<=5)): on the logged N<=5 endgame-commit
// states, run prod's planCheap (the rollout-planner, value-net NN genuinely in-loop),
// map its committed target candidate -> the boid it commits to, and compare to prod's
// actual intercept() egIdx. Reports by cell + by source (natural/scatter) + the
// E3D-patrol quirk rate (planCheap choosing the patrol point, not a boid).
//
// planCheap is closure-local in predator_cheap.js; we expose it by source-injection
// (window.__pcFork = planCheap) and capture cands+bi via the oracle hooks. One stepper
// game per CELL (planCheap reads cfg.W/Hc, set by configure()). Pure prod machinery —
// no NN of ours, no fallback edit. Float64.
//
//   node unify_measure.js [--heldOnlyNat] [--limit N]
'use strict';
const fs = require('fs'), zlib = require('zlib'), path = require('path'), vm = require('vm');
const { createGame } = require(path.join(__dirname, '..', 'stepper.js'));

const dv = new DataView(new ArrayBuffer(8));
const f64hex = x => { dv.setFloat64(0, x, true); return dv.getBigUint64(0, true).toString(16).padStart(16, '0'); };
const coordKey = (x, y) => f64hex(x) + f64hex(y);
const arg = (n) => process.argv.includes('--' + n);
const argv = (n, d) => { const i = process.argv.indexOf('--' + n); return i >= 0 ? process.argv[i + 1] : d; };
const LIMIT = parseInt(argv('limit', '0'), 10);

const CELLS = { '390x844': [390, 844], '820x1180': [820, 1180], '1024x768': [1024, 768],
    '1512x982': [1512, 982], '1680x1050': [1680, 1050], '2560x1440': [2560, 1440] };

// load all endgame commits, grouped by cell; tag source (0 data_eg,1 data_eg2,2 nat)
function loadCommits() {
    const dirs = ['data_eg', 'data_eg2', 'data_eg_nat'];
    const byCell = {};
    dirs.forEach((d, si) => {
        const dir = path.join(__dirname, d);
        if (!fs.existsSync(dir)) return;
        for (const f of fs.readdirSync(dir).filter(x => x.endsWith('.commits.jsonl.gz'))) {
            for (const line of zlib.gunzipSync(fs.readFileSync(path.join(dir, f))).toString().split('\n')) {
                if (!line) continue;
                const r = JSON.parse(line); r.src = si;
                if (r.boids.length < 1 || r.boids.length > 5) continue;
                (byCell[r.cell] = byCell[r.cell] || []).push(r);
            }
        }
    });
    return byCell;
}

// expose planCheap from the certified oracle fork in the live game context.
function exposePlanCheap(game) {
    const FORK = path.join(__dirname, '..', 'oracle_policy.js');
    let src = fs.readFileSync(FORK, 'utf8');
    if (src.indexOf('window.__cheap = {') < 0) throw new Error('anchor not found');
    src = src.replace('window.__cheap = {', 'window.__pcFork = planCheap;\n    window.__cheap = {');
    const win = game.win;
    const prev = { c: win.__cheap, r: win.__predatorReady, m: win.__predatorModel, pc: win.__pcFork };
    vm.runInThisContext(src, { filename: 'oracle_policy.js#expose' });
    const pc = win.__pcFork, ready = win.__predatorReady;
    win.__cheap = prev.c; win.__predatorReady = prev.r; win.__predatorModel = prev.m;
    return { pc, ready };
}

async function main() {
    const byCell = loadCommits();
    const sink = { cands: null, bi: -1 };
    const cap = {
        planStart(s, cands) { sink.cands = cands.map(c => [c.x, c.y]); },
        roll() {}, planEnd(pidx, score, bi) { sink.bi = bi; }, frameEnd() {},
    };
    const agg = {};   // cell -> stats
    const bySrc = {}; // src -> {ok,tot}
    let patrol = 0, total = 0, okSlot = 0, okNear = 0;

    for (const cell of Object.keys(byCell)) {
        const [W, H] = CELLS[cell]; if (!W) continue;
        let commits = byCell[cell];
        if (arg('heldOnlyNat')) commits = commits.filter(r => r.src === 2 && r.seed >= 120108);
        const g = await createGame({ policyDir: path.join(__dirname, '..', '..', '..', 'js'),
            W, H, seed: 7, startBoids: 5, fastRender: true });
        const { pc, ready } = exposePlanCheap(g);
        if (ready && ready.then) await ready;
        g.stepFrame();   // trigger configure() -> cfg.W/Hc = cell dims
        const st = agg[cell] = { ok: 0, tot: 0, patrol: 0 };
        let done = 0;
        for (const r of commits) {
            if (LIMIT && done >= LIMIT) break;
            const n = r.boids.length;
            const snap = { px: r.px, py: r.py, pvx: r.pvx, pvy: r.pvy, psize: r.psize,
                bx: r.boids.map(b => b.x), by: r.boids.map(b => b.y),
                bvx: r.boids.map(b => b.vx), bvy: r.boids.map(b => b.vy),
                lastFeed: -1e9, nowMs: 0 };
            sink.cands = null; sink.bi = -1;
            g.win.__oracle = cap;
            const target = pc(snap);
            g.win.__oracle = null;
            const cands = sink.cands, bi = sink.bi;
            // map committed candidate -> boid (exact candidates() order: cand0=E3D,
            // cand j(1..N)=j-1'th nearest boid lead-adjusted)
            const isPatrol = bi === 0 || coordKey(cands[bi][0], cands[bi][1]) === coordKey(cands[0][0], cands[0][1]);
            let committedBoid = -1;
            if (!isPatrol) {
                const order = r.boids.map((b, i) => [(b.x - r.px) ** 2 + (b.y - r.py) ** 2, i]).sort((a, b) => a[0] - b[0]);
                committedBoid = (bi >= 1 && bi <= n) ? order[bi - 1][1] : -1;
            }
            // nearest-current-boid-to-target cross-check
            let nb = -1, nd = Infinity;
            for (let i = 0; i < n; i++) { const d = (r.boids[i].x - target.x) ** 2 + (r.boids[i].y - target.y) ** 2; if (d < nd) { nd = d; nb = i; } }
            total++; st.tot++;
            if (isPatrol) { patrol++; st.patrol++; }
            const good = committedBoid === r.egIdx;
            if (good) { okSlot++; st.ok++; }
            if (nb === r.egIdx) okNear++;
            (bySrc[r.src] = bySrc[r.src] || { ok: 0, tot: 0 }); bySrc[r.src].tot++; if (good) bySrc[r.src].ok++;
            done++;
        }
        process.stderr.write(`[${cell}] S_dec ${(st.ok / st.tot).toFixed(4)} (n=${st.tot}) patrol ${(st.patrol / st.tot * 100).toFixed(1)}%\n`);
    }
    const srcName = { 0: 'scatter', 1: 'scatter2', 2: 'NATURAL' };
    console.log('=== UNIFIED planCheap-all-N vs prod intercept egBoid (N<=5) ===');
    console.log(`S_dec(N<=5) slot-mapped : ${(okSlot / total).toFixed(4)}  (n=${total})`);
    console.log(`S_dec(N<=5) nearest-boid: ${(okNear / total).toFixed(4)}  (cross-check)`);
    console.log(`E3D-patrol quirk rate   : ${(patrol / total * 100).toFixed(2)}%  (planCheap picks the patrol point, not a boid)`);
    console.log('by source:', Object.keys(bySrc).map(s => `${srcName[s]}:${(bySrc[s].ok / bySrc[s].tot).toFixed(4)}(${bySrc[s].tot})`).join('  '));
    console.log('by cell:', Object.keys(agg).map(c => `${c}:${(agg[c].ok / agg[c].tot).toFixed(3)}`).join(' '));
}
main().catch(e => { console.error(e); process.exit(1); });
