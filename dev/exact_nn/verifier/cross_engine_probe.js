// cross_engine_probe.js — SPEC §4c "max cross-engine score perturbation".
//
// exp/pow are the ONLY engine-divergent ops (float64 spike: 1–2 ulp on
// 10–27% of calls across V8/JSC/SpiderMonkey; every other op is correctly-
// rounded IEEE and identical). The §4c τ-safety condition is
//   min trusted margin  ≫  max cross-engine score perturbation,
// so we bound how a worst-case exp/pow ulp divergence propagates through prod's
// value-net scoring (cp_value: gelu→erf→Math.exp) into the 16 committed scores
// and, ultimately, into a deduped-argmax FLIP (a different committed target).
//
// Self-contained, no second engine needed: it uses the spike's measured
// divergence as the worst-case input. Harvest real plan states (feat[16][19],
// ctx[4], the 16 final score[], pidx4, cand coords) from games via an anchored
// transform; recompute vprior with Math.exp wrapped to perturb every result by
// ±k ulp (k∈{1,2}); substitute the perturbed vprior into the non-rolled score
// slots (the 4 rolled scores come from the sqrt-only rollout — engine-invariant
// — so holding them is faithful and conservative for the flip rate); measure
// max|Δscore| and whether the deduped committed-target coord changes.
//
//   node cross_engine_probe.js --seeds 6 --cells 390x844,1512x982 --maxFrames 9000 --out ce.json
'use strict';
const fs = require('fs');
const path = require('path');
const { createGame } = require('../stepper.js');

const _dv = new DataView(new ArrayBuffer(8));
function addUlp(x, k) {                       // ±k ulp, toward/away from 0 by sign of k
    if (!Number.isFinite(x) || x === 0) return x;
    _dv.setFloat64(0, x);
    let bits = _dv.getBigUint64(0);
    const neg = (bits >> 63n) & 1n, step = BigInt(Math.abs(k));
    if ((k > 0) !== (neg === 1n)) bits += step; else bits -= step;
    _dv.setBigUint64(0, bits);
    return _dv.getFloat64(0);
}

const _ck = new DataView(new ArrayBuffer(16));
const ckey = (x, y) => { _ck.setFloat64(0, x); _ck.setFloat64(8, y); return _ck.getBigUint64(0) + ':' + _ck.getBigUint64(8); };

function dedupMargin(score, cx, cy) {         // top-2 margin over distinct coord classes
    const best = new Map();
    for (let k = 0; k < score.length; k++) {
        const key = ckey(cx[k], cy[k]);
        if (!best.has(key) || score[k] > best.get(key)) best.set(key, score[k]);
    }
    const s = Array.from(best.values()).sort((a, b) => b - a);
    return s.length >= 2 ? s[0] - s[1] : Infinity;
}
function pickKey(score, cx, cy) {             // committed target coord (strict-> argmax, prod's rule)
    let bi = 0, bs = -Infinity;
    for (let k = 0; k < score.length; k++) if (score[k] > bs) { bs = score[k]; bi = k; }
    return ckey(cx[bi], cy[bi]);
}

const ANCHOR = '        return { x: cands[bi].x, y: cands[bi].y };';
function captureTransform(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(ANCHOR) < 0) throw new Error('cross_engine_probe: anchor not found');
    const inj =
        '        if (window.__ceLog) window.__ceLog.push({ '
        + 'feat: fr.feat.map(function(r){return r.slice();}), ctx: fr.ctx.slice(), '
        + 'score: score.slice(), pidx4: pidx.slice(0,4), '
        + 'cx: cands.map(function(c){return c.x;}), cy: cands.map(function(c){return c.y;}), '
        + 'n: s.bx.length });\n';
    return code.replace(ANCHOR, inj + ANCHOR);
}

function parseArgs() {
    const a = { seeds: 6, seedStart: 270000, maxFrames: 9000,
        cells: '390x844,1024x768,1512x982,2560x1440', out: null, dumpHarvest: null };
    for (let i = 2; i < process.argv.length; i++) {
        const k = process.argv[i];
        if (k === '--seeds') a.seeds = +process.argv[++i];
        else if (k === '--seedStart') a.seedStart = +process.argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +process.argv[++i];
        else if (k === '--cells') a.cells = process.argv[++i];
        else if (k === '--out') a.out = process.argv[++i];
        else if (k === '--dumpHarvest') a.dumpHarvest = process.argv[++i];   // write {feat,ctx,score,pidx4,cx,cy}[] for real-engine eval
    }
    if (a.seedStart >= 290000) throw new Error('refusing to probe on sealed range (>=290000)');
    return a;
}

async function main() {
    const args = parseArgs();
    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    const cp = require(path.join(policyDir, 'cheap_planner.js'));
    const NET = JSON.parse(fs.readFileSync(path.join(policyDir, 'value_net.json'), 'utf8'));
    const cells = args.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });

    const realExp = Math.exp;
    const expModes = { '+1': x => addUlp(realExp(x), 1), '-1': x => addUlp(realExp(x), -1),
        '+2': x => addUlp(realExp(x), 2), '-2': x => addUlp(realExp(x), -2) };
    const vpriorUnder = (expFn, rec) => { Math.exp = expFn; try { return cp.cp_value(NET, rec.feat, rec.ctx); } finally { Math.exp = realExp; } };

    const dScores = []; let plans = 0, flips = 0; const flipMargins = []; const byCell = {};
    const harvest = [];
    for (const c of cells) {
        const ckk = c.W + 'x' + c.H; byCell[ckk] = { plans: 0, flips: 0 };
        for (let i = 0; i < args.seeds; i++) {
            const game = await createGame({ policyDir, W: c.W, H: c.H, seed: args.seedStart + i,
                fastRender: true, transform: captureTransform });
            game.win.__ceLog = [];
            while (game.boidCount() > 0 && game.frame() < args.maxFrames) game.stepFrame();
            for (const rec of game.win.__ceLog) {
                if (args.dumpHarvest) harvest.push({ feat: rec.feat, ctx: rec.ctx, score: rec.score, pidx4: rec.pidx4, cx: rec.cx, cy: rec.cy, cell: ckk });
                plans++; byCell[ckk].plans++;
                const base = vpriorUnder(realExp, rec);              // == prod vprior (identity wrapper)
                const baseKey = pickKey(rec.score, rec.cx, rec.cy);
                const baseMargin = dedupMargin(rec.score, rec.cx, rec.cy);
                const rolled = new Set(rec.pidx4);
                let worstD = 0, flipped = false;
                for (const mk of Object.keys(expModes)) {
                    const vp = vpriorUnder(expModes[mk], rec);
                    for (let k = 0; k < base.length; k++) worstD = Math.max(worstD, Math.abs(vp[k] - base[k]));
                    const sc = rec.score.slice();
                    for (let k = 0; k < sc.length; k++) if (!rolled.has(k)) sc[k] = vp[k];
                    if (pickKey(sc, rec.cx, rec.cy) !== baseKey) { flipped = true; flipMargins.push(baseMargin); }
                }
                dScores.push(worstD);
                if (flipped) { flips++; byCell[ckk].flips++; }
            }
            game.win.__ceLog = null;
        }
        process.stderr.write(`[${ckk}] plans=${plans} flips=${flips}\n`);
    }

    const pct = (arr, p) => { const s = arr.slice().sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(p / 100 * s.length))] : null; };
    const report = {
        spec: 'SPEC §4c — cross-engine exp/pow ±1,2 ulp score-perturbation bound on PROD scoring',
        note: 'exp/pow are the ONLY engine-divergent ops (spike). vprior recomputed under '
            + 'worst-case ulp wrappers on Math.exp (gelu/erf path); rolled scores held (rollout '
            + 'is sqrt-only, engine-invariant). A flip = prod\'s committed target COORD changes.',
        cells: Object.keys(byCell), plans,
        dVprior_maxAbs: { max: dScores.length ? Math.max(...dScores) : null, p99: pct(dScores, 99), p50: pct(dScores, 50) },
        argmax_flips: flips, flip_rate: plans ? +(flips / plans).toFixed(6) : null,
        flip_margins_max: flipMargins.length ? Math.max(...flipMargins) : null,
        byCell,
        interpretation: 'flip_rate = fraction of plans where a worst-case ±2-ulp exp/pow divergence '
            + 'flips PROD\'s own committed target across engines — the engine-portability tax on prod, '
            + 'and the floor under any T2 student\'s cross-engine S_dec. flip_margins_max ≪ τ is the '
            + '§4c safety condition; flip_rate≈0 means the NN scoring is engine-robust at the decision '
            + 'level despite per-score drift.',
    };
    console.log(JSON.stringify(report, null, 1));
    if (args.out) fs.writeFileSync(args.out, JSON.stringify(report, null, 1));
    if (args.dumpHarvest) { fs.writeFileSync(args.dumpHarvest, JSON.stringify(harvest)); process.stderr.write(`harvest -> ${args.dumpHarvest} (${harvest.length} plans)\n`); }
}

if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
module.exports = { addUlp, dedupMargin };
