// calib_gen.js — generate the L1h calibration record {margin, agree} on the
// published calibration range [270000,280000), INDEPENDENTLY (verifier owns
// "agree" — computed against side-b's own frozen prod, not trusting side-a's
// calibration computation). Feeds tau_calibrate.js.
//
// At every prod plan it logs the snapshot + candidates + prod's committed index
// (anchored transform on planCheap, logging-only/digest-inert). Offline it runs
// the delivered student scorer on the SAME state, computes the student's deduped
// top-2 margin + deduped-argmax committed coord, and agree = (student coord ==
// prod coord, deduped §3). Output = exactly tau_calibrate.js --in's schema.
//
//   node calib_gen.js --seeds 200 --seedStart 270000 --maxFrames 20000 \
//     --cells 390x844,820x1180,1024x768,1512x982,1680x1050,2560x1440 \
//     --student ../student/studentScores.js --weights ../student/student_weights.json \
//     --out ../student/calib_margins.json
'use strict';
const fs = require('fs');
const path = require('path');
const { createGame } = require('../stepper.js');

const ANCHOR = '        return { x: cands[bi].x, y: cands[bi].y };';
function capTransform(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(ANCHOR) < 0) throw new Error('calib_gen: anchor not found');
    const inj =
        '        if (window.__calibLog) window.__calibLog.push({ '
        + 'px:s.px,py:s.py,pvx:s.pvx,pvy:s.pvy,psize:s.psize,'
        + 'bx:s.bx.slice(),by:s.by.slice(),bvx:s.bvx.slice(),bvy:s.bvy.slice(),'
        + 'cx: cands.map(function(c){return c.x;}), cy: cands.map(function(c){return c.y;}), bi: bi });\n';
    return code.replace(ANCHOR, inj + ANCHOR);
}

const _ck = new DataView(new ArrayBuffer(16));
function ckey(x, y) { _ck.setFloat64(0, x); _ck.setFloat64(8, y); return _ck.getBigUint64(0) + ':' + _ck.getBigUint64(8); }
function dedup(score, cx, cy) {
    const best = new Map();
    for (let k = 0; k < score.length; k++) {
        const key = ckey(cx[k], cy[k]); const cur = best.get(key);
        if (!cur || score[k] > cur.s || (score[k] === cur.s && k < cur.idx)) best.set(key, { s: score[k], idx: k });
    }
    const arr = Array.from(best.values()).sort((a, b) => (b.s - a.s) || (a.idx - b.idx));
    return { argIdx: arr[0].idx, margin: arr.length >= 2 ? arr[0].s - arr[1].s : Infinity };
}

async function main() {
    const a = { seeds: 200, seedStart: 270000, maxFrames: 20000,
        cells: '390x844,820x1180,1024x768,1512x982,1680x1050,2560x1440',
        student: path.join(__dirname, '..', 'student', 'studentScores.js'),
        weights: path.join(__dirname, '..', 'student', 'student_weights.json'),
        out: path.join(__dirname, '..', 'student', 'calib_margins.json') };
    for (let i = 2; i < process.argv.length; i++) {
        const k = process.argv[i];
        if (k === '--seeds') a.seeds = +process.argv[++i];
        else if (k === '--seedStart') a.seedStart = +process.argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +process.argv[++i];
        else if (k === '--cells') a.cells = process.argv[++i];
        else if (k === '--student') a.student = process.argv[++i];
        else if (k === '--weights') a.weights = process.argv[++i];
        else if (k === '--out') a.out = process.argv[++i];
    }
    if (a.seedStart >= 290000 || a.seedStart + a.seeds > 290000) throw new Error('calibration must stay < 290000 (sealed pool)');
    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    const { loadStudent } = require(path.resolve(a.student));
    const studentScores = loadStudent(path.resolve(a.weights));
    const cells = a.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });

    const out = [];
    for (const c of cells) {
        const cell = c.W + 'x' + c.H;
        const cfg = { W: c.W, Hc: c.H, PREDATOR_RANGE: 80, NUM_BOIDS: c.W <= 768 ? 60 : 120 };
        for (let i = 0; i < a.seeds; i++) {
            const game = await createGame({ policyDir, W: c.W, H: c.H, seed: a.seedStart + i,
                fastRender: true, transform: capTransform });
            game.win.__calibLog = [];
            while (game.boidCount() > 0 && game.frame() < a.maxFrames) game.stepFrame();
            for (const r of game.win.__calibLog) {
                const snap = { px: r.px, py: r.py, pvx: r.pvx, pvy: r.pvy, psize: r.psize,
                    bx: r.bx, by: r.by, bvx: r.bvx, bvy: r.bvy, nAlive: r.bx.length };
                const cands = r.cx.map((x, k) => ({ x, y: r.cy[k] }));
                const sc = studentScores(snap, cands, cfg);
                const d = dedup(sc, r.cx, r.cy);
                const prodKey = ckey(r.cx[r.bi], r.cy[r.bi]);
                const studKey = ckey(r.cx[d.argIdx], r.cy[d.argIdx]);
                out.push({ margin: Number.isFinite(d.margin) ? d.margin : null,
                    agree: studKey === prodKey, n: snap.nAlive, cell });
            }
            game.win.__calibLog = null;
        }
        process.stderr.write(`[${cell}] cumulative plans=${out.length}\n`);
    }
    fs.writeFileSync(a.out, JSON.stringify(out));
    const agree = out.filter(o => o.agree).length;
    console.log(JSON.stringify({ plans: out.length, agree, S_dec: +(agree / out.length).toFixed(4),
        out: a.out, seedRange: [a.seedStart, a.seedStart + a.seeds] }));
}

if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
