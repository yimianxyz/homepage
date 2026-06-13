// boot_decompose.js — independent verifier cross-check of side-a's #5 finding
// "the decisive variable is BOOT, not catch-count" (which redirects v2's
// objective, so it's worth confirming on side-b's own instrument).
//
// Captures, per prod plan, the 4 rolled candidates' (catches, boot) + the full
// 16 scores + cands + committed bi (anchored transform on planCheap's rollout
// loop, logging-only/digest-inert). Offline it classifies each plan by what
// sets the deduped top-2 margin, reproducing side-a's 4-way table:
//   - both top-2 ROLLED, same catch-count  -> margin = boot difference
//   - top is a non-rolled vprior candidate
//   - both top-2 rolled, different catch-count
//   - rolled winner vs vprior runner-up
// + the all-rolled-zero-catch fraction and the boot stats.
//
//   node boot_decompose.js --seeds 8 --seedStart 270000 --cells 1024x768,2560x1440 --maxFrames 12000
'use strict';
const fs = require('fs');
const path = require('path');
const { createGame } = require('../stepper.js');

const A_LOOP = '            score[ci] = rr.catches + boot;';
const A_RET = '        return { x: cands[bi].x, y: cands[bi].y };';
function tf(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(A_LOOP) < 0 || code.indexOf(A_RET) < 0) throw new Error('boot anchors not found');
    let out = code.replace(A_LOOP, A_LOOP + '\n            if (window.__bl) window.__bl.cur.push({ci:ci, catches:rr.catches, boot:boot});');
    out = out.replace(A_RET,
        '        if (window.__bl) { window.__bl.plans.push({ rolled: window.__bl.cur, '
        + 'cx: cands.map(function(c){return c.x;}), cy: cands.map(function(c){return c.y;}), '
        + 'score: score.slice(), bi: bi, n: s.bx.length }); window.__bl.cur = []; }\n' + A_RET);
    // reset cur at plan entry: pidx is built just before the rollout loop; reset there
    const A_RESET = '        var score = vprior.slice();';
    out = out.replace(A_RESET, A_RESET + '\n        if (window.__bl) window.__bl.cur = [];');
    return out;
}

const _ck = new DataView(new ArrayBuffer(16));
function ckey(x, y) { _ck.setFloat64(0, x); _ck.setFloat64(8, y); return _ck.getBigUint64(0) + ':' + _ck.getBigUint64(8); }
// deduped top-2: return [{key,score,idx}, ...] sorted desc
function dedupTop(score, cx, cy) {
    const best = new Map();
    for (let k = 0; k < score.length; k++) { const key = ckey(cx[k], cy[k]); const c = best.get(key);
        if (!c || score[k] > c.s || (score[k] === c.s && k < c.idx)) best.set(key, { key, s: score[k], idx: k }); }
    return Array.from(best.values()).sort((a, b) => (b.s - a.s) || (a.idx - b.idx));
}

async function main() {
    const a = { seeds: 8, seedStart: 270000, maxFrames: 12000, cells: '1024x768,2560x1440' };
    for (let i = 2; i < process.argv.length; i++) { const k = process.argv[i];
        if (k === '--seeds') a.seeds = +process.argv[++i]; else if (k === '--seedStart') a.seedStart = +process.argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +process.argv[++i]; else if (k === '--cells') a.cells = process.argv[++i]; }
    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    const cells = a.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });

    let nPlans = 0, zeroCatchAll = 0;
    const cat = { boot_sameRolled: 0, vprior_top: 0, diffCatch_rolled: 0, rolledVsVprior: 0, other: 0 };
    const bootMargins = []; const allBoots = [];
    for (const c of cells) {
        for (let i = 0; i < a.seeds; i++) {
            const game = await createGame({ policyDir, W: c.W, H: c.H, seed: a.seedStart + i, fastRender: true, transform: tf });
            game.win.__bl = { plans: [], cur: [] };
            while (game.boidCount() > 0 && game.frame() < a.maxFrames) game.stepFrame();
            for (const p of game.win.__bl.plans) {
                nPlans++;
                const rolledIdx = new Set(p.rolled.map(r => r.ci));
                const catchOf = {}; p.rolled.forEach(r => { catchOf[r.ci] = r.catches; allBoots.push(r.boot); });
                if (p.rolled.length && p.rolled.every(r => r.catches === 0)) zeroCatchAll++;
                const top = dedupTop(p.score, p.cx, p.cy);
                if (top.length < 2) { cat.other++; continue; }
                const t1 = top[0], t2 = top[1];
                const t1r = rolledIdx.has(t1.idx), t2r = rolledIdx.has(t2.idx);
                if (!t1r) cat.vprior_top++;
                else if (t1r && t2r) {
                    if (catchOf[t1.idx] === catchOf[t2.idx]) { cat.boot_sameRolled++; bootMargins.push(t1.s - t2.s); }
                    else cat.diffCatch_rolled++;
                } else { cat.rolledVsVprior++; }
            }
            game.win.__bl = null;
        }
        process.stderr.write(`[${c.W}x${c.H}] plans=${nPlans}\n`);
    }
    const pct = x => +(100 * x / nPlans).toFixed(1);
    bootMargins.sort((x, y) => x - y); allBoots.sort((x, y) => x - y);
    const med = arr => arr.length ? arr[Math.floor(arr.length / 2)] : null;
    const out = {
        spec: 'side-b independent cross-check of side-a #5 "boot is the decisive variable"',
        plans: nPlans,
        zeroCatchAll_pct: pct(zeroCatchAll),
        decisiveCategory_pct: { boot_sameRolled: pct(cat.boot_sameRolled), vprior_top: pct(cat.vprior_top),
            diffCatch_rolled: pct(cat.diffCatch_rolled), rolledVsVprior: pct(cat.rolledVsVprior), other: pct(cat.other) },
        bootMargin_sameRolled: { median: med(bootMargins), n: bootMargins.length,
            below_0p01: +(100 * bootMargins.filter(m => m < 0.01).length / Math.max(1, bootMargins.length)).toFixed(1),
            below_0p001: +(100 * bootMargins.filter(m => m < 0.001).length / Math.max(1, bootMargins.length)).toFixed(1) },
        boot_stats: { min: allBoots[0], max: allBoots[allBoots.length - 1], median: med(allBoots),
            mean: +(allBoots.reduce((s, x) => s + x, 0) / allBoots.length).toFixed(4) },
        sideA_claim: { zeroCatchAll: '59%', boot_sameRolled: '69.3%', vprior_top: '12.5%', diffCatch_rolled: '11.8%', rolledVsVprior: '6.4%', bootMargin_median: 0.019, boot_mean: 0.78 },
    };
    console.log(JSON.stringify(out, null, 1));
}
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
