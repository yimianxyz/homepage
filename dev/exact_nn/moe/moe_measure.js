// moe_measure.js — CANONICAL S_dec of the deployed JS MoE (moeForward.js) on the
// held-out (by-seed) val split of the packed data. The packed feature blocks were
// built by moe_features.js — the SAME featurizer the live deploy uses — so the
// decision depends only on (block + NN); packed-block S_dec == live-game S_dec.
// This is the canonical artifact measurement (the JS deploy is what side-b's
// sealed verdict drives). Reports pooled + per-regime + per-cell + natural-endgame.
//   node moe_measure.js <pack_dir> <moe_weights.json>
'use strict';
const fs = require('fs');
const { loadMoE } = require('./moeForward.js');
const { PLANNER_DIM, ENDGAME_DIM, NSLOT } = require('./moe_features.js');

const PACK = process.argv[2], WEIGHTS = process.argv[3];
const STRIDE = parseInt(process.argv[4] || '1', 10);   // subsample val for a fast JS cross-check
const moe = loadMoE(WEIGHTS);
const CELLS = ['iphone_390x844', 'ipad_820x1180', 'desk_1024x768', 'desk_1512x982',
    'desk_1680x1050', 'desk_2560x1440', '390x844', '820x1180', '1024x768',
    '1512x982', '1680x1050', '2560x1440'];

function f32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); }
function i32(p) { const b = fs.readFileSync(p); return new Int32Array(b.buffer, b.byteOffset, b.length / 4); }

const ZERO_E = Array.from({ length: NSLOT }, () => new Array(ENDGAME_DIM).fill(0));
const ZERO_P = Array.from({ length: NSLOT }, () => new Array(PLANNER_DIM).fill(0));
const ALL_T = new Array(NSLOT).fill(true);
const ALL_F = new Array(NSLOT).fill(false);

function argmaxValid(logit, valid) {
    let bi = -1, bs = -Infinity;
    for (let k = 0; k < NSLOT; k++) if (valid[k] && logit[k] > bs) { bs = logit[k]; bi = k; }
    return bi;
}

// ---- planner ----
function measurePlanner() {
    const pm = JSON.parse(fs.readFileSync(`${PACK}/planner_meta.json`));
    const P = pm.P;
    const slots = f32(`${PACK}/planner_slots.f32`);
    const cls = i32(`${PACK}/planner_cls.i32`);
    const bicls = i32(`${PACK}/planner_bicls.i32`);
    const gate = f32(`${PACK}/planner_gate.f32`);
    const GD = pm.gateDim;
    const cellAcc = {};
    let ok = 0, tot = 0, gsum = 0;
    let nearOk = 0, nearTot = 0;        // dmargin < 1e-3 near-tie subset
    for (let r = 0; r < P; r++) {
        if (pm.split[r] !== 1) continue;
        if (STRIDE>1 && (r % STRIDE)) continue;
        const pb = new Array(NSLOT);
        for (let k = 0; k < NSLOT; k++) { const o = (r * NSLOT + k) * PLANNER_DIM; pb[k] = Array.prototype.slice.call(slots, o, o + PLANNER_DIM); }
        const gf = Array.prototype.slice.call(gate, r * GD, r * GD + GD);
        const { logit, g } = moe.forward(pb, ZERO_E, gf, ALL_T, ALL_T, ALL_F);
        const am = argmaxValid(logit, ALL_T);
        const amCls = cls[r * NSLOT + am];
        const good = amCls === bicls[r];
        ok += good ? 1 : 0; tot++; gsum += g;
        const c = CELLS[pm.cell[r]] || ('cell' + pm.cell[r]);
        (cellAcc[c] = cellAcc[c] || [0, 0]); cellAcc[c][0] += good ? 1 : 0; cellAcc[c][1]++;
        if (pm.dmargin[r] < 1e-3) { nearOk += good ? 1 : 0; nearTot++; }
    }
    return { S: ok / tot, n: tot, gateMean: gsum / tot, cells: cellAcc, near: { ok: nearOk, tot: nearTot } };
}

// ---- endgame ----
function measureEndgame() {
    const em = JSON.parse(fs.readFileSync(`${PACK}/endgame_meta.json`));
    const E = em.E;
    const slots = f32(`${PACK}/endgame_slots.f32`);
    const valid = i32(`${PACK}/endgame_valid.i32`);
    const egidx = i32(`${PACK}/endgame_egidx.i32`);
    const gate = f32(`${PACK}/endgame_gate.f32`);
    const GD = em.gateDim;
    const cellAcc = {}; const srcAcc = {};
    let ok = 0, tot = 0, gsum = 0;
    for (let r = 0; r < E; r++) {
        if (em.split[r] !== 1) continue;
        if (STRIDE>1 && (r % STRIDE)) continue;
        const eb = new Array(NSLOT); const sv = new Array(NSLOT);
        for (let k = 0; k < NSLOT; k++) { const o = (r * NSLOT + k) * ENDGAME_DIM; eb[k] = Array.prototype.slice.call(slots, o, o + ENDGAME_DIM); sv[k] = valid[r * NSLOT + k] > 0; }
        const gf = Array.prototype.slice.call(gate, r * GD, r * GD + GD);
        const { logit, g } = moe.forward(ZERO_P, eb, gf, sv, ALL_F, sv);
        const am = argmaxValid(logit, sv);
        const good = am === egidx[r];
        ok += good ? 1 : 0; tot++; gsum += g;
        const c = CELLS[em.cell[r]] || ('cell' + em.cell[r]);
        (cellAcc[c] = cellAcc[c] || [0, 0]); cellAcc[c][0] += good ? 1 : 0; cellAcc[c][1]++;
        const sk = em.src[r];
        (srcAcc[sk] = srcAcc[sk] || [0, 0]); srcAcc[sk][0] += good ? 1 : 0; srcAcc[sk][1]++;
    }
    return { S: ok / tot, n: tot, gateMean: gsum / tot, cells: cellAcc, src: srcAcc };
}

const P = measurePlanner();
const E = measureEndgame();
const pooledOk = P.S * P.n + E.S * E.n, pooledN = P.n + E.n;
console.log('=== CANONICAL JS S_dec (held-out val) ===');
console.log(`PLANNER : S_dec ${P.S.toFixed(4)}  (n=${P.n}, gate~${P.gateMean.toFixed(3)})  near-tie(dmargin<1e-3) ${P.near.ok}/${P.near.tot}`);
console.log(`ENDGAME : S_dec ${E.S.toFixed(4)}  (n=${E.n}, gate~${E.gateMean.toFixed(3)})`);
console.log(`POOLED  : S_dec ${(pooledOk / pooledN).toFixed(4)}  (n=${pooledN})`);
const srcName = { 0: 'data_eg(scatter)', 1: 'data_eg2', 2: 'data_eg_nat(NATURAL)' };
console.log('endgame by source:', Object.keys(E.src).map(k => `${srcName[k] || k}:${(E.src[k][0] / E.src[k][1]).toFixed(4)}(${E.src[k][1]})`).join('  '));
console.log('planner per-cell:', Object.keys(P.cells).map(c => `${c}:${(P.cells[c][0] / P.cells[c][1]).toFixed(3)}`).join(' '));
console.log('endgame per-cell:', Object.keys(E.cells).map(c => `${c}:${(E.cells[c][0] / E.cells[c][1]).toFixed(3)}`).join(' '));
