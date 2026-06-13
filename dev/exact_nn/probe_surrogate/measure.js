// measure.js — the surrogate-rollout probe. For a representative sample of oracle
// plans, run the cheap rollout variants (V_const, V_avoid) AND the full-flock
// reproduction, and measure:
//   (1) catch agreement P(catches_cheap == catches_true) per rolled-candidate rank
//   (2) boot-error CDF |boot_cheap - boot_true| (fracBelow 1e-3/1e-2/5e-2/1e-1, median)
//   (3) surrogate-as-policy S_dec: score' = vprior with rolled overrides
//       (catches_cheap + boot_cheap); argmax (lowest-index max); committed coord
//       compared to lab.tx/ty (hex). Reported for const / avoid / full.
//   (4) wall-time per plan (cost proxy) for each variant vs full.
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const RV = require('./rollout_variants.js');

const DATA_DIR = path.join(__dirname, '..', 'data_1e6');

const _ab = new ArrayBuffer(8), _dv = new DataView(_ab);
function f64hex(x) { _dv.setFloat64(0, x, true); return _dv.getBigUint64(0, true).toString(16).padStart(16, '0'); }

function* records(file, max) {
    const txt = zlib.gunzipSync(fs.readFileSync(file)).toString('utf8');
    let cnt = 0;
    for (const line of txt.split('\n')) {
        if (!line) continue;
        yield JSON.parse(line);
        if (max && ++cnt >= max) return;
    }
}

// argmax over score (lowest index wins ties), return committed coord hex
function commitCoord(cands, score) {
    let bi = 0, bs = -Infinity;
    for (let k = 0; k < score.length; k++) if (score[k] > bs) { bs = score[k]; bi = k; }
    return { bi, hx: f64hex(cands[bi][0]), hy: f64hex(cands[bi][1]) };
}

(async () => {
    const N_RECORDS = +(process.argv[2] || 3200);
    const STRIDE = +(process.argv[3] || 5);
    const PER_SHARD = +(process.argv[4] || 0);
    await RV.init();
    const R = RV.makeRollout();

    const MODES = (process.argv[5] || 'full,avoid,const').split(',');
    // per-mode accumulators
    const acc = {};
    for (const m of MODES) acc[m] = {
        // catch agreement per rank (rk 0..3) and overall
        catchMatchByRk: [0, 0, 0, 0], catchTotalByRk: [0, 0, 0, 0],
        catchMatch: 0, catchTotal: 0,
        // boot errors (finite) and -inf bookkeeping
        bootErrs: [], nNegInfTrue: 0, nNegInfCheap: 0, nNegInfBothMatch: 0,
        // catch signed deltas histogram
        catchDelta: new Map(),
        // S_dec
        decMatch: 0, decTotal: 0,
        // wall time (ns)
        rollNs: 0, rollCount: 0,
    };

    let files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.decisions.jsonl.gz')).sort();
    if (STRIDE > 1) files = files.filter((_, i) => i % STRIDE === 0);

    let nRec = 0, nShards = 0;
    const shardsSeen = new Set();

    outer:
    for (const file of files) {
        for (const r of records(path.join(DATA_DIR, file), PER_SHARD || 0)) {
            R.cfg.W = r.cfg.W; R.cfg.Hc = r.cfg.Hc;
            const s = r.s, cands = r.cands;
            shardsSeen.add(file);

            // true rolled values keyed by ci
            const trueByCi = new Map();
            for (const [ci, c, b] of r.rolled) trueByCi.set(ci, { catches: c, boot: (b === null ? -Infinity : b) });

            // compute cheap (and full) rolled values per mode
            for (const m of MODES) {
                const A = acc[m];
                // build surrogate score' = vprior with rolled overrides
                const score = r.vprior.slice();
                for (let rk = 0; rk < r.rolled.length; rk++) {
                    const ci = r.rolled[rk][0];
                    const cand = cands[ci];
                    const t0 = process.hrtime.bigint();
                    const out = R.rollScore(s, cand[0], cand[1], m);
                    const t1 = process.hrtime.bigint();
                    A.rollNs += Number(t1 - t0); A.rollCount++;

                    const tr = trueByCi.get(ci);
                    // catch agreement
                    A.catchTotalByRk[rk]++; A.catchTotal++;
                    if (out.catches === tr.catches) { A.catchMatchByRk[rk]++; A.catchMatch++; }
                    const d = out.catches - tr.catches;
                    A.catchDelta.set(d, (A.catchDelta.get(d) || 0) + 1);
                    // boot error
                    if (tr.boot === -Infinity || out.boot === -Infinity) {
                        if (tr.boot === -Infinity) A.nNegInfTrue++;
                        if (out.boot === -Infinity) A.nNegInfCheap++;
                        if (tr.boot === out.boot) A.nNegInfBothMatch++;
                    } else {
                        A.bootErrs.push(Math.abs(out.boot - tr.boot));
                    }
                    // surrogate override
                    score[ci] = out.catches + out.boot;
                }
                // S_dec
                const cc = commitCoord(cands, score);
                A.decTotal++;
                if (cc.hx === r.lab.tx && cc.hy === r.lab.ty) A.decMatch++;
            }

            if (++nRec >= N_RECORDS) break outer;
        }
    }
    nShards = shardsSeen.size;

    function cdf(errs) {
        errs = errs.slice().sort((a, b) => a - b);
        const n = errs.length;
        const frac = t => errs.filter(e => e < t).length / Math.max(1, n);
        const q = p => n ? errs[Math.min(n - 1, Math.floor(n * p))] : 0;
        return {
            n, median: q(0.5), p90: q(0.9), p99: q(0.99), max: n ? errs[n - 1] : 0,
            fracBelow_0p001: frac(0.001), fracBelow_0p01: frac(0.01),
            fracBelow_0p05: frac(0.05), fracBelow_0p1: frac(0.1),
        };
    }

    const report = { nRecords: nRec, nShards, stride: STRIDE, perShard: PER_SHARD || 'all', perMode: {} };
    for (const m of MODES) {
        const A = acc[m];
        const deltaObj = {};
        for (const [k, v] of [...A.catchDelta.entries()].sort((a, b) => a[0] - b[0])) deltaObj[k] = v;
        report.perMode[m] = {
            catchAgreement: {
                overall: A.catchMatch / A.catchTotal,
                byRank: A.catchMatchByRk.map((mm, i) => A.catchTotalByRk[i] ? mm / A.catchTotalByRk[i] : null),
                signedDeltaHist: deltaObj,
            },
            bootErrorCDF: cdf(A.bootErrs),
            negInf: { true: A.nNegInfTrue, cheap: A.nNegInfCheap, bothMatch: A.nNegInfBothMatch },
            S_dec: A.decMatch / A.decTotal,
            usPerRollout: (A.rollNs / A.rollCount) / 1000,
            usPerPlan: (A.rollNs / A.decTotal) / 1000,
        };
    }
    console.log(JSON.stringify(report, null, 2));
    fs.writeFileSync(path.join(__dirname, 'measure_result.json'), JSON.stringify(report, null, 2));
})();
