// gate.js — VALIDATION GATE. With full flocking ON, reproduce the logged
// r.rolled[rk] = [ci, catches, boot] for a sample of oracle records and verify:
//   catches == logged exactly, |boot - logged| < 1e-7.
// Reports match rate. Must be ~100% before trusting the cheap variants.
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const RV = require('./rollout_variants.js');

const DATA_DIR = path.join(__dirname, '..', 'data_1e6');

const _ab = new ArrayBuffer(8), _dv = new DataView(_ab);
function unhex(h) { _dv.setBigUint64(0, BigInt('0x' + h), true); return _dv.getFloat64(0, true); }

function* records(file, max) {
    const txt = zlib.gunzipSync(fs.readFileSync(file)).toString('utf8');
    let cnt = 0;
    for (const line of txt.split('\n')) {
        if (!line) continue;
        yield JSON.parse(line);
        if (max && ++cnt >= max) return;
    }
}

(async () => {
    const N_RECORDS = +(process.argv[2] || 400);
    await RV.init();
    const R = RV.makeRollout();

    const STRIDE = +(process.argv[3] || 1);     // span shards
    const PER_SHARD = +(process.argv[4] || 0);   // 0 = unlimited per shard
    let files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.decisions.jsonl.gz')).sort();
    if (STRIDE > 1) files = files.filter((_, i) => i % STRIDE === 0);
    let nChecked = 0, nCatchMatch = 0, nBootMatch = 0, nPairs = 0;
    let maxBootErr = 0, maxBootErrFinite = 0;
    let nNegInf = 0, nNegInfMatch = 0;
    const bootErrs = [];

    outer:
    for (const file of files) {
        for (const r of records(path.join(DATA_DIR, file), PER_SHARD || 0)) {
            // set arena size from record cfg
            R.cfg.W = r.cfg.W; R.cfg.Hc = r.cfg.Hc;
            const s = r.s;
            for (let rk = 0; rk < r.rolled.length; rk++) {
                const [ci, logCatch, logBoot] = r.rolled[rk];
                const cand = r.cands[ci];
                const out = R.rollScore(s, cand[0], cand[1], 'full');
                nPairs++;
                if (out.catches === logCatch) nCatchMatch++;
                // boot: logged may be -Infinity (extermination terminal => boot null->-inf via packer,
                // but in JSONL it's the actual number; JSON maps -Infinity to null).
                const lb = (logBoot === null) ? -Infinity : logBoot;
                if (lb === -Infinity || out.boot === -Infinity) {
                    nNegInf++;
                    if (lb === out.boot) { nBootMatch++; nNegInfMatch++; }
                } else {
                    const err = Math.abs(out.boot - lb);
                    bootErrs.push(err);
                    if (err > maxBootErr) maxBootErr = err;
                    if (err > maxBootErrFinite) maxBootErrFinite = err;
                    if (err < 1e-7) nBootMatch++;
                }
            }
            if (++nChecked >= N_RECORDS) break outer;
        }
    }

    bootErrs.sort((a, b) => a - b);
    const med = bootErrs.length ? bootErrs[bootErrs.length >> 1] : 0;
    const p99 = bootErrs.length ? bootErrs[Math.min(bootErrs.length - 1, Math.floor(bootErrs.length * 0.99))] : 0;
    console.log(JSON.stringify({
        nRecords: nChecked, nPairs,
        catchMatchRate: nCatchMatch / nPairs,
        bootMatchRate: nBootMatch / nPairs,
        catchMatches: nCatchMatch, bootMatches: nBootMatch,
        nNegInf, nNegInfMatch,
        maxBootErrFinite, medBootErr: med, p99BootErr: p99,
    }, null, 2));
})();
