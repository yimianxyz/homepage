'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const IP = require('./intermediate_probe.js');
const DATA_DIR = path.join(__dirname, '..', 'data_1e6');

function* records(file, max) {
    const txt = zlib.gunzipSync(fs.readFileSync(file)).toString('utf8');
    let cnt = 0;
    for (const line of txt.split('\n')) { if (!line) continue; yield JSON.parse(line); if (max && ++cnt >= max) return; }
}
function pearson(xs, ys) {
    const n = xs.length; let mx = 0, my = 0; for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; } mx /= n; my /= n;
    let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n; i++) { const a = xs[i] - mx, b = ys[i] - my; sxy += a * b; sxx += a * a; syy += b * b; }
    return { r: sxy / Math.sqrt(sxx * syy), a: sxy / sxx, b: my - (sxy / sxx) * mx, sxx, syy, n };
}
function quant(arr, p) { const a = arr.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(a.length * p))]; }

(async () => {
    await IP.init();
    // mode registry
    const full = { kind: 'full' };
    const F = (coh, sep, align, pa) => ({ incCoh: coh, incSep: sep, incAlign: align, incPredAvoid: pa });
    const MODES = {
        full,
        short20: { kind: 'short', K: 20, fl: F(1, 1, 1, 1) },
        short30: { kind: 'short', K: 30, fl: F(1, 1, 1, 1) },
        short45: { kind: 'short', K: 45, fl: F(1, 1, 1, 1) },
        short60: { kind: 'short', K: 60, fl: F(1, 1, 1, 1) },
        cohsep: { kind: 'flags', fl: F(1, 1, 0, 1) },          // drop alignment
        nopredavoid: { kind: 'flags', fl: F(1, 1, 1, 0) },     // drop predator-avoid on boids
        cohsep_noavoid: { kind: 'flags', fl: F(1, 1, 0, 0) },
        coarsegrid: { kind: 'flags', fl: F(1, 1, 1, 1), gridMult: 1.5 },
    };
    const Rfull = IP.makeRollout({});
    const Rcoarse = IP.makeRollout({ gridMult: 1.5 });
    const N_REC = +(process.argv[2] || 1500);
    const STRIDE = +(process.argv[3] || 23);

    let files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.decisions.jsonl.gz')).sort();
    if (STRIDE > 1) files = files.filter((_, i) => i % STRIDE === 0);

    // gate: full-flock reproduction
    let gateOk = 0, gateN = 0, gateMaxErr = 0;

    // per-mode: store (boot_cheap, boot_true) finite pairs + catch agreement
    const data = {}; for (const m of Object.keys(MODES)) data[m] = { bc: [], bt: [], catchOk: 0, catchN: 0 };

    let nrec = 0;
    outer:
    for (const file of files) {
        for (const r of records(path.join(DATA_DIR, file), 0)) {
            const s = r.s;
            for (const [ci, c, b] of r.rolled) {
                const cand = r.cands[ci];
                const btrue = (b === null ? -Infinity : b);
                for (const m of Object.keys(MODES)) {
                    const R = (m === 'coarsegrid') ? Rcoarse : Rfull;
                    R.cfg.W = r.cfg.W; R.cfg.Hc = r.cfg.Hc;
                    const out = R.rollScore(s, cand[0], cand[1], MODES[m]);
                    if (m === 'full') { gateN++; if (out.catches === c && Math.abs(out.boot - btrue) < 1e-9) gateOk++; gateMaxErr = Math.max(gateMaxErr, Math.abs(out.boot - btrue)); }
                    data[m].catchN++; if (out.catches === c) data[m].catchOk++;
                    if (isFinite(btrue) && isFinite(out.boot)) { data[m].bc.push(out.boot); data[m].bt.push(btrue); }
                }
            }
            if (++nrec >= N_REC) break outer;
        }
    }

    const report = { nrec, gate: { ok: gateOk, n: gateN, maxErr: gateMaxErr }, modes: {} };
    for (const m of Object.keys(MODES)) {
        const d = data[m];
        const errs = d.bc.map((v, i) => Math.abs(v - d.bt[i]));
        const pr = pearson(d.bc, d.bt);
        // residual after best linear fit boot_true ~ a*boot_cheap + b
        const resid = d.bt.map((t, i) => t - (pr.a * d.bc[i] + pr.b));
        const residAbs = resid.map(Math.abs);
        const residMean = resid.reduce((a, b) => a + b, 0) / resid.length;
        const residStd = Math.sqrt(resid.reduce((a, b) => a + (b - residMean) ** 2, 0) / resid.length);
        report.modes[m] = {
            n: d.bc.length,
            catchAgree: d.catchOk / d.catchN,
            bootErr_median: quant(errs, 0.5), bootErr_p90: quant(errs, 0.9),
            bootErr_fracBelow_0p01: errs.filter(e => e < 0.01).length / errs.length,
            bootErr_fracBelow_0p05: errs.filter(e => e < 0.05).length / errs.length,
            pearson_r: pr.r, r2: pr.r * pr.r,
            resid_std_afterLinFit: residStd,
            resid_fracBelow_0p01: residAbs.filter(e => e < 0.01).length / residAbs.length,
        };
    }
    console.log(JSON.stringify(report, null, 2));
    fs.writeFileSync(path.join(__dirname, 'intermediate_result.json'), JSON.stringify(report, null, 2));
})();
