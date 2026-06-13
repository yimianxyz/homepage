// analyze_residual.js — the residual-learning question, quantified.
// Loads pairs_<mode>.json (boot_true, boot_cheap, catches). Reports, per mode:
//   - Pearson r(boot_cheap, boot_true), Spearman rho
//   - residual CDF after the BEST global linear fit boot_true ~ a*boot_cheap + b
//     (this upper-bounds what a 1-feature residual NN could do from boot_cheap alone)
//   - same, but ALSO conditioning on catches_cheap (2-feature linear fit) — a
//     looser upper bound on a richer residual NN that sees the cheap rollout's
//     catches too.
//   - the baseline: residual of just using vprior-style constant (std of boot_true).
'use strict';
const fs = require('fs');
const path = require('path');

function pearson(x, y) {
    const n = x.length; let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; syy += y[i] * y[i]; sxy += x[i] * y[i]; }
    const cov = sxy / n - (sx / n) * (sy / n);
    const vx = sxx / n - (sx / n) ** 2, vy = syy / n - (sy / n) ** 2;
    return cov / Math.sqrt(vx * vy);
}
function rank(a) {
    const idx = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]);
    const r = new Array(a.length);
    for (let i = 0; i < idx.length;) {
        let j = i; while (j < idx.length && idx[j][0] === idx[i][0]) j++;
        const avg = (i + j - 1) / 2 + 1;
        for (let k = i; k < j; k++) r[idx[k][1]] = avg;
        i = j;
    }
    return r;
}
function spearman(x, y) { return pearson(rank(x), rank(y)); }

// least squares y ~ X (X rows are feature vectors incl. intercept). Normal eqns.
function lstsq(X, y) {
    const m = X[0].length, n = X.length;
    const ATA = Array.from({ length: m }, () => new Array(m).fill(0));
    const ATy = new Array(m).fill(0);
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < m; a++) { ATy[a] += X[i][a] * y[i]; for (let b = 0; b < m; b++) ATA[a][b] += X[i][a] * X[i][b]; }
    }
    // gaussian elimination
    const A = ATA.map((row, i) => row.concat([ATy[i]]));
    for (let c = 0; c < m; c++) {
        let piv = c; for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
        [A[c], A[piv]] = [A[piv], A[c]];
        const d = A[c][c] || 1e-12;
        for (let j = c; j <= m; j++) A[c][j] /= d;
        for (let r = 0; r < m; r++) if (r !== c) { const f = A[r][c]; for (let j = c; j <= m; j++) A[r][j] -= f * A[c][j]; }
    }
    return A.map(row => row[m]);
}
function residCDF(resids) {
    const a = resids.map(Math.abs).sort((p, q) => p - q); const n = a.length;
    const frac = t => a.filter(e => e < t).length / Math.max(1, n);
    const q = p => n ? a[Math.min(n - 1, Math.floor(n * p))] : 0;
    return { n, median: q(0.5), p90: q(0.9),
        fracBelow_0p001: frac(0.001), fracBelow_0p01: frac(0.01), fracBelow_0p05: frac(0.05), fracBelow_0p1: frac(0.1) };
}

const MODES = (process.argv[2] || 'avoid,const').split(',');
const out = {};
for (const m of MODES) {
    const p = JSON.parse(fs.readFileSync(path.join(__dirname, `pairs_${m}.json`), 'utf8'));
    // keep only pairs where both boots finite
    const bt = [], bc = [], cc = [];
    for (let i = 0; i < p.bt.length; i++) {
        if (p.bt[i] == null || p.bc[i] == null) continue;
        bt.push(p.bt[i]); bc.push(p.bc[i]); cc.push(p.cc[i]);
    }
    const n = bt.length;
    const r = pearson(bc, bt), rho = spearman(bc, bt);
    // baseline residual: predict boot_true with its own mean (no info)
    const mean = bt.reduce((a, b) => a + b, 0) / n;
    const baseResid = bt.map(v => v - mean);
    // raw error (no correction): boot_cheap as-is
    const rawErr = bt.map((v, i) => v - bc[i]);
    // 1-feature linear fit: boot_true ~ a*boot_cheap + b
    const beta1 = lstsq(bc.map(v => [1, v]), bt);
    const resid1 = bt.map((v, i) => v - (beta1[0] + beta1[1] * bc[i]));
    // 2-feature: + catches_cheap
    const beta2 = lstsq(bc.map((v, i) => [1, v, cc[i]]), bt);
    const resid2 = bt.map((v, i) => v - (beta2[0] + beta2[1] * bc[i] + beta2[2] * cc[i]));
    out[m] = {
        nFinitePairs: n, pearson_r: r, spearman_rho: rho, bootTrue_std: Math.sqrt(baseResid.reduce((a, b) => a + b * b, 0) / n),
        rawError: residCDF(rawErr),
        baselineMeanResid: residCDF(baseResid),
        linFit1_residual: { coef: beta1, cdf: residCDF(resid1) },
        linFit2_residual_withCheapCatches: { coef: beta2, cdf: residCDF(resid2) },
    };
}
console.log(JSON.stringify(out, null, 2));
fs.writeFileSync(path.join(__dirname, 'residual_result.json'), JSON.stringify(out, null, 2));
