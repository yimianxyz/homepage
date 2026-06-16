// throughput_stats.js — paired statistics on the throughput farm. PAIRED across seeds:
// Wilcoxon signed-rank (throughput diff vs baseline) + bootstrap CI on the mean paired
// diff; calls "tie" when the 95% CI includes 0. Decides per-screen optimum + best single
// robust rule + whether anything beats count:T=5.
//   node throughput_stats.js farm.json [--baseline count:T=5]
'use strict';
const fs = require('fs');

function wilcoxonSignedRank(diffs) {   // paired; returns {W, z, p, n} (normal approx, zero-diffs dropped)
    const nz = diffs.filter(d => d !== 0);
    const n = nz.length;
    if (n < 6) return { n, p: null, z: null, note: 'n<6, no test' };
    const ranked = nz.map(d => ({ a: Math.abs(d), s: Math.sign(d) })).sort((x, y) => x.a - y.a);
    // average ranks for ties
    let i = 0; const ranks = new Array(n);
    while (i < n) { let j = i; while (j + 1 < n && ranked[j + 1].a === ranked[i].a) j++;
        const r = (i + j) / 2 + 1; for (let k = i; k <= j; k++) ranks[k] = r; i = j + 1; }
    let Wp = 0, Wm = 0;
    for (let k = 0; k < n; k++) { if (ranked[k].s > 0) Wp += ranks[k]; else Wm += ranks[k]; }
    const W = Math.min(Wp, Wm);
    const muW = n * (n + 1) / 4, sigW = Math.sqrt(n * (n + 1) * (2 * n + 1) / 24);
    const z = (W - muW) / sigW;
    const p = 2 * (1 - normcdf(Math.abs(z)));
    return { n, Wp, Wm, z: +z.toFixed(3), p: +p.toFixed(5) };
}
function normcdf(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
function bootstrapCI(diffs, B) {   // 95% CI on the mean paired diff (seeded LCG for determinism)
    B = B || 5000; const n = diffs.length; let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const means = new Array(B);
    for (let b = 0; b < B; b++) { let s = 0; for (let i = 0; i < n; i++) s += diffs[(rnd() * n) | 0]; means[b] = s / n; }
    means.sort((a, b) => a - b);
    return { lo: means[Math.floor(B * 0.025)], hi: means[Math.floor(B * 0.975)], mean: means[B >> 1] };
}

function main() {
    const fp = process.argv[2];
    const bi = process.argv.indexOf('--baseline');
    const baseline = bi >= 0 ? process.argv[bi + 1] : 'count:T=5';
    const rep = JSON.parse(fs.readFileSync(fp, 'utf8'));
    const configs = rep.configs.map(l => (typeof l === 'string' ? { label: l } : l)), cells = rep.cells;
    console.log('# Throughput paired-stats — ' + rep.seedSet + ', baseline ' + baseline + ', n=' + rep.seeds);
    const perScreenBest = {}; const beatsBaseline = {};
    for (const cell of cells) {
        const R = rep.R[cell];
        // align by seed (paired): map seed->thru per config
        const bySeed = {};
        for (const cfg of configs) for (const row of R[cfg.label]) { (bySeed[row.seed] = bySeed[row.seed] || {})[cfg.label] = row.thru; }
        const seeds = Object.keys(bySeed).filter(s => configs.every(c => bySeed[s][c.label] != null));
        const meanThru = {}; for (const cfg of configs) meanThru[cfg.label] = mean(seeds.map(s => bySeed[s][cfg.label]));
        // best config this screen (max mean throughput)
        const best = configs.map(c => c.label).sort((a, b) => meanThru[b] - meanThru[a])[0];
        perScreenBest[cell] = { best, meanThru };
        console.log('\n## ' + cell + '  (paired n=' + seeds.length + ')  best=' + best);
        for (const cfg of configs) {
            if (cfg.label === baseline) { console.log('  ' + pad(cfg.label) + ' thru=' + sci(meanThru[cfg.label]) + '  [baseline]'); continue; }
            const diffs = seeds.map(s => bySeed[s][cfg.label] - bySeed[s][baseline]);   // config − baseline (paired)
            const w = wilcoxonSignedRank(diffs), ci = bootstrapCI(diffs);
            const tie = ci.lo <= 0 && ci.hi >= 0;
            const verdict = tie ? 'TIE' : (ci.mean > 0 ? 'BEATS baseline' : 'WORSE');
            console.log('  ' + pad(cfg.label) + ' thru=' + sci(meanThru[cfg.label]) +
                '  Δvs' + baseline + '=' + sci(ci.mean) + ' CI[' + sci(ci.lo) + ',' + sci(ci.hi) + ']  W-p=' + w.p + '  → ' + verdict);
            (beatsBaseline[cfg.label] = beatsBaseline[cfg.label] || {})[cell] = { verdict, dmean: ci.mean, p: w.p, tie };
        }
    }
    // best single robust rule: max worst-screen mean throughput
    console.log('\n# Best single robust config (max worst-screen throughput):');
    const worst = {};
    for (const cfg of configs) worst[cfg.label] = Math.min(...cells.map(c => perScreenBest[c].meanThru[cfg.label]));
    const robust = configs.map(c => c.label).sort((a, b) => worst[b] - worst[a]);
    for (const lab of robust) console.log('  ' + pad(lab) + ' worst-screen thru=' + sci(worst[lab]));
    console.log('\n# Per-screen optimum: ' + cells.map(c => c + '→' + perScreenBest[c].best).join(', '));
    console.log('# ROBUST WINNER: ' + robust[0] + (robust[0] === baseline ? ' (= baseline T=5)' : ' (beats baseline on worst-screen)'));
}
function mean(a) { return a.reduce((s, x) => s + x, 0) / a.length; }
function sci(x) { return x == null ? 'n/a' : (x * 1e4).toFixed(4) + 'e-4'; }
function pad(s) { return (s + '                    ').slice(0, 18); }
if (require.main === module) main();
