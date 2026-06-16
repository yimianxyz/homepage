// tstar_stats.js — paired analysis of the T*(screen,N0) farm. Per cell: mean
// throughput per T (+SEM), the argmax T*, the statistically-tied T-plateau (paired
// bootstrap CI of T−best includes 0), and CIs vs the fixed baselines T=5 / T=8.
// Emits a clean table to stderr and a machine-readable SURFACE json to stdout/--out.
//   node tstar_stats.js farm.json [--out surface.json]
'use strict';
const fs = require('fs');

function mean(a) { return a.length ? a.reduce((s, x) => s + x, 0) / a.length : null; }
function sem(a) { if (a.length < 2) return 0; const m = mean(a);
    return Math.sqrt(a.reduce((s, x) => s + (x - m) * (x - m), 0) / (a.length - 1) / a.length); }
function bootstrapCI(diffs, B) {   // 95% CI on mean paired diff, deterministic LCG
    B = B || 4000; const n = diffs.length; if (!n) return { lo: 0, hi: 0, mean: 0 };
    let seed = 987654321; const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const means = new Array(B);
    for (let b = 0; b < B; b++) { let s = 0; for (let i = 0; i < n; i++) s += diffs[(rnd() * n) | 0]; means[b] = s / n; }
    means.sort((a, b) => a - b);
    return { lo: means[(B * 0.025) | 0], hi: means[(B * 0.975) | 0], mean: means[B >> 1] };
}
function wilcoxonP(diffs) {
    const nz = diffs.filter(d => d !== 0); const n = nz.length; if (n < 6) return null;
    const ranked = nz.map(d => ({ a: Math.abs(d), s: Math.sign(d) })).sort((x, y) => x.a - y.a);
    let i = 0; const ranks = new Array(n);
    while (i < n) { let j = i; while (j + 1 < n && ranked[j + 1].a === ranked[i].a) j++;
        const r = (i + j) / 2 + 1; for (let k = i; k <= j; k++) ranks[k] = r; i = j + 1; }
    let Wp = 0, Wm = 0; for (let k = 0; k < n; k++) { if (ranked[k].s > 0) Wp += ranks[k]; else Wm += ranks[k]; }
    const W = Math.min(Wp, Wm), muW = n * (n + 1) / 4, sigW = Math.sqrt(n * (n + 1) * (2 * n + 1) / 24);
    const z = (W - muW) / sigW; return +(2 * (1 - normcdf(Math.abs(z)))).toFixed(5);
}
function normcdf(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }

function main() {
    const fp = process.argv[2];
    const oi = process.argv.indexOf('--out'); const outFp = oi >= 0 ? process.argv[oi + 1] : null;
    const rep = JSON.parse(fs.readFileSync(fp, 'utf8'));
    const Ts = rep.Ts;
    console.error(`# T* surface — ${rep.seedSet}, rule=${rep.rule}, up to n=${rep.seeds} paired seeds, forkN=${rep.forkN}`);
    const surface = { seedSet: rep.seedSet, rule: rep.rule, Ts, cells: [] };
    for (const ckey of rep.cells) {
        const C = rep.R[ckey];
        // paired matrix: seed → {T: thru}
        const bySeed = {};
        for (const T of Ts) for (const row of C.byT[T]) (bySeed[row.seed] = bySeed[row.seed] || {})[T] = row.thru;
        const seeds = Object.keys(bySeed).filter(s => Ts.every(T => bySeed[s][T] != null));
        const perT = Ts.map(T => ({ T, mean: mean(seeds.map(s => bySeed[s][T])), sem: sem(seeds.map(s => bySeed[s][T])), n: seeds.length }));
        const best = perT.slice().sort((a, b) => b.mean - a.mean)[0];
        // plateau: T whose paired diff vs best has CI including 0 (tie with best)
        const plateau = [];
        for (const pt of perT) { const diffs = seeds.map(s => bySeed[s][pt.T] - bySeed[s][best.T]);
            const ci = bootstrapCI(diffs); if (ci.lo <= 0 && ci.hi >= 0) plateau.push(pt.T); }
        // clearance sanity at T=8 (or first T)
        const clT = C.byT[8] ? 8 : Ts[0];
        const clr = mean(C.byT[clT].map(r => r.cleared ? 1 : 0));
        const prefMean = mean((C.prefix || []).map(p => p.prefixFrames));
        const cappedN = (C.prefix || []).filter(p => p.capped).length;
        const area = (C.W + 20) * (C.H + 20);
        const cellOut = { key: ckey, W: C.W, H: C.H, uaMobile: C.uaMobile, N0: C.N0eff, forcedN0: C.forcedN0,
            area, sqrtArea: Math.sqrt(area), minDim: Math.min(C.W, C.H), maxDim: Math.max(C.W, C.H),
            density: C.N0eff / area, n: seeds.length, clearRate: clr, prefixFrames: prefMean, capped: cappedN,
            perT: perT.map(p => ({ T: p.T, mean: p.mean, sem: p.sem })),
            Tstar: best.T, plateau,
            vs5: cmp(seeds, bySeed, 5, Ts), vs8: cmp(seeds, bySeed, 8, Ts) };
        surface.cells.push(cellOut);
        // table line
        const bar = perT.map(p => `${p.T}:${(p.mean * 1e4).toFixed(2)}${p.T === best.T ? '*' : (plateau.includes(p.T) ? '~' : ' ')}`).join(' ');
        console.error(`\n## ${ckey}  N0=${C.N0eff} area=${(area / 1e6).toFixed(2)}M √A=${Math.sqrt(area).toFixed(0)} clr=${(clr * 100).toFixed(0)}% n=${seeds.length} pref=${prefMean ? prefMean.toFixed(0) : '?'}f${cappedN ? ' CAPPED=' + cappedN : ''}`);
        console.error(`   T*=${best.T} plateau=[${plateau.join(',')}]  thru(e-4): ${bar}`);
    }
    if (outFp) fs.writeFileSync(outFp, JSON.stringify(surface, null, 1));
    console.error(`\n# T* by cell: ` + surface.cells.map(c => `${c.key}→${c.Tstar}`).join('  '));
    if (outFp) console.error(`# surface → ${outFp}`);
}
// compare every T vs a fixed baseline bT (paired): returns {T:{d,lo,hi,p,verdict}}
function cmp(seeds, bySeed, bT, Ts) {
    const o = {};
    for (const T of Ts) { if (T === bT) continue;
        const diffs = seeds.map(s => bySeed[s][T] - bySeed[s][bT]);
        const ci = bootstrapCI(diffs), p = wilcoxonP(diffs);
        const tie = ci.lo <= 0 && ci.hi >= 0;
        o[T] = { d: ci.mean, lo: ci.lo, hi: ci.hi, p, verdict: tie ? 'tie' : (ci.mean > 0 ? 'beats' : 'worse') }; }
    return o;
}
if (require.main === module) main();
