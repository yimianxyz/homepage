// margin_merge.js — merge sharded margin_cdf --raw dumps into the final
// stratified CDF report (pooled + N-bucket + per-device), so the §6.3
// margin CDF can be produced in parallel (one process per device cell).
//   node margin_merge.js out.json shard1.json shard2.json ...
'use strict';
const fs = require('fs');
const { dedupMargin } = require('./margin_cdf.js');   // (unused; kept for cohesion)

function summarize(margins) {
    const s = margins.filter(m => m != null && Number.isFinite(m)).sort((a, b) => a - b);
    const n = s.length;
    const pct = p => n ? s[Math.min(n - 1, Math.floor(p / 100 * n))] : null;
    const TAUS = [0, 1e-9, 1e-6, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0];
    const fracBelow = {};
    for (const t of TAUS) fracBelow[t] = n ? +(s.filter(m => m < t).length / n).toFixed(6) : null;
    return {
        nPlansFinite: n, nPlansTotal: margins.length,
        exactTies_margin0: margins.filter(m => m === 0).length,
        oneClassPlans: margins.filter(m => m == null).length,
        pctiles: { p1: pct(1), p5: pct(5), p10: pct(10), p25: pct(25), p50: pct(50), p75: pct(75), p90: pct(90), p99: pct(99) },
        fracBelowTau: fracBelow,
    };
}

const [, , out, ...shards] = process.argv;
const recs = [];
for (const f of shards) recs.push(...JSON.parse(fs.readFileSync(f, 'utf8')));
const all = recs.map(r => r.m);
const byBucket = { 'N6-14': recs.filter(r => r.n <= 14).map(r => r.m), 'N15+': recs.filter(r => r.n >= 15).map(r => r.m) };
const cells = {};
for (const r of recs) (cells[r.cell] = cells[r.cell] || []).push(r.m);
const report = {
    spec: 'SPEC §6.3 margin CDF (deduped top1-top2 plan-score margin) — sharded',
    seedSet: 'calibration[270000,..)', totalPlans: recs.length,
    pooled: summarize(all),
    byBucket: { 'N6-14': summarize(byBucket['N6-14']), 'N15+': summarize(byBucket['N15+']) },
    byCell: Object.fromEntries(Object.entries(cells).map(([k, v]) => [k, summarize(v)])),
    interpretation: 'fracBelowTau[τ] ≈ L1h fallback fraction at threshold τ; (1−it) bounds NN-alone share.',
};
fs.writeFileSync(out, JSON.stringify(report, null, 1));
console.log(JSON.stringify({ totalPlans: report.totalPlans, pooled_pctiles: report.pooled.pctiles,
    pooled_fracBelow: report.pooled.fracBelowTau, exactTies: report.pooled.exactTies_margin0 }, null, 1));
