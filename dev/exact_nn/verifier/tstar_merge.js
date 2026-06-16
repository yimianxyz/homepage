// tstar_merge.js — merge (cell × seed-block) tstar shards into one report.
//   node tstar_merge.js <shardsDir> <TAG> <outFile>
'use strict';
const fs = require('fs'), path = require('path');
const [dir, TAG, out] = process.argv.slice(2);
const files = fs.readdirSync(dir).filter(f => f.startsWith(TAG + '_') && f.endsWith('.json'));
const merged = { metric: null, rule: null, Ts: null, cells: [], seeds: 0,
    seedSet: null, forkN: null, maxFrames: null, R: {} };
const cellOrder = [];
let totalSeeds = 0;
for (const f of files.sort()) {
    const r = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
    merged.metric = merged.metric || r.metric; merged.rule = merged.rule || r.rule;
    merged.Ts = merged.Ts || r.Ts; merged.seedSet = merged.seedSet || r.seedSet;
    merged.forkN = merged.forkN || r.forkN; merged.maxFrames = merged.maxFrames || r.maxFrames;
    for (const ckey in r.R) {
        const src = r.R[ckey];
        if (!merged.R[ckey]) { merged.R[ckey] = { W: src.W, H: src.H, uaMobile: src.uaMobile,
            N0eff: src.N0eff, forcedN0: src.forcedN0, byT: {}, prefix: [] }; cellOrder.push(ckey);
            for (const T of merged.Ts) merged.R[ckey].byT[T] = []; }
        for (const T of merged.Ts) if (src.byT[T]) merged.R[ckey].byT[T].push(...src.byT[T]);
        if (src.prefix) merged.R[ckey].prefix.push(...src.prefix);
    }
}
// dedup seeds within each cell/T (in case blocks overlap), keep unique by seed
for (const ckey in merged.R) for (const T of merged.Ts) {
    const seen = new Set(), uniq = [];
    for (const row of merged.R[ckey].byT[T]) { if (!seen.has(row.seed)) { seen.add(row.seed); uniq.push(row); } }
    merged.R[ckey].byT[T] = uniq.sort((a, b) => a.seed - b.seed);
}
merged.cells = cellOrder;
merged.seeds = Math.max(...cellOrder.map(c => merged.R[c].byT[merged.Ts[0]].length));
fs.writeFileSync(out, JSON.stringify(merged));
console.log(`merged ${files.length} shards → ${cellOrder.length} cells, up to ${merged.seeds} seeds each → ${out}`);
