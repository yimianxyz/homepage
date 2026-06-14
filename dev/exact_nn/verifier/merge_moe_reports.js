// merge_moe_reports.js — recombine per-cell verdict_moe shard reports into one
// EXACT pooled Phase-2 verdict. Sharding by cell lets the (slow) natural sealed
// sweep run cell-parallel; this merges them losslessly from the per-cell `_raw`
// summable counters (S_dec is recomputed from summed counts — exact, not averaged).
//
//   node merge_moe_reports.js out.json shard1.json shard2.json ...
'use strict';
const fs = require('fs');
const GATE = 0.95;
function rnd(x) { return x == null ? null : +x.toFixed(9); }
function sdec(dis, tot) { return tot ? 1 - dis / tot : null; }
function median(arr) { if (!arr.length) return null; const s = arr.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; }
function addFS(d, s) { d.cos += s.cos; d.rel += s.rel; d.n += s.n; }
function mean(a) { return a.n ? a.cos / a.n : null; }
function meanRel(a) { return a.n ? a.rel / a.n : null; }

function main() {
    const outFp = process.argv[2];
    const inFps = process.argv.slice(3);
    if (!outFp || !inFps.length) { console.error('usage: merge_moe_reports.js out.json shard1.json ...'); process.exit(2); }

    const reports = inFps.map(f => JSON.parse(fs.readFileSync(f, 'utf8')));
    // collect every per-cell entry across all shard reports (dedup by cell — last wins)
    const byCell = new Map();
    let meta = null;
    for (const r of reports) {
        meta = meta || r;
        for (const c of (r.perCell || [])) {
            if (!c._raw) throw new Error('shard ' + c.cell + ' lacks _raw (re-run verdict_moe with the _raw cellReport)');
            byCell.set(c.cell, c);   // last occurrence wins (idempotent re-runs)
        }
    }
    const cells = [...byCell.values()];

    const P = { plans: 0, planDisagree: 0, egCommits: 0, egDisagree: 0, games: 0,
        lockstepCleared: 0, forkCleared: 0, trajIdentical: 0, gateFlips: 0, gateMalformed: 0,
        firstDiv: [], fsLockP: { cos: 0, rel: 0, n: 0 }, fsLockE: { cos: 0, rel: 0, n: 0 },
        fsForkP: { cos: 0, rel: 0, n: 0 }, fsForkE: { cos: 0, rel: 0, n: 0 } };
    for (const c of cells) {
        const r = c._raw;
        for (const k of ['plans', 'planDisagree', 'egCommits', 'egDisagree', 'games', 'lockstepCleared', 'forkCleared', 'trajIdentical', 'gateFlips', 'gateMalformed']) P[k] += r[k];
        P.firstDiv.push(...(r.firstDiv || []));
        for (const k of ['fsLockP', 'fsLockE', 'fsForkP', 'fsForkE']) addFS(P[k], r[k]);
    }
    const S_dec_pooled = sdec(P.planDisagree + P.egDisagree, P.plans + P.egCommits);
    const S_dec_planner = sdec(P.planDisagree, P.plans);
    const S_dec_endgame = sdec(P.egDisagree, P.egCommits);
    const gate = { threshold: GATE,
        pooled_pass: S_dec_pooled != null && S_dec_pooled >= GATE,
        planner_pass: S_dec_planner == null || S_dec_planner >= GATE,
        endgame_pass: S_dec_endgame == null || S_dec_endgame >= GATE };
    gate.PASS = gate.pooled_pass && gate.planner_pass && gate.endgame_pass;

    const merged = {
        phase: 2, merged_from: inFps.length + ' shard reports', mode: meta.mode,
        weights: meta.weights, weights_sha256: meta.weights_sha256,
        seedSet: meta.seedSet, seedCount: meta.seedCount, sealOffset: meta.sealOffset,
        commitment_sha256_salt: meta.commitment_sha256_salt, distribution: meta.distribution,
        cells: cells.length, devicematrix: cells.map(c => c.cell), games: P.games,
        S_dec_pooled: rnd(S_dec_pooled), S_dec_planner: rnd(S_dec_planner), S_dec_endgame: rnd(S_dec_endgame),
        plans_total: P.plans, planDisagree_total: P.planDisagree,
        egCommits_total: P.egCommits, egDisagree_total: P.egDisagree,
        decisions_total: P.plans + P.egCommits, disagree_total: P.planDisagree + P.egDisagree,
        gateMalformed_total: P.gateMalformed, gateFlips_total: P.gateFlips,
        S_force_lockstep: { planner_cos: rnd(mean(P.fsLockP)), planner_rel: rnd(meanRel(P.fsLockP)),
            endgame_cos: rnd(mean(P.fsLockE)), endgame_rel: rnd(meanRel(P.fsLockE)) },
        S_force_fork: { planner_cos: rnd(mean(P.fsForkP)), planner_rel: rnd(meanRel(P.fsForkP)),
            endgame_cos: rnd(mean(P.fsForkE)), endgame_rel: rnd(meanRel(P.fsForkE)) },
        S_traj_identical_frac: rnd(P.games ? P.trajIdentical / P.games : null),
        S_traj_median_first_div: median(P.firstDiv),
        forkClearedFrac: rnd(P.games ? P.forkCleared / P.games : null),
        lockstepClearedFrac: rnd(P.games ? P.lockstepCleared / P.games : null),
        gate, perCell: cells,
    };
    fs.writeFileSync(outFp, JSON.stringify(merged, null, 1));
    const pct = x => x == null ? 'n/a' : (x * 100).toFixed(4) + '%';
    console.error('=== MERGED S_dec VERDICT (' + merged.mode + ', ' + merged.distribution + ', ' + merged.seedSet + ') ===');
    console.error('  cells merged: ' + cells.length + '  games: ' + P.games + '  decisions: ' + (P.plans + P.egCommits));
    console.error('  POOLED  S_dec = ' + pct(S_dec_pooled) + '  (' + (gate.pooled_pass ? 'PASS' : 'FAIL') + ')');
    console.error('  PLANNER S_dec = ' + pct(S_dec_planner) + '  (' + (gate.planner_pass ? 'PASS' : 'FAIL') + ')');
    console.error('  ENDGAME S_dec = ' + pct(S_dec_endgame) + '  (' + (gate.endgame_pass ? 'PASS' : 'FAIL') + ')');
    console.error('  malformed: ' + P.gateMalformed + '   GATE (≥95% pooled+per-regime): ' + (gate.PASS ? '✅ PASS' : '❌ FAIL'));
    process.exit(gate.PASS ? 0 : 2);
}
main();
