// endgame_margin.js — D4 / L1e deliverable-zero: the endgame egBoid-COMMIT
// near-tie density (SPEC §3/§4 D4, the N≤5 analogue of the §6.3 plan-margin CDF;
// flagged unmeasured by side-a on #5).
//
// intercept() commits egBoid to the SOONEST-REACHABLE boid: argmin over boids of
// scan(boid).t (earliest frame the predator can stand where the boid will be);
// nearest-distance fallback if none reachable. An L1e student that PROPOSES the
// egBoid (with the exact scan as validator/fallback) mismatches prod only when
// two boids are reachable at (nearly) the same t and the student picks the other.
// So the commit-margin = t(2nd-soonest) − t(soonest) CDF bounds L1e's NN-alone
// share, exactly as the plan-margin CDF bounds L1's.
//
// Captures, at every commit (egBoid (re)assigned — first N≤5 entry and after each
// catch), the sorted reachable scan-t's of all boids, via an anchored transform
// on intercept() (logging only; digest-inert). Runs endgame-heavy games.
//
//   node endgame_margin.js --seeds 300 --seedStart 270000 --startBoids 5 \
//        --cells 390x844,1024x768,2560x1440 --maxFrames 6000 --out eg.json
'use strict';
const fs = require('fs');
const path = require('path');
const { createGame } = require('../stepper.js');

// anchor: the line right after intercept() finishes choosing egBoid (commit
// block). We inject AFTER the nearest-fallback close-brace, before the aim scan.
const ANCHOR = '        // aim at the earliest-reachable point (perpendicular cut-off onto its line if none)';
function egTransform(filename, code) {
    if (filename !== 'predator_cheap.js') return null;
    if (code.indexOf(ANCHOR) < 0) throw new Error('endgame_margin: intercept anchor not found');
    // `justCommitted` is true when this frame (re)picked egBoid: detect via a
    // module flag set inside the `if (!egBoid)` block. Simpler: recompute the
    // commit set here only when we know a commit happened. We log every frame
    // where the prior egBoid was null/stale — approximated by logging whenever
    // the reachable set has ≥1 entry AND egBoid was chosen this call. To keep it
    // exact + cheap, we log ONLY on frames where boids.indexOf(egBoid)===0-state
    // changed; but the closure can't see "changed" without a flag. So: set a flag.
    const inj =
        '        if (window.__egLog && window.__egJustCommitted) {\n'
        + '            var __ts = [];\n'
        + '            for (var __z = 0; __z < boids.length; __z++) { var __c = scan(boids[__z]); if (__c) __ts.push(__c.t); }\n'
        + '            __ts.sort(function (a, b) { return a - b; });\n'
        + '            window.__egLog.push({ n: boids.length, reachable: __ts.length, ts0: __ts[0], ts1: __ts.length > 1 ? __ts[1] : null });\n'
        + '            window.__egJustCommitted = false;\n'
        + '        }\n';
    // also set the flag inside the commit block (when !egBoid path runs)
    let out = code.replace(ANCHOR, inj + ANCHOR);
    // mark commit: the block starts with `if (!egBoid) {` — set the flag true there
    const COMMIT = '        if (!egBoid) {';
    if (out.indexOf(COMMIT) < 0) throw new Error('commit anchor not found');
    out = out.replace(COMMIT, '        if (!egBoid) { if (window.__egLog) window.__egJustCommitted = true;');
    return out;
}

function summarize(margins) {
    const s = margins.filter(m => m != null && Number.isFinite(m)).sort((a, b) => a - b);
    const n = s.length;
    const pct = p => n ? s[Math.min(n - 1, Math.floor(p / 100 * n))] : null;
    // margins here are integer-frame differences (scan t is an integer frame)
    const TAUS = [0, 1, 2, 3, 5, 10, 20, 50];
    const fracAt = {};
    for (const t of TAUS) fracAt[t] = n ? +(s.filter(m => m <= t).length / n).toFixed(4) : null;
    return { n, ties_margin0: margins.filter(m => m === 0).length,
        pctiles: { p1: pct(1), p5: pct(5), p10: pct(10), p25: pct(25), p50: pct(50), p75: pct(75), p90: pct(90) },
        fracAtOrBelow: fracAt };
}

async function main() {
    const args = { seeds: 300, seedStart: 270000, startBoids: 5, maxFrames: 6000,
        cells: '390x844,1024x768,1512x982,2560x1440', out: null };
    for (let i = 2; i < process.argv.length; i++) {
        const k = process.argv[i];
        if (k === '--seeds') args.seeds = +process.argv[++i];
        else if (k === '--seedStart') args.seedStart = +process.argv[++i];
        else if (k === '--startBoids') args.startBoids = +process.argv[++i];
        else if (k === '--maxFrames') args.maxFrames = +process.argv[++i];
        else if (k === '--cells') args.cells = process.argv[++i];
        else if (k === '--out') args.out = process.argv[++i];
    }
    if (args.seedStart >= 290000) throw new Error('refusing sealed range');
    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    const cells = args.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });

    const allMargins = [], byCell = {}; let commits = 0, unreachableCommits = 0, soleReachable = 0;
    for (const c of cells) {
        const key = c.W + 'x' + c.H; byCell[key] = [];
        for (let i = 0; i < args.seeds; i++) {
            const game = await createGame({ policyDir, W: c.W, H: c.H, seed: args.seedStart + i,
                startBoids: args.startBoids, scatter: true, fastRender: true, transform: egTransform });
            game.win.__egLog = []; game.win.__egJustCommitted = false;
            while (game.boidCount() > 0 && game.frame() < args.maxFrames) game.stepFrame();
            for (const rec of game.win.__egLog) {
                commits++;
                if (rec.reachable === 0) { unreachableCommits++; continue; }   // nearest-distance fallback path
                if (rec.ts1 == null) { soleReachable++; continue; }            // only 1 reachable -> no contest
                const m = rec.ts1 - rec.ts0;
                allMargins.push(m); byCell[key].push(m);
            }
            game.win.__egLog = null;
        }
        process.stderr.write(`[${key}] commits-with-margin=${byCell[key].length}\n`);
    }
    const report = {
        spec: 'SPEC §4 D4 / L1e — endgame egBoid-commit near-tie density (frames between 1st and 2nd soonest-reachable boid)',
        startBoids: args.startBoids, cells: cells.map(c => c.W + 'x' + c.H),
        commits_total: commits, commits_nearestFallback: unreachableCommits, commits_soleReachable: soleReachable,
        commits_contested: allMargins.length,
        pooled: summarize(allMargins),
        byCell: Object.fromEntries(Object.entries(byCell).map(([k, v]) => [k, summarize(v)])),
        interpretation: 'margin = t(2nd-soonest) − t(soonest) in FRAMES. margin==0 means two boids are '
            + 'reachable on the SAME frame → an L1e egBoid student must match prod\'s argmin index '
            + 'tiebreak (lowest index) to be exact; fracAtOrBelow[k] ≈ the fraction of commits an L1e '
            + 'student must route to the exact-scan fallback at a k-frame guard. Large soleReachable / '
            + 'nearestFallback fractions mean most commits are uncontested → high L1e NN-alone ceiling.',
    };
    console.log(JSON.stringify(report, null, 1));
    if (args.out) fs.writeFileSync(args.out, JSON.stringify(report, null, 1));
}

if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
