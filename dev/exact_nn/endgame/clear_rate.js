// clear_rate.js — OUTCOME metric (#5): full-game CLEAR-RATE of the unified policy
// [prod planner N>5 + the GENUINE raw-kinematics endgame NN (egboidPickRaw, ~88%
// decision-agreement, NO analytic feature) for N≤5, PURE no-fallback] vs prod.
//
// The endgame change: prod intercept() selects egBoid = argmin scan-t; the unified
// policy selects egBoid = the raw NN's pick instead (window.__egNN), keeping prod's
// commit-and-hold + scan-based AIM verbatim. So the NN decides WHICH boid; prod's
// geometry decides HOW to aim. Pure no-fallback in the decision.
//
// Hypothesis: the 12% NN decision-disagreements are outcome-equivalent near-ties →
// the unified policy may still CLEAR >=95% of games (vs prod 100%) even at 88%
// agreement — the metric that matters is task success, not behavioral cloning.
//
//   node clear_rate.js --cell 2560x1440 --policy nn|prod --seeds 20 --seedStart 290000 --maxFrames 20000
'use strict';
const path = require('path');
const { createGame } = require(path.join(__dirname, '..', 'stepper.js'));
const { loadEgStudentRaw } = require(path.join(__dirname, 'egboidPickRaw.js'));

const argv = (n, d) => { const i = process.argv.indexOf('--' + n); return i >= 0 ? process.argv[i + 1] : d; };
const [W, H] = argv('cell', '1024x768').split('x').map(Number);
const POLICY = argv('policy', 'prod');
const SEEDS = parseInt(argv('seeds', '20'), 10);
const SEED0 = parseInt(argv('seedStart', '290000'), 10);   // sealed natural (fresh, never trained)
const MAXF = parseInt(argv('maxFrames', '20000'), 10);
const pickRaw = loadEgStudentRaw(path.join(__dirname, 'eg_weights_raw.json'));

// UNIFIED-NN transform: replace intercept's scan-argmin egBoid SELECTION with the NN.
const SCAN_ARGMIN = 'for (i = 0; i < boids.length; i++) { var c = scan(boids[i]); if (c && c.t < bestT) { bestT = c.t; egBoid = boids[i]; } }';
function nnTransform(file, code) {
    if (file !== 'predator_cheap.js') return null;
    if (code.indexOf(SCAN_ARGMIN) < 0) throw new Error('scan-argmin anchor not found');
    return code.replace(SCAN_ARGMIN, 'egBoid = boids[window.__egNN(pred, boids)];');
}

async function runGame(seed) {
    const g = await createGame({ policyDir: path.join(__dirname, '..', '..', '..', 'js'),
        W, H, seed, fastRender: true, transform: POLICY === 'nn' ? nnTransform : null });
    if (POLICY === 'nn') {
        g.win.__egNN = (pred, boids) => pickRaw({
            px: pred.position.x, py: pred.position.y, pvx: pred.velocity.x, pvy: pred.velocity.y,
            psize: pred.currentSize, bx: boids.map(b => b.position.x), by: boids.map(b => b.position.y),
            bvx: boids.map(b => b.velocity.x), bvy: boids.map(b => b.velocity.y),
        }, { W, Hc: H }).egIdx;
    }
    let f = 0;
    for (; f < MAXF; f++) { g.stepFrame(); if (g.boidCount() === 0) break; }
    return { cleared: g.boidCount() === 0, frames: g.frame(), left: g.boidCount() };
}

(async () => {
    let cleared = 0; const frames = [];
    for (let s = 0; s < SEEDS; s++) {
        const r = await runGame(SEED0 + s);
        if (r.cleared) { cleared++; frames.push(r.frames); }
    }
    frames.sort((a, b) => a - b);
    const med = frames.length ? frames[frames.length >> 1] : -1;
    // emit a parseable line for the orchestrator
    console.log(JSON.stringify({ cell: argv('cell'), policy: POLICY, seeds: SEEDS, seedStart: SEED0,
        cleared, clearRate: cleared / SEEDS, medianFramesToClear: med }));
})().catch(e => { console.error(e); process.exit(1); });
