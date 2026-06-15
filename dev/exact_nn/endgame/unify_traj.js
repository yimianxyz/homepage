// unify_traj.js — measurement 2: full-game OUTCOME of the UNIFIED policy (prod
// planCheap rollout-planner for ALL N≥1, no N≤5 intercept gate; commit-and-hold
// every D=16 frames) vs prod (planner N>5 + intercept N≤5), same seeds, to
// extinction. The decisive, mapping-free test: does un-gating the planner into the
// endgame REPRODUCE prod's clears, or REGRESS to the tail-chase the intercept()
// torus-scan was added to fix (prod comment: planner "never clears the last boid
// 12-18% of the time on big screens")?
//
//   node unify_traj.js [--seeds 20] [--maxFrames 14000] [--cells a,b,..]
'use strict';
const fs = require('fs'), path = require('path');
const { createGame } = require(path.join(__dirname, '..', 'stepper.js'));

const argv = (n, d) => { const i = process.argv.indexOf('--' + n); return i >= 0 ? process.argv[i + 1] : d; };
const SEEDS = parseInt(argv('seeds', '20'), 10);
const MAXF = parseInt(argv('maxFrames', '14000'), 10);
const START_BOIDS = parseInt(argv('startBoids', '0'), 10);   // >0 → endgame-only games (fast)
const SCATTER = process.argv.includes('--scatter');
const CELLS = (argv('cells', '390x844,1024x768,1512x982,1680x1050,2560x1440')).split(',')
    .map(s => { const [W, H] = s.split('x').map(Number); return { id: s, W, H }; });

// UNIFIED transform: delete the N≤5 intercept gate → planCheap runs for all N≥1.
function unifyTransform(file, code) {
    if (file !== 'predator_cheap.js') return null;
    const GATE = 'if (boids.length <= 5) return intercept(pred, boids);';
    if (code.indexOf(GATE) < 0) throw new Error('intercept gate not found');
    return code.replace(GATE, '/* UNIFIED: planCheap for all N (gate removed) */');
}

async function runToEnd(opt, transform) {
    const g = await createGame({ policyDir: path.join(__dirname, '..', '..', '..', 'js'),
        W: opt.W, H: opt.H, seed: opt.seed, fastRender: true, transform,
        startBoids: START_BOIDS > 0 ? START_BOIDS : undefined, scatter: SCATTER });
    let f = 0;
    for (; f < MAXF; f++) { g.stepFrame(); if (g.boidCount() === 0) break; }
    return { cleared: g.boidCount() === 0, frames: g.frame(), left: g.boidCount(), eaten: g.sim.boidsEaten };
}

(async () => {
    const agg = {};
    let pClear = 0, uClear = 0, n = 0, framesMatch = 0;
    for (const cell of CELLS) {
        const st = agg[cell.id] = { pClear: 0, uClear: 0, n: 0, uLeftSum: 0, pLeftSum: 0 };
        for (let s = 0; s < SEEDS; s++) {
            const seed = 120000 + s;   // natural full-game seeds (disjoint from sealed ≥290000)
            const prod = await runToEnd({ W: cell.W, H: cell.H, seed }, null);
            const uni = await runToEnd({ W: cell.W, H: cell.H, seed }, unifyTransform);
            st.n++; n++;
            if (prod.cleared) { st.pClear++; pClear++; }
            if (uni.cleared) { st.uClear++; uClear++; }
            st.pLeftSum += prod.left; st.uLeftSum += uni.left;
            if (prod.cleared && uni.cleared && prod.frames === uni.frames) framesMatch++;
        }
        process.stderr.write(`[${cell.id}] prod-clear ${st.pClear}/${st.n}  UNIFIED-clear ${st.uClear}/${st.n}  (uni left-uncaught avg ${(st.uLeftSum / st.n).toFixed(2)} vs prod ${(st.pLeftSum / st.n).toFixed(2)})\n`);
    }
    console.log('=== UNIFIED (planCheap all-N) vs prod — full-game clear outcome ===');
    console.log(`prod cleared    : ${pClear}/${n} (${(pClear / n * 100).toFixed(1)}%)`);
    console.log(`UNIFIED cleared : ${uClear}/${n} (${(uClear / n * 100).toFixed(1)}%)`);
    console.log(`clear regression: ${((pClear - uClear) / n * 100).toFixed(1)}pp  (games prod clears but the unified planner does NOT → the tail-chase intercept() fixes)`);
    console.log('per-cell prod/UNIFIED clear:', Object.keys(agg).map(c => `${c}:${agg[c].pClear}/${agg[c].uClear} of ${agg[c].n}`).join('  '));
})().catch(e => { console.error(e); process.exit(1); });
