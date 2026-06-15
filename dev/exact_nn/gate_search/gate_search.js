// gate_search.js — full-game eval harness with a CONFIGURABLE planner/endgame gate,
// for the split-threshold search (#5). Drives the DEPLOYED policy (planner N>gate +
// the raw-kinematics endgame NN N<=gate) to extinction, with the gate pluggable for
// three split-rule families (methodology correction, 3-lab panel):
//   count   : enter endgame when N <= T                         (param T)
//   density : enter endgame when N <= rho * (W*Hc/1e6)          (param rho, boids/Mpx^2)
//   horizon : enter endgame when #boids within the planner's ~Hs-frame torus reach
//             (R = PREDATOR_MAX_SPEED*Hs) is <= h               (param h; auto-adapts to screen)
// N is monotone in extinction games, so the gate is a one-way LATCH (no exit
// hysteresis — methodology #4). Metrics per game:
//   stuck   : longest run of frames with 0 catches >= STUCK (=300) -> frozen-homepage
//             disaster (PRIMARY, minimize), counted even if the game eventually clears
//   thru    : catches / frames (THROUGHPUT, incl. stuck tails)
//   cleared : sanity check (near-binary in T)
//   frames  : time-to-clear (cleared games)
//
//   node gate_search.js --cell 1024x768:120 --rule count --param 5 --seeds 8 \
//        --seedStart 700000 --maxFrames 30000 --policyDir <deployed js/>
'use strict';
const path = require('path');
const STEPPER = '/workspace/.team/wt-exact-nn-oracle/dev/exact_nn/stepper.js';
const { createGame } = require(STEPPER);

const argv = (n, d) => { const i = process.argv.indexOf('--' + n); return i >= 0 ? process.argv[i + 1] : d; };
const cellSpec = argv('cell', '1024x768:120').split(':');
const [W, H] = cellSpec[0].split('x').map(Number);
const BOIDS = parseInt(cellSpec[1] || '120', 10);
const RULE = argv('rule', 'count');
const PARAM = parseFloat(argv('param', '5'));
const SEEDS = parseInt(argv('seeds', '8'), 10);
const SEED0 = parseInt(argv('seedStart', '700000'), 10);
const MAXF = parseInt(argv('maxFrames', '30000'), 10);
const STUCK = parseInt(argv('stuck', '300'), 10);
const POLICY = argv('policyDir', '/workspace/.team/wt-gate-search/js');
const BORDER = 10, PRED_SPEED = 2.5, HS = 90;

// the gate ENTER condition (latched once true). pure fn of current state + cfg.
function gateEnter(boids, pred, cfg) {
    const n = boids.length;
    if (RULE === 'count') return n <= PARAM;
    if (RULE === 'density') return n <= PARAM * (cfg.W * cfg.Hc / 1e6);
    if (RULE === 'horizon') {
        // # boids within the planner's ~Hs-frame torus reach R; switch when <= h
        const R = PRED_SPEED * HS, R2 = R * R, PX = cfg.W + 2 * BORDER, PY = cfg.Hc + 2 * BORDER;
        let within = 0;
        for (let i = 0; i < n; i++) {
            let dx = boids[i].position.x - pred.position.x, dy = boids[i].position.y - pred.position.y;
            dx -= PX * Math.round(dx / PX); dy -= PY * Math.round(dy / PY);
            if (dx * dx + dy * dy <= R2) within++;
        }
        return within <= PARAM;
    }
    throw new Error('unknown rule ' + RULE);
}

// transform: replace the deployed count<=5/>=7 latch with a window.__gate latch.
function gateTransform(file, code) {
    if (file !== 'predator_cheap.js') return null;
    const A = '            if (!inEndgame && boids.length <= 5) inEndgame = true;\n' +
        '            else if (inEndgame && boids.length >= 7) inEndgame = false;\n' +
        '            if (inEndgame) return intercept(pred, boids);   // ENDGAME: NN-selected egBoid + torus aim';
    if (code.indexOf(A) < 0) throw new Error('gate anchor not found');
    return code.replace(A,
        '            if (!inEndgame && window.__gate(pred, boids, cfg)) inEndgame = true;\n' +
        '            if (inEndgame) return intercept(pred, boids);');
}

async function runGame(seed) {
    const g = await createGame({ policyDir: POLICY, W, H, seed,
        startBoids: BOIDS, fastRender: true, transform: gateTransform });
    g.win.__gate = (pred, boids, cfg) => gateEnter(boids, pred, cfg);
    let f = 0, prevEaten = 0, lastCatch = 0, maxGap = 0;
    for (; f < MAXF; f++) {
        g.stepFrame();
        const eaten = g.eaten();
        if (eaten > prevEaten) { const gap = g.frame() - lastCatch; if (gap > maxGap) maxGap = gap; lastCatch = g.frame(); prevEaten = eaten; }
        if (g.boidCount() === 0) break;
    }
    const endF = g.frame();
    const tailGap = endF - lastCatch;                          // last-catch -> end: the endgame stuck tail
    if (tailGap > maxGap) maxGap = tailGap;
    return { cleared: g.boidCount() === 0, frames: endF, catches: g.eaten(), maxGap, tailGap, stuck: maxGap >= STUCK };
}

function pct(arr, p) { if (!arr.length) return -1; const a = arr.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(p * a.length))]; }

(async () => {
    let cleared = 0, totCatch = 0, totFrames = 0; const clearFrames = [], maxGaps = [], tailGaps = [];
    let s600 = 0, s1000 = 0;
    for (let s = 0; s < SEEDS; s++) {
        const r = await runGame(SEED0 + s);
        if (r.cleared) { cleared++; clearFrames.push(r.frames); }
        maxGaps.push(r.maxGap); tailGaps.push(r.tailGap);
        if (r.maxGap >= 600) s600++; if (r.maxGap >= 1000) s1000++;
        totCatch += r.catches; totFrames += r.frames;
    }
    console.log(JSON.stringify({
        cell: cellSpec[0], boids: BOIDS, rule: RULE, param: PARAM, seeds: SEEDS, seedStart: SEED0,
        stuck300: maxGaps.filter(g => g >= 300).length / SEEDS, stuck600: s600 / SEEDS, stuck1000: s1000 / SEEDS,
        throughput: totCatch / totFrames, clearRate: cleared / SEEDS,
        maxGapMed: pct(maxGaps, 0.5), maxGapP90: pct(maxGaps, 0.9),
        tailGapMed: pct(tailGaps, 0.5), tailGapP90: pct(tailGaps, 0.9),
        medClearFrames: pct(clearFrames, 0.5),
    }));
})().catch(e => { console.error('ERR', RULE, PARAM, cellSpec[0], e.message); process.exit(1); });
