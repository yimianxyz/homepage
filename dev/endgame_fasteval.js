// Endgame eval: the LIVE-UX metric the 1500-frame benchmark misses. The real
// page runs forever and boids deplete, so the predator eventually faces a few
// SCATTERED fast singletons on the open torus — the hardest case for a slow
// pursuer and the one the user flagged. Here we seed K boids at random scattered
// positions (not the game's single-point spawn), predator at center, and measure
// frames-to-clear all K (and how many are caught within a budget).
//
//   node dev/endgame_fasteval.js --policyDir dev/exp/js --config '{"wrap":true}' \
//        --W 390 --H 844 --startBoids 3 --seeds 64 --maxFrames 3000
'use strict';
const path = require('path');
const { buildHarness } = require('./fasteval.js');

function parseArgs(argv) {
    const a = { policyDir: path.join(__dirname, '..', 'js'), W: 390, H: 844,
        seedStart: 300000, seeds: 64, startBoids: 3, maxFrames: 3000, config: null };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--policyDir') a.policyDir = argv[++i];
        else if (k === '--W') a.W = +argv[++i];
        else if (k === '--H') a.H = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--startBoids') a.startBoids = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--config') a.config = argv[++i];
        else if (k === '--perseed') a.perseed = true;
    }
    return a;
}

async function main() {
    const opt = parseArgs(process.argv);
    const ttc = [];     // frames-to-clear (maxFrames if not cleared)
    const caught = [];  // boids caught within maxFrames
    for (let i = 0; i < opt.seeds; i++) {
        const seed = opt.seedStart + i;
        const built = buildHarness(opt);
        const api = built.api, win = built.win, g = global;
        if (win.__predatorReady && win.__predatorReady.then) await win.__predatorReady;
        g.NUM_BOIDS = opt.startBoids;       // override the device boid count
        api.setSimSeed(seed, 12);
        const sim = new api.Simulation('boids1');
        sim.canvasWidth = opt.W; sim.canvasHeight = opt.H;
        sim.initialize(false);
        // Scatter the boids to random positions (the game spawns them at one point;
        // the endgame is about already-dispersed singletons).
        for (let b = 0; b < sim.boids.length; b++) {
            sim.boids[b].position.x = g.simRandom() * opt.W;
            sim.boids[b].position.y = g.simRandom() * opt.H;
        }
        if (api.setFrameMs) api.setFrameMs(12);
        sim.tick();   // one-time pre-loop tick() matching the browser's run()
        let clearedAt = opt.maxFrames;
        for (let f = 0; f < opt.maxFrames; f++) {
            api.simTick(); sim.tick(); sim.render();
            if (sim.boids.length === 0) { clearedAt = f + 1; break; }
        }
        ttc.push(clearedAt);
        caught.push(opt.startBoids - sim.boids.length);
    }
    const n = ttc.length;
    const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const se = arr => { const m = mean(arr); const sd = Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, arr.length - 1)); return sd / Math.sqrt(arr.length); };
    const clearedFrac = ttc.filter(t => t < opt.maxFrames).length / n;
    const out = { policyDir: path.basename(opt.policyDir), W: opt.W, H: opt.H,
        startBoids: opt.startBoids, seeds: n, maxFrames: opt.maxFrames,
        meanTTC: +mean(ttc).toFixed(1), seTTC: +se(ttc).toFixed(1),
        clearedFrac: +clearedFrac.toFixed(3),
        meanCaught: +mean(caught).toFixed(3) };
    if (opt.perseed) { out.ttc = ttc; out.caught = caught; }
    console.log(JSON.stringify(out));
}
main().catch(e => { console.error(e); process.exit(1); });
