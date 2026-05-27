// Dump per-frame predator/boid state from the JS Oracle for a single seed.
// Used by dev/trace_diff.py to compare against sim_torch frame-by-frame.
'use strict';
const fs = require('fs');
const path = require('path');
const { Oracle } = require('./oracle');

function parseArgs(argv) {
    const args = { seed: 100, frames: 50, weights: 'js/predator_weights.json',
                   out: '/tmp/trace_js.json', autoTarget: 'flock_centroid' };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--seed') args.seed = +argv[++i];
        else if (a === '--frames') args.frames = +argv[++i];
        else if (a === '--weights') args.weights = argv[++i];
        else if (a === '--out') args.out = argv[++i];
        else if (a === '--autoTarget') args.autoTarget = argv[++i];
    }
    return args;
}

function main() {
    const args = parseArgs(process.argv);
    const PredatorNN = require('../js/predator_nn');
    const json = JSON.parse(fs.readFileSync(args.weights, 'utf8'));
    const model = PredatorNN.loadModel(json);
    const nnFn = features => model.forward(features);
    const o = new Oracle({ seed: args.seed, nnFn, autoTargetMode: args.autoTarget });

    const states = [];
    for (let f = 0; f < args.frames; f++) {
        // Capture pre-step state
        const p = o.sim.predator;
        const boids = o.sim.boids;
        states.push({
            frame: f,
            pred_pos: [p.position.x, p.position.y],
            pred_vel: [p.velocity.x, p.velocity.y],
            pred_auto: [p.autonomousTarget.x, p.autonomousTarget.y],
            pred_size: p.currentSize,
            n_alive: boids.length,
            catches: o.sim.catchCount || 0,
            boid0_pos: boids.length > 0 ? [boids[0].position.x, boids[0].position.y] : null,
            boid0_vel: boids.length > 0 ? [boids[0].velocity.x, boids[0].velocity.y] : null,
        });
        o.step(false);
    }
    fs.writeFileSync(args.out, JSON.stringify(states, null, 2));
    console.log('wrote', args.out, 'frames=', states.length);
}

main();
