// eg_disagree_probe.js — dissect L1e egBoid DISAGREEMENTS. Re-runs given cells/seeds
// (shadow: follows prod's exact trajectory), and at every commit where the NN's argmin
// scan-t pick != prod's exact egPick, dumps the full state + the NN's predicted scan-t
// per boid vs prod's EXACT scan-t (eg_scan) + the cert decision + the margin. Reveals
// WHETHER a disagreement is a genuine NN misprediction (and which boid it mispredicts)
// or a regime artifact (nearest-fallback / unreachable). Diagnostic for the audit.
//
//   node eg_disagree_probe.js --cells 2560x1440 --base 270000 --seeds 700 [--minMargin 16]
'use strict';
const path = require('path');
const { createGame } = require('../stepper.js');
const { loadEgStudent } = require(path.join(__dirname, '..', 'endgame', 'egboidPick.js'));
const { egPick, scanT } = require(path.join(__dirname, '..', 'endgame', 'eg_scan.js'));
const { certify } = require(path.join(__dirname, '..', 'endgame', 'eg_bound.js'));

function parse(argv) {
    const a = { cells: '2560x1440', base: 270000, seeds: 700, minMargin: 0, max: 50, maxFrames: 4000, natural: false };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--cells') a.cells = argv[++i];
        else if (k === '--base') a.base = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--minMargin') a.minMargin = +argv[++i];
        else if (k === '--max') a.max = +argv[++i];
        else if (k === '--maxFrames') a.maxFrames = +argv[++i];
        else if (k === '--natural') a.natural = true;
        else throw new Error('unknown arg ' + k);
    }
    return a;
}

async function main() {
    const opt = parse(process.argv);
    const pick = loadEgStudent(path.join(__dirname, '..', 'endgame', 'eg_weights.json'));
    const cells = opt.cells.split(',').map(s => { const [W, H] = s.split('x').map(Number); return { W, H }; });
    const policyDir = path.join(__dirname, '..', '..', '..', 'js');
    const found = [];
    let commits = 0;

    for (const c of cells) {
        for (let si = 0; si < opt.seeds && found.length < opt.max; si++) {
            const seed = opt.base + si;
            const startBoids = opt.natural ? 0 : (2 + (si % 4));
            const game = await createGame({ policyDir, W: c.W, H: c.H, seed,
                startBoids, scatter: !opt.natural, fastRender: true });
            const sim = game.sim, pred = sim.predator;
            for (let f = 0; f < opt.maxFrames && game.boidCount() > 0; f++) {
                const boids = sim.boids ? sim.boids.filter(b => b && b.alive !== false) : null;
                // only inspect commit decisions in the endgame regime
                if (boids && boids.length >= 1 && boids.length <= 5) {
                    const bs = boids.map(b => ({ x: b.position.x, y: b.position.y, vx: b.velocity.x, vy: b.velocity.y }));
                    const snap = { px: pred.position.x, py: pred.position.y,
                        bx: bs.map(b => b.x), by: bs.map(b => b.y), bvx: bs.map(b => b.vx), bvy: bs.map(b => b.vy) };
                    const r = pick(snap, { W: c.W, Hc: c.H });
                    const gt = egPick(pred.position.x, pred.position.y, bs, c.W, c.H);
                    commits++;
                    if (r.egIdx !== gt.egIdx && Number.isFinite(r.margin) && r.margin >= opt.minMargin) {
                        const certFires = certify(pred.position.x, pred.position.y, bs, c.W, c.H, r.egIdx);
                        if (!certFires) found.push({
                            cell: c.W + 'x' + c.H, seed, frame: f, n: boids.length,
                            margin: +r.margin.toFixed(2), nnPick: r.egIdx, prodPick: gt.egIdx,
                            nnScanT: r.ts.map(t => +t.toFixed(1)),
                            prodScanT: gt.ts.map(t => t == null ? null : t),
                            nearestFallback: gt.nearestFallback,
                            predErr_onNNpick: +(r.ts[r.egIdx] - (gt.ts[r.egIdx] == null ? NaN : gt.ts[r.egIdx])).toFixed(1),
                            predErr_onProdPick: +(r.ts[gt.egIdx] - (gt.ts[gt.egIdx] == null ? NaN : gt.ts[gt.egIdx])).toFixed(1),
                        });
                    }
                }
                game.stepFrame();
                if (found.length >= opt.max) break;
            }
        }
    }
    console.log(JSON.stringify({ commitsInspected: commits, disagreementsFound: found.length,
        minMargin: opt.minMargin, examples: found }, null, 1));
}
main().catch(e => { console.error(e); process.exit(1); });
