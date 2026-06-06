// Per-frame trace of the JS browser cheap policy (seed) — predator pos + catches,
// to diff vs the GPU trace and localize the 2x gap.
//   node trace_js_cheap.js --js /tmp/js_strict --seed 200000 --frames 160
const fs = require('fs');
const path = require('path');
const vm = require('vm');
function parseArgs(argv) {
    const a = { js: path.join(__dirname, '..', 'js'), seed: 200000, frames: 160, width: 1680, height: 1680, numBoids: 120 };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--js') a.js = argv[++i];
        else if (k === '--seed') a.seed = +argv[++i];
        else if (k === '--frames') a.frames = +argv[++i];
    }
    return a;
}
function stubCtx() { const noop = () => {}; return new Proxy({}, { get: () => noop }); }
const JS_FILES = ['rng.js', 'vector.js', 'boid.js', 'predator.js', 'simulation.js', 'cheap_planner.js', 'predator_cheap.js'];
function buildContext(opt) {
    const ctx = stubCtx();
    const win = { innerWidth: opt.width, innerHeight: opt.height, matchMedia: () => ({ matches: false, addEventListener: () => {} }), addEventListener: () => {} };
    const sandbox = { navigator: { userAgent: 'Node' }, window: win,
        document: { getElementById: () => ({ getContext: () => ctx, width: opt.width, height: opt.height }), addEventListener: () => {} },
        fetch: function (url) { const f = path.join(opt.js, path.basename(url)); return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(JSON.parse(fs.readFileSync(f, 'utf8'))) }); },
        renderActivationViz: () => {}, Math, Date, console, setTimeout, setImmediate, Promise };
    sandbox.self = sandbox; sandbox.global = sandbox;
    const context = vm.createContext(sandbox);
    for (const f of JS_FILES) vm.runInContext(fs.readFileSync(path.join(opt.js, f), 'utf8'), context, { filename: f });
    return { sandbox, win };
}
async function waitReady(win) {
    for (let i = 0; i < 100; i++) { let done = false; win.__predatorReady.then(() => done = true); await new Promise(r => setImmediate(r)); if (done) return; }
    throw new Error('__predatorReady never resolved');
}
async function main() {
    const opt = parseArgs(process.argv);
    const { sandbox, win } = buildContext(opt);
    await waitReady(win);
    sandbox.setSimSeed(opt.seed, 12);
    sandbox.NUM_BOIDS = opt.numBoids;
    const sim = new sandbox.Simulation('boids1');
    sim.canvasWidth = opt.width; sim.canvasHeight = opt.height;
    sim.initialize(false);
    sandbox.setFrameMs(12);
    const pred = sim.predator || (sim.predators && sim.predators[0]);
    for (let f = 0; f < opt.frames; f++) {
        sandbox.simTick(); sim.tick(); sim.render();
        const p = sim.predator || (sim.predators && sim.predators[0]);
        const b0 = sim.boids[0];
        console.log(JSON.stringify({ f: f, px: +p.position.x.toFixed(4), py: +p.position.y.toFixed(4),
            pvx: +p.velocity.x.toFixed(4), pvy: +p.velocity.y.toFixed(4),
            b0x: b0 ? +b0.position.x.toFixed(4) : null, b0y: b0 ? +b0.position.y.toFixed(4) : null,
            sz: +(p.currentSize != null ? p.currentSize : 0).toFixed(2), catches: sim.boidsEaten }));
    }
}
main().catch(e => { console.error(e); process.exit(1); });
