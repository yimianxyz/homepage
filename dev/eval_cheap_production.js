// Faithful headless eval of the SHIPPED production path. Loads the real js/
// files exactly as index.html does (predator.js delegates to window.__cheap),
// gates on window.__predatorReady (the value_net.json load) like boids.js, then
// runs the live two-pass frame loop (simTick; tick; render) and reports catches.
// Seeds == sim_torch (mulberry32). Should match dev/eval_cheap_headless.js's
// cheap number (~8.6 @1500f/seed300000 two-pass) — that's the wiring check.
//
//   node dev/eval_cheap_production.js --seedStart 300000 --seeds 16 --frames 1500
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function parseArgs(argv) {
    const a = { js: path.join(__dirname, '..', 'js'), seedStart: 300000, seeds: 16,
        frames: 1500, width: 1680, height: 1680, numBoids: 120, twopass: true };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--frames') a.frames = +argv[++i];
        else if (k === '--singlepass') a.twopass = false;
    }
    return a;
}
function stubCtx() {
    const noop = function () {}; const o = {};
    ['beginPath', 'moveTo', 'lineTo', 'stroke', 'fill', 'arc', 'clearRect', 'fillRect',
        'strokeRect', 'closePath', 'save', 'restore', 'translate', 'rotate', 'scale',
        'fillText', 'setLineDash', 'ellipse', 'quadraticCurveTo', 'bezierCurveTo'].forEach(m => o[m] = noop);
    o.createLinearGradient = () => ({ addColorStop: noop });
    return o;
}
// Same set + order index.html loads (minus pure-DOM files).
const JS_FILES = ['rng.js', 'vector.js', 'boid.js', 'predator.js', 'simulation.js',
    'cheap_planner.js', 'predator_cheap.js'];

function buildContext(opt) {
    const ctx = stubCtx();
    const win = { innerWidth: opt.width, innerHeight: opt.height,
        matchMedia: () => ({ matches: false, addEventListener: () => {} }), addEventListener: () => {} };
    const sandbox = {
        navigator: { userAgent: 'Node' }, window: win,
        document: { getElementById: () => ({ getContext: () => ctx, width: opt.width, height: opt.height }), addEventListener: () => {} },
        fetch: function (url) {
            const f = path.join(opt.js, path.basename(url));
            return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(JSON.parse(fs.readFileSync(f, 'utf8'))) });
        },
        renderActivationViz: () => {}, Math, Date, console, setTimeout, setImmediate, Promise,
    };
    sandbox.self = sandbox; sandbox.global = sandbox;
    const context = vm.createContext(sandbox);
    for (const f of JS_FILES) vm.runInContext(fs.readFileSync(path.join(opt.js, f), 'utf8'), context, { filename: f });
    return { sandbox, win };
}
async function waitReady(win) {
    for (let i = 0; i < 100; i++) {
        let done = false;
        win.__predatorReady.then(() => done = true);
        await new Promise(r => setImmediate(r));
        if (done) return;
    }
    throw new Error('__predatorReady never resolved (value_net.json load)');
}
function runEpisode(sandbox, seed, opt) {
    sandbox.setSimSeed(seed, 12);
    sandbox.NUM_BOIDS = opt.numBoids;
    const sim = new sandbox.Simulation('boids1');
    sim.canvasWidth = opt.width; sim.canvasHeight = opt.height;
    sim.initialize(false);
    sandbox.setFrameMs(12);
    for (let f = 0; f < opt.frames; f++) {
        sandbox.simTick();
        if (opt.twopass) sim.tick();
        sim.render();
    }
    return sim.boidsEaten;
}
async function main() {
    const opt = parseArgs(process.argv);
    const { sandbox, win } = buildContext(opt);
    await waitReady(win);
    let sum = 0; const per = [];
    for (let i = 0; i < opt.seeds; i++) {
        const c = runEpisode(sandbox, opt.seedStart + i, opt);
        sum += c; per.push(c);
        process.stderr.write('  ' + (i + 1) + '/' + opt.seeds + '\r');
    }
    console.log(JSON.stringify({ seedStart: opt.seedStart, seeds: opt.seeds, frames: opt.frames,
        twopass: opt.twopass, mean_cheap: sum / opt.seeds, per: per }));
}
main().catch(e => { console.error(e); process.exit(1); });
