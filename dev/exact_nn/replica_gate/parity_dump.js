// Dump a real prod rollout (+ plan decision) to JSON so skeleton_torch.py can
// validate its STRUCTURE against the shipped rolloutFlatState/planCheap.
// Patches `window.__cheap = {` in a /tmp copy of predator_cheap.js to expose
// the closure-private internals (same trick as dev/diff_rollout_vs_game.js).
//
//   node parity_dump.js --N 120 --H 10 --out parity_N120_H10.json
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');

function parseArgs(argv) {
    const a = { N: 120, W: 1512, H: 982, seed: 200000, warmFrames: 120, horizon: 90,
        out: null, policyDir: path.join(__dirname, '..', '..', '..', 'js') };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--N') a.N = +argv[++i];
        else if (k === '--H') a.horizon = +argv[++i];
        else if (k === '--seed') a.seed = +argv[++i];
        else if (k === '--out') a.out = argv[++i];
    }
    if (!a.out) a.out = `parity_N${a.N}_H${a.horizon}.json`;
    return a;
}

async function main() {
    const opt = parseArgs(process.argv);
    // patched copy of js/ with instrumented predator_cheap.js
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pjs-'));
    for (const f of fs.readdirSync(opt.policyDir)) {
        if (f.endsWith('.js') || f.endsWith('.json'))
            fs.copyFileSync(path.join(opt.policyDir, f), path.join(tmp, f));
    }
    const pc = path.join(tmp, 'predator_cheap.js');
    const src = fs.readFileSync(pc, 'utf8');
    const patched = src.replace('window.__cheap = {',
        'window.__cheap = { _rollout: rolloutFlatState, _snapshot: snapshot, ' +
        '_configure: configure, _plan: planCheap, _candidates: candidates,');
    if (patched === src) throw new Error('patch anchor not found');
    fs.writeFileSync(pc, patched);

    const { buildHarness } = require(path.join(__dirname, '..', '..', 'fasteval.js'));
    const built = buildHarness({ W: opt.W, H: opt.H, policyDir: tmp });
    const win = built.win;
    if (win.__predatorReady) await win.__predatorReady;
    built.api.setSimSeed(opt.seed, 12);
    const sim = new built.api.Simulation('boids1');
    sim.canvasWidth = opt.W; sim.canvasHeight = opt.H;
    sim.initialize(false);
    if (built.api.setFrameMs) built.api.setFrameMs(12);
    sim.tick();
    const V = global.Vector;
    const snapState = () => ({
        boids: sim.boids.map(b => ({
            position: new V(b.position.x, b.position.y),
            velocity: new V(b.velocity.x, b.velocity.y) })),
        pred: { x: sim.predator.position.x, y: sim.predator.position.y,
            vx: sim.predator.velocity.x, vy: sim.predator.velocity.y,
            currentSize: sim.predator.currentSize, lastFeedTime: sim.predator.lastFeedTime },
    });
    let frozen = snapState();
    for (let f = 0; f < opt.warmFrames; f++) {
        built.api.simTick(); sim.tick(); sim.render();
        if (sim.boids.length >= opt.N) frozen = snapState();
    }
    let boids = frozen.boids;
    if (opt.N < boids.length) {
        const sub = [], stride = boids.length / opt.N;
        for (let i = 0; i < opt.N; i++) sub.push(boids[Math.round(i * stride)]);
        boids = sub;
    }
    const p = frozen.pred;
    const predStub = { position: new V(p.x, p.y), velocity: new V(p.vx, p.vy),
        currentSize: p.currentSize, lastFeedTime: p.lastFeedTime };
    win.__cheap._configure({ canvasWidth: opt.W, canvasHeight: opt.H });
    const snap = win.__cheap._snapshot(predStub, boids);
    // target = E3D patrol (computeEvolvedTarget is a global in this context)
    const lite = boids.map(b => ({ position: { x: b.position.x, y: b.position.y },
        velocity: { x: b.velocity.x, y: b.velocity.y } }));
    const e3d = global.computeEvolvedTarget({ x: p.x, y: p.y }, lite,
        global.EVOLVED_PATROL, null);
    const rr = win.__cheap._rollout(snap, e3d.x, e3d.y, opt.horizon);
    const plan = win.__cheap._plan(snap);
    // phase-A internals for the structural diff (cp_* are globals here)
    const cands = win.__cheap._candidates(snap);
    const st = { px: snap.px, py: snap.py, pvx: snap.pvx, pvy: snap.pvy,
        psize: snap.psize, bx: snap.bx, by: snap.by, bvx: snap.bvx, bvy: snap.bvy,
        nAlive: snap.bx.length };
    const fr = global.cp_features(st, cands, global.PREDATOR_MAX_SPEED, global.PREDATOR_MAX_FORCE);
    const net = JSON.parse(fs.readFileSync(path.join(tmp, 'value_net.json'), 'utf8'));
    const vprior = global.cp_value(net, fr.feat, fr.ctx);
    const pidx = cands.map((_, k) => k);
    pidx.sort((a, b) => {
        const pa = fr.feat[a][18] - fr.feat[a][16], pb = fr.feat[b][18] - fr.feat[b][16];
        return (pb - pa) || (a - b);
    });
    const out = { W: opt.W, Hc: opt.H, H: opt.horizon, n: snap.bx.length,
        tx: e3d.x, ty: e3d.y, snap, catches: rr.catches, term: rr.term, plan,
        cands, feat: fr.feat, ctx: fr.ctx, vprior, pidx4: pidx.slice(0, 4) };
    fs.writeFileSync(path.join(__dirname, opt.out), JSON.stringify(out));
    console.log(`wrote ${opt.out}: n=${out.n} H=${out.H} catches=${rr.catches} ` +
        `termAlive=${rr.term.bx.length} plan=(${plan.x.toFixed(3)},${plan.y.toFixed(3)})`);
}
main().catch(e => { console.error(e); process.exit(1); });
