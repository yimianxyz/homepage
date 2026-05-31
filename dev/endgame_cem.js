'use strict';
// ===========================================================================
// ENDGAME CEM  — closed-loop search for the best LAST-BOID interception law
// ---------------------------------------------------------------------------
// North-star metric: faithful-JS endgame time-to-extinction (TTE). The deployed
// radial set-net catches a lone boid 32/32 but SLOWLY (mean ~2331 frames ≈ a
// viewer watching one dot for ~40s) — perceptually "it never catches it". Naive
// pursuit / lead-intercept laws are all WORSE than the net (see endgame_eval).
// So we SEARCH a compact policy directly against closed-loop TTE.
//
// Policy: tiny MLP   in(6) -> tanh(H) -> out(2)
//   inputs  = [ dx,dy (toroidal nearest-boid offset), bvx,bvy (boid vel),
//               pvx,pvy (predator vel) ]  (scaled)
//   output  = heading o; desired = setMag(o, 2.5); steer = limit(desired-pvel, 0.05)
//   (same control structure as the deployed analytic chase, so it is a drop-in
//    replacement for getAutonomousForce in the endgame regime.)
//
// Optimizer: CEM (diagonal Gaussian) with common-random-number episodes so
// candidates are compared on identical seeds/sizes/pump-speeds. Population eval
// is sharded across worker_threads. Run one instance per VM (island model);
// champions are JS-verified by dev/endgame_eval against the radial baseline.
//
//   node dev/endgame_cem.js [--H 16] [--pop 48] [--elite 0.25] [--iters 30]
//        [--seeds 12] [--workers 8] [--maxFrames 6000] [--sigma0 0.6]
//        [--out dev/reports/endgame/<tag>.json] [--seedBase 1000] [--islandSeed 0]
// ===========================================================================
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

const DEV_DIR = '/tmp/dev_wt';
const FILES = ['rng.js', 'predator_radial.js', 'vector.js', 'boid.js', 'predator.js', 'simulation.js'];
const PMS = 2.5, PMF = 0.05;
const NIN = 6;
const PSCALE = 200, VSCALE = 6; // input normalizers

// ---------- MLP ----------
function thetaLen(H) { return NIN * H + H + H * 2 + 2; }

function makeMlp(theta, H) {
    const W1 = theta.subarray(0, NIN * H);
    const b1 = theta.subarray(NIN * H, NIN * H + H);
    const W2 = theta.subarray(NIN * H + H, NIN * H + H + 2 * H);
    const b2 = theta.subarray(NIN * H + H + 2 * H, NIN * H + H + 2 * H + 2);
    const h = new Float64Array(H);
    return function (inp) {
        for (let j = 0; j < H; j++) {
            let s = b1[j];
            const base = j * NIN;
            for (let k = 0; k < NIN; k++) s += W1[base + k] * inp[k];
            h[j] = Math.tanh(s);
        }
        let o0 = b2[0], o1 = b2[1];
        for (let j = 0; j < H; j++) { o0 += W2[j] * h[j]; o1 += W2[H + j] * h[j]; }
        return [o0, o1];
    };
}

// ---------- vector helpers ----------
function setMag(x, y, m) { const d = Math.sqrt(x * x + y * y) || 1e-12; const k = m / d; return [x * k, y * k]; }
function limit(x, y, m) { const d = Math.sqrt(x * x + y * y); if (d > m) { const k = m / d; return [x * k, y * k]; } return [x, y]; }
// RAW nearest — matches deployed perception AND the boid's own avoidance/catch
// geometry (boid.js getPredatorAvoidanceVector uses non-toroidal getDistance).
function nearestRaw(px, py, boids) {
    let bd = Infinity, idx = -1, bdx = 0, bdy = 0;
    for (let i = 0; i < boids.length; i++) {
        const dx = boids[i].position.x - px, dy = boids[i].position.y - py, d2 = dx * dx + dy * dy;
        if (d2 < bd) { bd = d2; idx = i; bdx = dx; bdy = dy; }
    }
    return { idx, dx: bdx, dy: bdy };
}

// ---------- sandbox / sim ----------
function buildSandbox(W, H) {
    const noop = () => {};
    const ctx = { beginPath: noop, moveTo: noop, lineTo: noop, stroke: noop, fill: noop, arc: noop, clearRect: noop, fillRect: noop, strokeRect: noop, save: noop, restore: noop, translate: noop, rotate: noop, scale: noop, set strokeStyle(v) {}, set fillStyle(v) {}, set lineWidth(v) {}, get strokeStyle() { return ''; }, get fillStyle() { return ''; }, get lineWidth() { return 1; } };
    const s = {
        navigator: { userAgent: 'N' },
        window: { innerWidth: W, innerHeight: H, matchMedia: () => ({ matches: false, addEventListener: noop }) },
        document: { getElementById: () => ({ getContext: () => ctx, width: W, height: H }), addEventListener: noop },
        fetch: () => new Promise(() => {}), renderActivationViz: noop, Math: Math, Date: Date, console: console,
    };
    s.global = s;
    return vm.createContext(s);
}

function loadSim(W, H) {
    const S = buildSandbox(W, H); S.W = W; S.H = H;
    for (const f of FILES) vm.runInContext(fs.readFileSync(path.join(DEV_DIR, 'js', f), 'utf8'), S, { filename: f });
    const json = JSON.parse(fs.readFileSync(path.join(DEV_DIR, 'js/predator_radial_weights.json'), 'utf8'));
    S.window.__predatorModel = S.PredatorRadial.loadModel(json);
    return S;
}

const CSCALE = PMF; // correction scale: net~O(1) -> full-cap force nudge

// Install the endgame policy as a RESIDUAL on the deployed radial/chase force:
//   f = limit( f_radial + 0.05*net(state), 0.05 )
// zero weights => f == f_radial (exact radial, can't regress); large weights can
// fully override the direction (escape the pumping basin). The net only learns the
// anti-pump correction on top of an already-strong policy.
function installResidual(sim, S, mlp) {
    const orig = sim.predator.getAutonomousForce.bind(sim.predator);
    const inp = new Float64Array(NIN);
    sim.predator.getAutonomousForce = function (boids) {
        if (boids.length === 0) return new S.Vector(0, 0);
        const f0 = orig(boids);
        const p = sim.predator;
        const n = nearestRaw(p.position.x, p.position.y, boids);
        const b = boids[n.idx];
        inp[0] = n.dx / PSCALE; inp[1] = n.dy / PSCALE;
        inp[2] = b.velocity.x / VSCALE; inp[3] = b.velocity.y / VSCALE;
        inp[4] = p.velocity.x / VSCALE; inp[5] = p.velocity.y / VSCALE;
        const net = mlp(inp);
        // apply correction with the sim's OWN Vector math => bit-identical to how the
        // deployed predator would compute it (zero net => exactly f0 => radial).
        f0.iAdd(new S.Vector(CSCALE * net[0], CSCALE * net[1]));
        f0.iFastLimit(PMF);
        return f0;
    };
}

// run one endgame episode with an MLP override; return TTE (frames) or maxFrames
function runEpisode(S, mlp, W, H, seed, startN, pumpSpeed, maxFrames, refreshMs) {
    S.setSimSeed(seed, refreshMs);
    S.NUM_BOIDS = startN;
    const sim = new S.Simulation('boids1');
    sim.canvasWidth = W; sim.canvasHeight = H;
    sim.initialize(false);
    S.setFrameMs(refreshMs);
    if (pumpSpeed > 0) {
        for (const b of sim.boids) {
            const ang = S.simRandom() * 2 * Math.PI;
            b.velocity.x = Math.cos(ang) * pumpSpeed;
            b.velocity.y = Math.sin(ang) * pumpSpeed;
        }
    }
    installResidual(sim, S, mlp);
    for (let f = 0; f < maxFrames; f++) {
        S.simTick(); sim.render();
        if (sim.boids.length === 0) return f + 1;
    }
    return maxFrames;
}

// full closed-loop game (startN boids) to extinction. If mlp given, gate it in
// when boids.length <= K (endgame); otherwise the deployed radial+chase drives.
function runFullGame(S, mlp, K, W, H, seed, startN, maxFrames, refreshMs) {
    S.setSimSeed(seed, refreshMs);
    S.NUM_BOIDS = startN;
    const sim = new S.Simulation('boids1');
    sim.canvasWidth = W; sim.canvasHeight = H;
    sim.initialize(false);
    S.setFrameMs(refreshMs);
    if (mlp) {
        const orig = sim.predator.getAutonomousForce.bind(sim.predator);
        const inp = new Float64Array(NIN);
        sim.predator.getAutonomousForce = function (boids) {
            if (boids.length === 0) return new S.Vector(0, 0);
            const f0 = orig(boids);
            if (boids.length > K) return f0; // early game: pure radial
            const p = sim.predator;
            const n = nearestRaw(p.position.x, p.position.y, boids);
            const b = boids[n.idx];
            inp[0] = n.dx / PSCALE; inp[1] = n.dy / PSCALE;
            inp[2] = b.velocity.x / VSCALE; inp[3] = b.velocity.y / VSCALE;
            inp[4] = p.velocity.x / VSCALE; inp[5] = p.velocity.y / VSCALE;
            const net = mlp(inp);
            f0.iAdd(new S.Vector(CSCALE * net[0], CSCALE * net[1]));
            f0.iFastLimit(PMF);
            return f0;
        };
    }
    for (let f = 0; f < maxFrames; f++) {
        S.simTick(); sim.render();
        if (sim.boids.length === 0) return f + 1;
    }
    return maxFrames;
}

// battery of episodes (common random numbers): returns array of {seed,startN,pump,W,H}
function makeBattery(seeds, seedBase) {
    const sizes = [[1440, 900], [1920, 1080]];
    const startNs = [1, 1, 2, 3]; // N=1 double-weighted (user's stated "last boid" case)
    const pumps = [0, 4];
    const battery = [];
    for (const [W, H] of sizes)
        for (const sN of startNs)
            for (const pump of pumps)
                for (let i = 0; i < seeds; i++)
                    battery.push({ W, H, startN: sN, pump, seed: seedBase + i });
    return battery;
}

// score one theta over a battery using sims cached per-size
function scoreTheta(theta, H, battery, maxFrames, refreshMs, simCache) {
    const mlp = makeMlp(theta, H);
    let sum = 0;
    for (const e of battery) {
        const key = e.W + 'x' + e.H;
        let S = simCache[key];
        if (!S) { S = loadSim(e.W, e.H); simCache[key] = S; }
        const tte = runEpisode(S, mlp, e.W, e.H, e.seed, e.startN, e.pump, maxFrames, refreshMs);
        sum += tte; // uncaught => maxFrames (heavy penalty)
    }
    return sum / battery.length;
}

// score the DEPLOYED radial policy (no override) over a battery — the bar to beat
function runEpisodeRadial(S, W, H, seed, startN, pumpSpeed, maxFrames, refreshMs) {
    S.setSimSeed(seed, refreshMs);
    S.NUM_BOIDS = startN;
    const sim = new S.Simulation('boids1');
    sim.canvasWidth = W; sim.canvasHeight = H;
    sim.initialize(false);
    S.setFrameMs(refreshMs);
    if (pumpSpeed > 0) {
        for (const b of sim.boids) {
            const ang = S.simRandom() * 2 * Math.PI;
            b.velocity.x = Math.cos(ang) * pumpSpeed;
            b.velocity.y = Math.sin(ang) * pumpSpeed;
        }
    }
    for (let f = 0; f < maxFrames; f++) {
        S.simTick(); sim.render();
        if (sim.boids.length === 0) return f + 1;
    }
    return maxFrames;
}

// ===================== worker: eval a chunk of candidates =====================
if (!isMainThread && workerData && workerData.mode === 'baseline') {
    const { battery, maxFrames, refreshMs } = workerData;
    const simCache = {};
    let sum = 0, caught = 0;
    for (const e of battery) {
        const key = e.W + 'x' + e.H;
        if (!simCache[key]) simCache[key] = loadSim(e.W, e.H);
        const tte = runEpisodeRadial(simCache[key], e.W, e.H, e.seed, e.startN, e.pump, maxFrames, refreshMs);
        sum += tte; if (tte < maxFrames) caught++;
    }
    parentPort.postMessage({ sum, caught, n: battery.length });
} else if (!isMainThread) {
    const { thetas, H, battery, maxFrames, refreshMs } = workerData;
    const simCache = {};
    const scores = thetas.map(t => scoreTheta(Float64Array.from(t), H, battery, maxFrames, refreshMs, simCache));
    parentPort.postMessage(scores);
} else {
    // ----------------- CEM driver -----------------
    const A = { H: 16, pop: 48, elite: 0.25, iters: 30, seeds: 12, workers: 8, maxFrames: 6000, sigma0: 0.6, refreshMs: 12, seedBase: 1000, islandSeed: 0, out: '' };
    for (let i = 2; i < process.argv.length; i++) {
        const a = process.argv[i];
        const m = { '--H': 'H', '--pop': 'pop', '--elite': 'elite', '--iters': 'iters', '--seeds': 'seeds', '--workers': 'workers', '--maxFrames': 'maxFrames', '--sigma0': 'sigma0', '--seedBase': 'seedBase', '--islandSeed': 'islandSeed' };
        if (m[a]) A[m[a]] = +process.argv[++i];
        else if (a === '--out') A.out = process.argv[++i];
        else if (a === '--baseline') A.baseline = true;
        else if (a === '--verify') A.verify = process.argv[++i];
        else if (a === '--gamefull') A.gamefull = process.argv[++i];
        else if (a === '--K') A.K = +process.argv[++i];
        else if (a === '--startN') A.startN = +process.argv[++i];
        else if (a === '--size') A.size = process.argv[++i];
    }
    const H = A.H, D = thetaLen(H);
    const battery = makeBattery(A.seeds, A.seedBase);

    if (A.baseline) {
        const nw = Math.min(A.workers, battery.length);
        const chunk = Math.ceil(battery.length / nw);
        const chunks = [];
        for (let i = 0; i < battery.length; i += chunk) chunks.push(battery.slice(i, i + chunk));
        Promise.all(chunks.map(cs => new Promise((res, rej) => {
            const w = new Worker(__filename, { workerData: { mode: 'baseline', battery: cs, maxFrames: A.maxFrames, refreshMs: A.refreshMs } });
            w.on('message', res); w.on('error', rej); w.on('exit', c => { if (c !== 0) rej(new Error('worker ' + c)); });
        }))).then(all => {
            const sum = all.reduce((a, b) => a + b.sum, 0), caught = all.reduce((a, b) => a + b.caught, 0), n = all.reduce((a, b) => a + b.n, 0);
            console.log(`[baseline radial] battery=${n} meanScore=${(sum / n).toFixed(1)} caught=${caught}/${n} (maxFrames=${A.maxFrames})`);
        }).catch(e => { console.error(e); process.exit(1); });
        return;
    }

    if (A.gamefull) {
        // full closed-loop games (startN boids) to extinction: gated champion vs radial-only.
        const ck = JSON.parse(fs.readFileSync(A.gamefull, 'utf8'));
        const mlp = makeMlp(Float64Array.from(ck.bestTheta), ck.H);
        const [W, H] = (A.size || '1920x1080').split('x').map(Number);
        const startN = A.startN || 120, K = A.K || 3, mf = A.maxFrames;
        const S = loadSim(W, H);
        let gSum = 0, gC = 0, rSum = 0, rC = 0;
        for (let i = 0; i < A.seeds; i++) {
            const seed = A.seedBase + i;
            const gt = runFullGame(S, mlp, K, W, H, seed, startN, mf, A.refreshMs);
            const rt = runFullGame(S, null, K, W, H, seed, startN, mf, A.refreshMs);
            gSum += gt; gC += (gt < mf ? 1 : 0); rSum += rt; rC += (rt < mf ? 1 : 0);
            console.log(`  seed=${seed} gated=${gt}${gt < mf ? '' : '(maxF)'} radial=${rt}${rt < mf ? '' : '(maxF)'}`);
        }
        console.log(`[gamefull] startN=${startN} K=${K} ${W}x${H} seeds=${A.seeds} maxF=${mf}`);
        console.log(`  GATED  meanTTE=${(gSum / A.seeds).toFixed(0)} extinct=${gC}/${A.seeds}`);
        console.log(`  RADIAL meanTTE=${(rSum / A.seeds).toFixed(0)} extinct=${rC}/${A.seeds}`);
        return;
    }

    if (A.verify) {
        // JS-verify a CEM champion on FRESH seeds vs the radial baseline, per regime.
        const ck = JSON.parse(fs.readFileSync(A.verify, 'utf8'));
        const Hv = ck.H, theta = Float64Array.from(ck.bestTheta);
        const mlp = makeMlp(theta, Hv);
        const simCache = {};
        const cells = {}; // key startN|pump -> {mTte,mCt,rTte,rCt,n}
        for (const e of battery) {
            const key = e.W + 'x' + e.H;
            if (!simCache[key]) simCache[key] = loadSim(e.W, e.H);
            const S = simCache[key];
            const mt = runEpisode(S, mlp, e.W, e.H, e.seed, e.startN, e.pump, A.maxFrames, A.refreshMs);
            const rt = runEpisodeRadial(S, e.W, e.H, e.seed, e.startN, e.pump, A.maxFrames, A.refreshMs);
            const ck2 = e.startN + '|' + e.pump;
            const c = cells[ck2] || (cells[ck2] = { mTte: 0, mCt: 0, rTte: 0, rCt: 0, n: 0 });
            c.mTte += mt; c.mCt += (mt < A.maxFrames ? 1 : 0); c.rTte += rt; c.rCt += (rt < A.maxFrames ? 1 : 0); c.n++;
        }
        console.log(`[verify] ${A.verify}  H=${Hv} seeds=${A.seeds} seedBase=${A.seedBase} maxFrames=${A.maxFrames}`);
        console.log(`  startN pump |  champ TTE  catch |  radial TTE  catch`);
        let mAll = 0, rAll = 0, nAll = 0, mC = 0, rC = 0;
        for (const k of Object.keys(cells).sort()) {
            const c = cells[k], [sN, pm] = k.split('|');
            console.log(`     N=${sN}  s=${pm} |  ${(c.mTte / c.n).toFixed(0).padStart(8)}  ${c.mCt}/${c.n} |  ${(c.rTte / c.n).toFixed(0).padStart(9)}  ${c.rCt}/${c.n}`);
            mAll += c.mTte; rAll += c.rTte; nAll += c.n; mC += c.mCt; rC += c.rCt;
        }
        console.log(`  -------- OVERALL champ meanTTE=${(mAll / nAll).toFixed(0)} catch=${mC}/${nAll} | radial meanTTE=${(rAll / nAll).toFixed(0)} catch=${rC}/${nAll}`);
        return;
    }
    console.log(`[cem] D=${D} H=${H} pop=${A.pop} iters=${A.iters} battery=${battery.length} maxFrames=${A.maxFrames} island=${A.islandSeed}`);

    // deterministic RNG for CEM sampling (island-specific)
    let rs = (A.islandSeed * 2654435761 + 12345) >>> 0;
    function rnd() { rs = (rs * 1664525 + 1013904223) >>> 0; return rs / 4294967296; }
    function gauss() { let u = 0, v = 0; while (u === 0) u = rnd(); while (v === 0) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

    const mean = new Float64Array(D); // start at 0 (zero policy)
    const std = new Float64Array(D).fill(A.sigma0);
    const nElite = Math.max(2, Math.round(A.pop * A.elite));

    function evalPop(thetas) {
        const nw = Math.min(A.workers, thetas.length);
        const chunk = Math.ceil(thetas.length / nw);
        const chunks = [];
        for (let i = 0; i < thetas.length; i += chunk) chunks.push(thetas.slice(i, i + chunk));
        return Promise.all(chunks.map(cs => new Promise((res, rej) => {
            const w = new Worker(__filename, { workerData: { thetas: cs.map(t => Array.from(t)), H, battery, maxFrames: A.maxFrames, refreshMs: A.refreshMs } });
            w.on('message', res); w.on('error', rej); w.on('exit', c => { if (c !== 0) rej(new Error('worker ' + c)); });
        }))).then(all => [].concat(...all));
    }

    (async () => {
        let bestScore = Infinity, bestTheta = Float64Array.from(mean);
        const t0 = Date.now();
        for (let it = 0; it < A.iters; it++) {
            const thetas = [];
            for (let p = 0; p < A.pop; p++) {
                const t = new Float64Array(D);
                for (let d = 0; d < D; d++) t[d] = mean[d] + std[d] * gauss();
                thetas.push(t);
            }
            // always include current mean as an anchor
            thetas[0] = Float64Array.from(mean);
            const scores = await evalPop(thetas);
            const idx = scores.map((s, i) => [s, i]).sort((a, b) => a[0] - b[0]);
            // refit on elites
            const newMean = new Float64Array(D), newStd = new Float64Array(D);
            for (let e = 0; e < nElite; e++) { const t = thetas[idx[e][1]]; for (let d = 0; d < D; d++) newMean[d] += t[d]; }
            for (let d = 0; d < D; d++) newMean[d] /= nElite;
            for (let e = 0; e < nElite; e++) { const t = thetas[idx[e][1]]; for (let d = 0; d < D; d++) { const dv = t[d] - newMean[d]; newStd[d] += dv * dv; } }
            for (let d = 0; d < D; d++) newStd[d] = Math.sqrt(newStd[d] / nElite) + 1e-3;
            for (let d = 0; d < D; d++) { mean[d] = newMean[d]; std[d] = newStd[d]; }
            if (idx[0][0] < bestScore) { bestScore = idx[0][0]; bestTheta = Float64Array.from(thetas[idx[0][1]]); }
            const eMean = idx.slice(0, nElite).reduce((a, b) => a + b[0], 0) / nElite;
            console.log(`[cem] it=${it} best=${idx[0][0].toFixed(0)} eliteMean=${eMean.toFixed(0)} globalBest=${bestScore.toFixed(0)} sigma=${(std.reduce((a, b) => a + b, 0) / D).toFixed(3)} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
            if (A.out) fs.writeFileSync(A.out, JSON.stringify({ H, D, bestScore, bestTheta: Array.from(bestTheta), mean: Array.from(mean), iter: it, args: A }, null, 0));
        }
        console.log(`[cem] DONE globalBest=${bestScore.toFixed(1)} -> ${A.out || '(no out)'}`);
    })();
}
