// Zero-staleness closed-loop ceiling: drive the verified flat sim in-worker for
// a full episode, re-planning FRESH every D frames (no main-thread lag), for
// both controllers. If planner >> e3d here, the browser shortfall is purely
// real-time staleness; if planner ~= e3d, the port/policy itself is the issue.
//   node dev/verify_planner_ceiling.js [frames] [D] [seeds]
'use strict';
const fs = require('fs'), path = require('path'), http = require('http'), url = require('url');
const { chromium } = require('playwright');

const FRAMES = parseInt(process.argv[2], 10) || 3000;
const D = parseInt(process.argv[3], 10) || 8;
const SEEDS = parseInt(process.argv[4], 10) || 2;

function startServer(port, root) {
    return new Promise((resolve) => {
        const srv = http.createServer((req, res) => {
            let p = path.join(root, url.parse(req.url).pathname);
            if (p.endsWith('/')) p += 'index.html';
            fs.stat(p, (err, stat) => {
                if (err || !stat.isFile()) { res.statusCode = 404; return res.end(); }
                const ext = path.extname(p).toLowerCase();
                const types = { '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png' };
                res.setHeader('Content-Type', types[ext] || 'application/octet-stream');
                res.setHeader('Access-Control-Allow-Origin', '*');
                fs.createReadStream(p).pipe(res);
            });
        });
        srv.listen(port, () => resolve(srv));
    });
}

(async () => {
    const port = 8775;
    const server = await startServer(port, '/workspace');
    const browser = await chromium.launch();
    console.log(`frames=${FRAMES} D=${D} seeds=${SEEDS}\n`);
    let sumE = 0, sumP = 0;
    for (let r = 0; r < SEEDS; r++) {
        const seed = 9001 + r * 53;
        const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
        const page = await ctx.newPage();
        await page.addInitScript((s) => { Math.random = () => ((s % 1000) / 1000); Date.now = () => 1700000000000 + s; }, seed);
        const errs = [];
        page.on('pageerror', e => errs.push(String(e)));
        page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
        await page.goto(`http://localhost:${port}/index.html?policy=planner`, { waitUntil: 'load' });
        await page.waitForFunction(() => window.__sim != null, null, { timeout: 8000 }).catch(() => {});
        await page.waitForTimeout(400);   // near-initial spread

        const res = await page.evaluate(({ frames, D }) => new Promise(resolve => {
            const sim = window.__sim, pred = sim.predator, boids = sim.boids, n = boids.length;
            const bx = new Float64Array(n), by = new Float64Array(n), bvx = new Float64Array(n), bvy = new Float64Array(n);
            for (let i = 0; i < n; i++) { bx[i] = boids[i].position.x; by[i] = boids[i].position.y; bvx[i] = boids[i].velocity.x; bvy[i] = boids[i].velocity.y; }
            const snap = { bx, by, bvx, bvy, px: pred.position.x, py: pred.position.y, pvx: pred.velocity.x, pvy: pred.velocity.y, psize: pred.currentSize, lastFeed: pred.lastFeedTime, nowMs: (typeof simNow === 'function' ? simNow() : 0) };
            const w = new Worker('js/predator_planner_worker.js');
            w.postMessage({ type: 'config', K: 16, H: 120, POLICY_R: (window.__predatorModel && window.__predatorModel.POLICY_R) || 80, W: sim.canvasWidth, Hc: sim.canvasHeight, predRange: (typeof PREDATOR_RANGE !== 'undefined' ? PREDATOR_RANGE : 80) });
            const out = {};
            w.onmessage = (e) => {
                out[e.data.controller] = { catches: e.data.catches, aliveEnd: e.data.aliveEnd, ms: e.data.ms };
                if (out.e3d && out.planner) { w.terminate(); resolve({ nStart: n, ...out }); }
            };
            w.postMessage({ type: 'eval', controller: 'e3d', frames, D, snapshot: snap });
            w.postMessage({ type: 'eval', controller: 'planner', frames, D, snapshot: snap });
        }), { frames: FRAMES, D });

        await ctx.close();
        sumE += res.e3d.catches; sumP += res.planner.catches;
        console.log(`seed ${seed} (n0=${res.nStart}): e3d=${res.e3d.catches} (alive ${res.e3d.aliveEnd}, ${res.e3d.ms}ms)  |  planner=${res.planner.catches} (alive ${res.planner.aliveEnd}, ${res.planner.ms}ms)  | x${(res.planner.catches / Math.max(1, res.e3d.catches)).toFixed(2)} | err:${errs.length}`);
    }
    await browser.close();
    server.close();
    console.log(`\nmean over ${SEEDS} seeds: e3d=${(sumE / SEEDS).toFixed(1)}  planner=${(sumP / SEEDS).toFixed(1)}  ratio=${(sumP / Math.max(1, sumE)).toFixed(2)}x`);
})().catch(e => { console.error(e); process.exit(1); });
