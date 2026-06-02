// Parity + latency check for the flat planner rollout.
// Grabs a live snapshot from the running sim, then asks the worker to run BOTH
// rolloutFlat and rolloutRef (the real imported Boid class) on every candidate
// and report the max catch-count difference and the timing of each.
//   node dev/verify_planner_parity.js
'use strict';
const fs = require('fs'), path = require('path'), http = require('http'), url = require('url');
const { chromium } = require('playwright');

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
    const port = 8773;
    const server = await startServer(port, '/workspace');
    const browser = await chromium.launch();
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(String(e)));
    page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
    await page.goto(`http://localhost:${port}/index.html?policy=planner`, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__sim != null, null, { timeout: 8000 }).catch(() => {});
    await page.waitForTimeout(4000);   // let boids spread + predator feed a few times

    // Run 3 self-tests at different moments so we cover varied scenes.
    const results = [];
    for (let r = 0; r < 3; r++) {
        const res = await page.evaluate(() => new Promise(resolve => {
            const sim = window.__sim, pred = sim.predator, boids = sim.boids, n = boids.length;
            const bx = new Float64Array(n), by = new Float64Array(n), bvx = new Float64Array(n), bvy = new Float64Array(n);
            for (let i = 0; i < n; i++) { bx[i] = boids[i].position.x; by[i] = boids[i].position.y; bvx[i] = boids[i].velocity.x; bvy[i] = boids[i].velocity.y; }
            const w = new Worker('js/predator_planner_worker.js');
            w.postMessage({ type: 'config', K: 16, H: 120, POLICY_R: (window.__predatorModel && window.__predatorModel.POLICY_R) || 80, W: sim.canvasWidth, Hc: sim.canvasHeight, predRange: (typeof PREDATOR_RANGE !== 'undefined' ? PREDATOR_RANGE : 80) });
            w.onmessage = (e) => { w.terminate(); resolve(e.data); };
            w.postMessage({ type: 'selftest', snapshot: { bx, by, bvx, bvy, px: pred.position.x, py: pred.position.y, pvx: pred.velocity.x, pvy: pred.velocity.y, psize: pred.currentSize, lastFeed: pred.lastFeedTime, nowMs: (typeof simNow === 'function' ? simNow() : Date.now()) } });
        }));
        results.push(res);
        await page.waitForTimeout(3000);
    }
    await browser.close();
    server.close();

    let fail = false;
    console.log('=== flat-vs-ref parity + latency (K=16 candidates, H=120) ===');
    results.forEach((res, i) => {
        const flatPer = (res.tFlatMs / res.rows.length).toFixed(1);
        const refPer = (res.tRefMs / res.rows.length).toFixed(1);
        console.log(`run ${i}: maxCatchDiff=${res.maxCatchDiff}  flat=${res.tFlatMs}ms (${flatPer}/cand)  ref=${res.tRefMs}ms (${refPer}/cand)  speedup=${(res.tRefMs / Math.max(1, res.tFlatMs)).toFixed(1)}x`);
        console.log('   per-cand [flat,ref]:', JSON.stringify(res.rows));
        if (res.maxCatchDiff > 0) { fail = true; }
    });
    console.log('errors:', errs.slice(0, 5));
    console.log('\n' + (fail ? 'RESULT: FAIL (flat diverges from real Boid class)' : 'RESULT: PASS (flat == real Boid class on all candidates)'));
    process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
