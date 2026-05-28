// Headless smoke test of the LIVE page after the weighted_predicted patrol
// change. Serves the filesystem source, loads it in Chromium, runs the real
// simulation (real predator.js getAutonomousForce + shipped NN), and reports
// console errors + catches (window.__sim.boidsEaten) over a wall-clock window.
'use strict';
const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '..');
const PORT = 8771;
const RUN_MS = 45000;

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };

function serve() {
    return http.createServer((req, res) => {
        let p = decodeURIComponent(req.url.split('?')[0]);
        if (p === '/') p = '/index.html';
        const fp = path.join(ROOT, p);
        if (!fp.startsWith(ROOT) || !fs.existsSync(fp)) { res.writeHead(404); res.end(); return; }
        res.writeHead(200, { 'Content-Type': MIME[path.extname(fp)] || 'application/octet-stream' });
        fs.createReadStream(fp).pipe(res);
    }).listen(PORT);
}

(async () => {
    const server = serve();
    const browser = await chromium.launch();
    const page = await browser.newPage();
    const errors = [];
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));

    await page.goto(`http://localhost:${PORT}/index.html`, { waitUntil: 'networkidle' });
    // wait for sim + NN to boot
    await page.waitForFunction(() => window.__sim && window.__predatorModel, { timeout: 15000 });
    const startEaten = await page.evaluate(() => window.__sim.boidsEaten || 0);
    const startBoids = await page.evaluate(() => window.__sim.boids.length);

    await page.waitForTimeout(RUN_MS);

    const endEaten = await page.evaluate(() => window.__sim.boidsEaten || 0);
    const endBoids = await page.evaluate(() => window.__sim.boids.length);
    const autoTarget = await page.evaluate(() => ({ x: window.__sim.predator.autonomousTarget.x, y: window.__sim.predator.autonomousTarget.y }));

    console.log(JSON.stringify({
        run_ms: RUN_MS,
        catches_in_window: endEaten - startEaten,
        boids_start: startBoids, boids_end: endBoids,
        autoTarget_sample: autoTarget,
        console_errors: errors,
    }, null, 2));

    await browser.close();
    server.close();
    if (errors.length) { console.error('FAIL: console errors present'); process.exit(1); }
    if (endEaten - startEaten <= 0) { console.error('WARN: no catches in window'); process.exit(2); }
    console.log('OK: page runs, predator catching, no errors');
})();
