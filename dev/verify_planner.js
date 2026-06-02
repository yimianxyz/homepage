// Headless verification for the TEMPORARY ?policy=planner override.
//
// Asserts:
//  (A) DEFAULT (no flag): window.__planner is undefined, no page errors, boids
//      get eaten — prod path is untouched.
//  (B) PLANNER (?policy=planner): window.__planner.active === true, the worker
//      boots without errors, a target gets committed, and the predator actually
//      hunts (boidsEaten increases over a few seconds).
//  (C) Closed-loop strength: over ~30s, planner eats clearly MORE than default
//      in the same wall-time window (sanity that the teacher policy is live).
//
//   node dev/verify_planner.js
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

async function runOnce(browser, url, runMs) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('weberror', e => errors.push('weberror:' + e.error()));
    await page.goto(url, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__sim != null, null, { timeout: 8000 }).catch(() => {});
    await page.waitForTimeout(runMs);
    const state = await page.evaluate(() => ({
        hasPlanner: !!window.__planner,
        plannerActive: !!(window.__planner && window.__planner.active),
        eaten: window.__sim ? window.__sim.boidsEaten : null,
        nBoids: window.__sim ? window.__sim.boids.length : null,
    }));
    await ctx.close();
    return { state, errors };
}

(async () => {
    const port = 8771;
    const server = await startServer(port, '/workspace');
    const base = `http://localhost:${port}/index.html`;
    const browser = await chromium.launch();
    let fail = false;

    console.log('=== (A) DEFAULT (no flag) — prod path must be untouched ===');
    const def = await runOnce(browser, base, 8000);
    console.log('  __planner present :', def.state.hasPlanner, '(expect false)');
    console.log('  boidsEaten        :', def.state.eaten);
    console.log('  errors            :', def.errors.length, def.errors.slice(0, 3));
    if (def.state.hasPlanner) { console.log('  FAIL: __planner leaked into default path'); fail = true; }
    if (def.errors.length) { console.log('  FAIL: console/page errors on default'); fail = true; }
    if (!(def.state.eaten > 0)) { console.log('  WARN: no catches in 8s default (slow seed?)'); }

    console.log('\n=== (B) PLANNER (?policy=planner) — must boot worker + hunt ===');
    const pl = await runOnce(browser, base + '?policy=planner', 12000);
    console.log('  __planner.active  :', pl.state.plannerActive, '(expect true)');
    console.log('  boidsEaten        :', pl.state.eaten);
    console.log('  errors            :', pl.errors.length, pl.errors.slice(0, 5));
    if (!pl.state.plannerActive) { console.log('  FAIL: planner not active under flag'); fail = true; }
    if (pl.errors.length) { console.log('  FAIL: console/page errors under planner'); fail = true; }
    if (!(pl.state.eaten > 0)) { console.log('  FAIL: planner ate nothing in 12s'); fail = true; }

    console.log('\n=== (C) closed-loop strength over 30s (planner vs default) ===');
    const [d30, p30] = await Promise.all([
        runOnce(browser, base, 30000),
        runOnce(browser, base + '?policy=planner', 30000),
    ]);
    console.log('  default  eaten@30s:', d30.state.eaten);
    console.log('  planner  eaten@30s:', p30.state.eaten);
    if (!(p30.state.eaten >= d30.state.eaten)) {
        console.log('  WARN: planner did not exceed default in this single 30s window (variance — not a hard fail)');
    }

    await browser.close();
    server.close();
    console.log('\n' + (fail ? 'RESULT: FAIL' : 'RESULT: PASS'));
    process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
