// Deterministic closed-loop head-to-head: production radial net vs the planner
// override, started from an IDENTICAL initial sim (we pin the bootstrap seed by
// freezing Date.now + Math.random before load). We run each in real-time for a
// fixed wall window and report catches normalised per 1000 sim frames, so the
// comparison is independent of the exact frame rate.
//   node dev/verify_planner_loop.js [windowMs] [repeats]
'use strict';
const fs = require('fs'), path = require('path'), http = require('http'), url = require('url');
const { chromium } = require('playwright');

const WIN = parseInt(process.argv[2], 10) || 40000;
const REP = parseInt(process.argv[3], 10) || 2;

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

async function run(browser, url, seed, winMs) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    // Pin the bootstrap seed: rng.js derives it from Date.now() ^ Math.random().
    await page.addInitScript((s) => {
        Math.random = () => ((s % 1000) / 1000);
        Date.now = () => 1700000000000 + s;
    }, seed);
    const errs = [];
    page.on('pageerror', e => errs.push(String(e)));
    page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
    await page.goto(url, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__sim != null, null, { timeout: 8000 }).catch(() => {});
    await page.waitForTimeout(winMs);
    const st = await page.evaluate(() => ({
        eaten: window.__sim ? window.__sim.boidsEaten : null,
        frame: (typeof getSimFrame === 'function') ? getSimFrame() : null,
        nBoids: window.__sim ? window.__sim.boids.length : null,
    }));
    await ctx.close();
    return { ...st, errs };
}

(async () => {
    const port = 8774;
    const server = await startServer(port, '/workspace');
    const base = `http://localhost:${port}/index.html`;
    const browser = await chromium.launch();
    console.log(`window=${WIN}ms repeats=${REP}  (catches per 1000 sim frames)\n`);
    let sumD = 0, sumP = 0;
    for (let r = 0; r < REP; r++) {
        const seed = 12345 + r * 777;
        const d = await run(browser, base, seed, WIN);
        const p = await run(browser, base + '?policy=planner', seed, WIN);
        const dr = d.eaten / (d.frame / 1000), pr = p.eaten / (p.frame / 1000);
        sumD += dr; sumP += pr;
        console.log(`seed ${seed}: prod eaten=${d.eaten}@${d.frame}f (${dr.toFixed(2)}/1k)  |  planner eaten=${p.eaten}@${p.frame}f (${pr.toFixed(2)}/1k)  | err p:${p.errs.length} d:${d.errs.length}`);
        if (p.errs.length) console.log('   planner errors:', p.errs.slice(0, 3));
    }
    await browser.close();
    server.close();
    console.log(`\nmean: prod ${(sumD / REP).toFixed(2)}/1k   planner ${(sumP / REP).toFixed(2)}/1k`);
})().catch(e => { console.error(e); process.exit(1); });
