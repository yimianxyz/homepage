// One-off: verify the planner-preview deploy works when served from the
// origin/main base (worktree at /tmp/planner-deploy). Confirms (A) default
// path has no __planner and still hunts, (B) ?policy=planner boots the worker,
// hunts, and logs zero errors. Playwright resolves from /workspace/node_modules.
'use strict';
const fs = require('fs'), path = require('path'), http = require('http'), url = require('url');
const { chromium } = require('playwright');
const ROOT = '/tmp/planner-deploy';

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

async function check(browser, base, suffix, secs) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(String(e)));
    page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
    await page.goto(base + suffix, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__sim != null, null, { timeout: 8000 }).catch(() => {});
    await page.waitForTimeout(secs * 1000);
    const st = await page.evaluate(() => ({
        planner: !!(window.__planner && window.__planner.active),
        eaten: window.__sim ? window.__sim.boidsEaten : null,
    }));
    await ctx.close();
    return { ...st, errs };
}

(async () => {
    const port = 8779;
    const server = await startServer(port, ROOT);
    const base = `http://localhost:${port}/index.html`;
    const browser = await chromium.launch();
    const a = await check(browser, base, '', 8);
    const b = await check(browser, base, '?policy=planner', 8);
    await browser.close();
    server.close();
    console.log('=== served from origin/main base + planner deploy ===');
    console.log(`(A) default       : __planner=${a.planner} (expect false)  eaten=${a.eaten}  errors=${a.errs.length} ${JSON.stringify(a.errs.slice(0,3))}`);
    console.log(`(B) ?policy=planner: __planner=${b.planner} (expect true)   eaten=${b.eaten}  errors=${b.errs.length} ${JSON.stringify(b.errs.slice(0,3))}`);
    const pass = a.planner === false && a.errs.length === 0 && b.planner === true && b.errs.length === 0 && b.eaten > 0;
    console.log(`\nRESULT: ${pass ? 'PASS' : 'FAIL'}`);
    process.exit(pass ? 0 : 1);
})().catch(e => { console.error(e); process.exit(1); });
