'use strict';
const fs = require('fs');
const path = require('path');
const http = require('http');
const url = require('url');
const { chromium } = require('playwright');

const ROOT = '/workspace';
const PORT = 8771;
const OUT = process.argv[2] || '/tmp/e3d_brain.png';

function startServer(port, root) {
    return new Promise((resolve) => {
        const srv = http.createServer((req, res) => {
            const u = url.parse(req.url);
            let p = path.join(root, decodeURIComponent(u.pathname));
            if (p.endsWith('/')) p += 'index.html';
            fs.stat(p, (err, stat) => {
                if (err || !stat.isFile()) { res.statusCode = 404; return res.end(); }
                const ext = path.extname(p).toLowerCase();
                const types = { '.html': 'text/html', '.js': 'application/javascript',
                                '.css': 'text/css', '.json': 'application/json', '.png': 'image/png' };
                res.setHeader('Content-Type', types[ext] || 'application/octet-stream');
                fs.createReadStream(p).pipe(res);
            });
        });
        srv.listen(port, () => resolve(srv));
    });
}

(async () => {
    const srv = await startServer(PORT, ROOT);
    const browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1480, height: 760 }, deviceScaleFactor: 2 });
    const errs = [];
    page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
    page.on('pageerror', e => errs.push('PAGEERROR ' + e.message));
    await page.goto(`http://localhost:${PORT}/dev/e3d_viz/index.html`, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => window.__brainReady === true, { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2500); // let EMA normalization settle into a clean frame
    await page.screenshot({ path: OUT, fullPage: true });
    if (errs.length) console.log('PAGE ERRORS:\n' + errs.join('\n'));
    console.log('wrote ' + OUT);
    await browser.close();
    srv.close();
})();
