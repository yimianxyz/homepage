// End-to-end visual verification for the canvas-rendered page.
// Spins up a local http.server on 8765, then loads the page in headless
// Chromium across a matrix of viewports, waits for boids/predator/viz to
// boot, and dumps PNG screenshots to /tmp/snapshots/.
//
//   node dev/snapshots.js [--url <override>]
//
// Default URL is http://localhost:8765/index.html so we test the file
// system source, not whatever is shipped on prod. Pass --url to point at
// yimianxyz.github.io etc.

'use strict';

const fs = require('fs');
const path = require('path');
const http = require('http');
const url = require('url');
const { chromium, devices } = require('playwright');

const OUT_DIR = '/tmp/snapshots';

function parseArgs(argv) {
    const a = { url: 'http://localhost:8765/index.html', port: 8765 };
    for (let i = 2; i < argv.length; i++) {
        if (argv[i] === '--url') a.url = argv[++i];
        else if (argv[i] === '--port') a.port = +argv[++i];
    }
    return a;
}

// Tiny static file server rooted at /workspace.
function startServer(port, root) {
    return new Promise((resolve) => {
        const srv = http.createServer((req, res) => {
            const u = url.parse(req.url);
            let p = path.join(root, u.pathname);
            if (p.endsWith('/')) p += 'index.html';
            fs.stat(p, (err, stat) => {
                if (err || !stat.isFile()) { res.statusCode = 404; return res.end(); }
                const ext = path.extname(p).toLowerCase();
                const types = { '.html': 'text/html', '.js': 'application/javascript',
                                '.css': 'text/css', '.json': 'application/json',
                                '.png': 'image/png' };
                res.setHeader('Content-Type', types[ext] || 'application/octet-stream');
                res.setHeader('Access-Control-Allow-Origin', '*');
                fs.createReadStream(p).pipe(res);
            });
        });
        srv.listen(port, () => resolve(srv));
    });
}

// Viewport matrix — covers the main breakpoints + a few real devices, PLUS
// "with-chrome" variants that shrink the height to simulate iOS Safari's
// visible viewport when the URL bar + bottom toolbar are showing. Playwright
// doesn't render the OS chrome, so we approximate by reducing height by
// ~115px (typical iOS toolbar overhead).
const VIEWPORTS = [
    { label: 'desktop-1440x900',       width: 1440, height: 900,  deviceScale: 1 },
    { label: 'laptop-1280x800',        width: 1280, height: 800,  deviceScale: 1 },
    { label: 'tablet-iPad-768x1024',   width: 768,  height: 1024, deviceScale: 2 },
    { label: 'iPhone-13-portrait',     ...devices['iPhone 13'].viewport, deviceScale: devices['iPhone 13'].deviceScaleFactor, ua: devices['iPhone 13'].userAgent },
    { label: 'iPhone-13-with-chrome',  width: 390, height: 549, deviceScale: 3, ua: devices['iPhone 13'].userAgent },
    { label: 'iPhone-SE-portrait-375x667', width: 375, height: 667, deviceScale: 2, ua: devices['iPhone SE'].userAgent },
    { label: 'iPhone-SE-with-chrome',  width: 375, height: 553, deviceScale: 2, ua: devices['iPhone SE'].userAgent },
    { label: 'iPhone-SE-landscape-667x375', width: 667, height: 375, deviceScale: 2, ua: devices['iPhone SE'].userAgent },
    { label: 'Pixel-7-412x915',        width: 412, height: 915, deviceScale: 2.625, ua: devices['Pixel 7'].userAgent },
    { label: 'small-360x640',          width: 360, height: 640, deviceScale: 2, ua: devices['Galaxy S5'].userAgent },
];

async function main() {
    const args = parseArgs(process.argv);
    fs.mkdirSync(OUT_DIR, { recursive: true });

    let server = null;
    if (args.url.startsWith('http://localhost')) {
        server = await startServer(args.port, '/workspace');
        console.log('local server: http://localhost:' + args.port);
    }

    const browser = await chromium.launch();
    const results = [];

    for (const vp of VIEWPORTS) {
        const contextOpts = {
            viewport: { width: vp.width, height: vp.height },
            deviceScaleFactor: vp.deviceScale,
        };
        if (vp.ua) contextOpts.userAgent = vp.ua;
        if (vp.ua && vp.ua.match(/iPhone|Android|Pixel/i)) contextOpts.isMobile = true;
        if (vp.ua && vp.ua.match(/iPhone|Android|Pixel/i)) contextOpts.hasTouch = true;
        const ctx = await browser.newContext(contextOpts);
        const page = await ctx.newPage();

        // Capture console errors for diagnosis.
        const errors = [];
        page.on('pageerror', err => errors.push(String(err)));
        page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });

        await page.goto(args.url, { waitUntil: 'load' });
        // Wait for __predatorReady + a couple of animation frames.
        await page.waitForFunction(() => window.__predatorReady !== undefined, null, { timeout: 5000 }).catch(() => {});
        await page.waitForFunction(() => window.__predatorModel != null, null, { timeout: 5000 }).catch(() => {});
        await page.waitForTimeout(800);                                // let boids spread + viz EMA settle

        // Snapshot diagnostic state from the page.
        const diag = await page.evaluate(() => {
            const c = document.querySelector('#boids1');
            return {
                hasModel: !!window.__predatorModel,
                modelFeatureDim: window.__predatorModel && window.__predatorModel.featureDim,
                canvasAttrW: c && c.width,
                canvasAttrH: c && c.height,
                canvasClientW: c && c.clientWidth,
                canvasClientH: c && c.clientHeight,
                innerW: window.innerWidth,
                innerH: window.innerHeight,
                dpr: window.devicePixelRatio,
            };
        });

        const outPath = path.join(OUT_DIR, vp.label + '.png');
        await page.screenshot({ path: outPath, fullPage: false });
        results.push({ ...vp, diag, errors, outPath });
        await ctx.close();
        console.log(vp.label.padEnd(36) + ' canvas=' + diag.canvasAttrW + 'x' + diag.canvasAttrH +
                    ' inner=' + diag.innerW + 'x' + diag.innerH +
                    ' dpr=' + diag.dpr +
                    ' model=' + (diag.hasModel ? 'OK' : 'MISSING') +
                    (errors.length ? ' [' + errors.length + ' err]' : ''));
    }

    await browser.close();
    if (server) server.close();

    // Write summary
    fs.writeFileSync(path.join(OUT_DIR, '_summary.json'), JSON.stringify(results, null, 2));
    console.log('\nScreenshots saved to ' + OUT_DIR + '/');
    console.log('Summary at ' + OUT_DIR + '/_summary.json');
}

main().catch(e => { console.error(e); process.exit(1); });
