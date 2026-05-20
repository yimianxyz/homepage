// Verify the new click/tap-to-spawn-a-boid interaction end-to-end.
// Loads the page in headless Chromium, performs a battery of clicks and
// touches at specific coordinates, and asserts the boid count changes
// the expected way. Captures one screenshot ~150ms after spawn so we
// can see the ripple mid-animation.
//
//   node dev/interaction_test.js
//
// Exits non-zero if any check fails.

'use strict';
const fs = require('fs');
const path = require('path');
const http = require('http');
const url = require('url');
const { chromium, devices } = require('playwright');

const PORT = 8765;
const OUT_DIR = '/tmp/snapshots';

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
                fs.createReadStream(p).pipe(res);
            });
        });
        srv.listen(port, () => resolve(srv));
    });
}

async function setupPage(browser, viewport, isMobile) {
    const ctx = await browser.newContext({
        viewport,
        deviceScaleFactor: 2,
        isMobile,
        hasTouch: isMobile,
    });
    const page = await ctx.newPage();
    await page.goto('http://localhost:' + PORT + '/index.html', { waitUntil: 'load' });
    await page.waitForFunction(() => window.__sim != null, null, { timeout: 8000 });
    await page.waitForTimeout(300);  // let initial boids spread
    return { ctx, page };
}

async function boidCount(page) {
    return await page.evaluate(() => window.__sim.boids.length);
}

const checks = [];
function check(name, ok, detail) {
    checks.push({ name, ok, detail });
    console.log((ok ? 'PASS' : 'FAIL') + '  ' + name + (detail ? ' — ' + detail : ''));
}

(async () => {
    fs.mkdirSync(OUT_DIR, { recursive: true });
    const server = await startServer(PORT, '/workspace');
    const browser = await chromium.launch();

    // ===== Desktop checks =====
    {
        const { ctx, page } = await setupPage(browser, { width: 1280, height: 800 }, false);

        // 1. Single click on empty background spawns 1 boid.
        const before = await boidCount(page);
        await page.mouse.click(50, 50);                        // far top-left, no text there
        await page.waitForTimeout(150);
        await page.screenshot({ path: path.join(OUT_DIR, 'interact-desktop-after-click.png') });
        const after = await boidCount(page);
        check('desktop single-click spawns 1 boid', after === before + 1, 'before=' + before + ' after=' + after);

        // 2. Three sequential clicks add three boids.
        const b2 = await boidCount(page);
        await page.mouse.click(100, 100);
        await page.mouse.click(200, 200);
        await page.mouse.click(300, 300);
        await page.waitForTimeout(80);
        const a2 = await boidCount(page);
        check('desktop three clicks spawn 3 boids', a2 === b2 + 3, 'before=' + b2 + ' after=' + a2);

        // 3. Clicking a link does NOT spawn a boid. Use the GitHub link.
        //    Predator may eat boids between samples; for no-spawn checks we
        //    assert that the count did not INCREASE (the only thing a click
        //    could do).
        const b3 = await boidCount(page);
        const linkHandle = await page.locator('a[href*="github.com"]').first();
        await linkHandle.click({ modifiers: ['Control'] }).catch(() => {});
        await page.waitForTimeout(120);
        const a3 = await boidCount(page);
        check('desktop link click does NOT spawn boid', a3 <= b3, 'before=' + b3 + ' after=' + a3);

        // 4. Right-click does NOT spawn.
        const b4 = await boidCount(page);
        await page.mouse.click(60, 60, { button: 'right' });
        await page.waitForTimeout(80);
        const a4 = await boidCount(page);
        check('desktop right-click does NOT spawn', a4 <= b4, 'before=' + b4 + ' after=' + a4);

        // 5. Active text selection: drag over a paragraph, the mouseup should NOT
        //    spawn (the browser only fires `click` on a no-drag down→up; drag-select
        //    does not). If headless Chromium fires click anyway despite the drag,
        //    that's a tooling artifact, not a real-user bug.
        const b5 = await boidCount(page);
        const para = await page.locator('p').first();
        const box = await para.boundingBox();
        let selText = '';
        if (box) {
            await page.mouse.move(box.x + 5, box.y + box.height / 2);
            await page.mouse.down();
            await page.mouse.move(box.x + box.width * 0.8, box.y + box.height / 2, { steps: 8 });
            await page.mouse.up();
            await page.waitForTimeout(80);
            selText = await page.evaluate(() => window.getSelection().toString());
        }
        const a5 = await boidCount(page);
        check('desktop drag-to-select does NOT spawn', a5 <= b5,
              'before=' + b5 + ' after=' + a5 + ' selection=' + JSON.stringify(selText.slice(0, 40)));

        await ctx.close();
    }

    // ===== Mobile checks =====
    {
        const { ctx, page } = await setupPage(browser, { width: 390, height: 664 }, true);

        // 6. Single tap on background spawns 1 boid.
        const before = await boidCount(page);
        await page.touchscreen.tap(50, 600);                  // bottom-left, well below any text
        await page.waitForTimeout(150);
        await page.screenshot({ path: path.join(OUT_DIR, 'interact-mobile-after-tap.png') });
        const after = await boidCount(page);
        check('mobile single tap spawns 1 boid', after === before + 1, 'before=' + before + ' after=' + after);

        // 7. Three sequential taps add three boids.
        const b7 = await boidCount(page);
        await page.touchscreen.tap(60, 620);
        await page.touchscreen.tap(120, 600);
        await page.touchscreen.tap(180, 580);
        await page.waitForTimeout(120);
        const a7 = await boidCount(page);
        check('mobile three taps spawn 3 boids', a7 === b7 + 3, 'before=' + b7 + ' after=' + a7);

        // 8. Tap on a link does NOT spawn (and the link receives the tap).
        const b8 = await boidCount(page);
        const linkHandle = await page.locator('a[href*="github.com"]').first();
        const lbox = await linkHandle.boundingBox();
        if (lbox) {
            await page.touchscreen.tap(lbox.x + lbox.width / 2, lbox.y + lbox.height / 2);
        }
        await page.waitForTimeout(120);
        const a8 = await boidCount(page);
        check('mobile tap on link does NOT spawn', a8 <= b8, 'before=' + b8 + ' after=' + a8);

        await ctx.close();
    }

    await browser.close();
    server.close();

    const failed = checks.filter(c => !c.ok);
    console.log('\n' + checks.length + ' checks, ' + failed.length + ' failed');
    process.exit(failed.length === 0 ? 0 : 1);
})().catch(e => { console.error(e); process.exit(2); });
