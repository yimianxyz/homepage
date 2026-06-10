// Ground-truth per-device eval of the SHIPPED production policy, in a REAL
// browser (Playwright/Chromium → JIT-fast, unlike the vm-sandbox harness).
//
// Loads index.html at a device-realistic viewport. Because the page derives
// NUM_BOIDS / PREDATOR_RANGE / isMobileDevice from window.innerWidth at script
// load, the right viewport reproduces the device's game exactly (boid count,
// avoidance range, canvas size = full screen). We then build a FRESH Simulation
// inside the page and step it manually (simTick; tick; render) — the same
// two-pass loop boids.js runs — so we can run faster than real time and read
// boidsEaten. This is the Tier-B ground truth the GPU search is calibrated to.
//
//   NODE_PATH=/workspace/node_modules node dev/eval_device_browser.js \
//     --W 390 --H 844 --seedStart 200000 --seeds 8 --frames 1500
'use strict';
const path = require('path');
const { chromium } = require('playwright');

function parseArgs(argv) {
    const a = { W: 390, H: 844, seedStart: 200000, seeds: 8, frames: 1500,
        url: 'http://localhost:8099/index.html' };
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (k === '--W') a.W = +argv[++i];
        else if (k === '--H') a.H = +argv[++i];
        else if (k === '--seedStart') a.seedStart = +argv[++i];
        else if (k === '--seeds') a.seeds = +argv[++i];
        else if (k === '--frames') a.frames = +argv[++i];
        else if (k === '--url') a.url = argv[++i];
    }
    return a;
}

async function main() {
    const opt = parseArgs(process.argv);
    const browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: opt.W, height: opt.H }, deviceScaleFactor: 1 });
    page.on('pageerror', e => console.error('PAGEERR', e.message));
    // Neuter the page's own auto-run sim: boids.js starts window.__sim on a
    // setInterval that, firing during our manual loop, would advance the SHARED
    // mulberry32 RNG and make our seeded episodes non-deterministic. Stub
    // setInterval so run() schedules nothing; we drive frames ourselves.
    await page.addInitScript(() => { window.setInterval = function () { return 0; }; });
    await page.goto(opt.url, { waitUntil: 'load' });
    // Wait for the value net to load (the page boot gate).
    await page.waitForFunction(() => window.__predatorReady !== undefined, null, { timeout: 20000 });
    await page.evaluate(() => window.__predatorReady);

    const meta = await page.evaluate(() => ({
        innerW: window.innerWidth, innerH: window.innerHeight,
        numBoids: typeof NUM_BOIDS !== 'undefined' ? NUM_BOIDS : null,
        predRange: typeof PREDATOR_RANGE !== 'undefined' ? PREDATOR_RANGE : null,
        mobile: typeof isMobileDevice === 'function' ? isMobileDevice() : null,
        canvasW: (document.getElementById('boids1') || {}).width,
        canvasH: (document.getElementById('boids1') || {}).height,
    }));

    const res = await page.evaluate(async (opt) => {
        const out = [];
        for (let i = 0; i < opt.seeds; i++) {
            const seed = opt.seedStart + i;
            setSimSeed(seed, 12);
            const sim = new Simulation('boids1');
            // canvas dims come from the real canvas element (full screen)
            const cv = document.getElementById('boids1');
            sim.canvasWidth = cv.width; sim.canvasHeight = cv.height;
            sim.initialize(false);
            if (typeof setFrameMs === 'function') setFrameMs(12);
            sim.tick();   // one-time pre-loop tick() matching the browser's run()
            for (let f = 0; f < opt.frames; f++) { simTick(); sim.tick(); sim.render(); }
            out.push(sim.boidsEaten);
        }
        return out;
    }, opt);

    await browser.close();
    const sum = res.reduce((a, b) => a + b, 0);
    const mean = sum / res.length;
    const sd = Math.sqrt(res.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, res.length - 1));
    const se = sd / Math.sqrt(res.length);
    console.log(JSON.stringify({ W: opt.W, H: opt.H, seedStart: opt.seedStart, seeds: opt.seeds,
        frames: opt.frames, meta, mean, se, per: res }));
}
main().catch(e => { console.error(e); process.exit(1); });
