// Theoretical floor for the 1-boid endgame TTC.
//
// A lone boid flies a straight line on the torus at its current velocity (no
// flock force; no flee while the predator is far). The ABSOLUTE-fastest a
// point predator of speed vP can catch it is the earliest time t at which the
// predator can stand where the boid will be: torus_dist(pred0, boid0 + t*vB) <=
// vP * t  (min-image over the boid's torus period). This ignores (a) the
// predator's bounded turn-rate (max force) and (b) the boid's flee evasion —
// both of which only make the REAL TTC larger. So mean(t*) is a hard LOWER
// BOUND on any policy's endgame TTC. Comparing it to TERI's measured TTC tells
// us how much head-room is left (and whether "4x faster endgame" is geometric-
// ally possible at all).
//
//   node dev/endgame_floor.js --W 390 --H 844 --speed 6 --seeds 256
'use strict';

// mulberry32 matching js/rng.js simRandom seeding used by the evals
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function parse(argv) {
  const a = { W: 390, H: 844, speed: 6, seeds: 256, seedStart: 300000,
              vP: 2.5, border: 10, Tmax: 4000 };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    if (k === '--W') a.W = +argv[++i];
    else if (k === '--H') a.H = +argv[++i];
    else if (k === '--speed') a.speed = +argv[++i];
    else if (k === '--seeds') a.seeds = +argv[++i];
    else if (k === '--seedStart') a.seedStart = +argv[++i];
    else if (k === '--vP') a.vP = +argv[++i];
    else if (k === '--Tmax') a.Tmax = +argv[++i];
  }
  return a;
}

function minimg(d, P) { return d - P * Math.round(d / P); }

function main() {
  const o = parse(process.argv);
  const Px = o.W + 2 * o.border, Py = o.H + 2 * o.border;
  const pcx = o.W / 2, pcy = o.H / 2;          // predator starts at center
  const reach = 24;                             // catch reach (triangle SAT ~24px)
  const tstars = [];
  for (let s = 0; s < o.seeds; s++) {
    // match endgame_fasteval's RNG draw order as closely as possible: the boid
    // position is drawn from simRandom() after sim init. We approximate the
    // scatter distribution (uniform on screen) + random heading.
    const rnd = mulberry32((o.seedStart + s) ^ 0x9e3779b9);
    const bx = rnd() * o.W, by = rnd() * o.H;
    const ang = rnd() * Math.PI * 2;
    const vbx = Math.cos(ang) * o.speed, vby = Math.sin(ang) * o.speed;
    // scan t for earliest reachable (sub-frame resolution near the root)
    let tstar = o.Tmax;
    for (let t = 1; t <= o.Tmax; t++) {
      const fx = bx + vbx * t, fy = by + vby * t;
      const dx = minimg(fx - pcx, Px), dy = minimg(fy - pcy, Py);
      const dist = Math.hypot(dx, dy);
      if (dist - reach <= o.vP * t) { tstar = t; break; }
    }
    tstars.push(tstar);
  }
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const se = a => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / Math.max(1, a.length - 1)) / Math.sqrt(a.length); };
  const sorted = [...tstars].sort((a, b) => a - b);
  const pct = p => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  console.log(JSON.stringify({
    W: o.W, H: o.H, speed: o.speed, vP: o.vP, seeds: o.seeds,
    floorTTC_mean: +mean(tstars).toFixed(1), se: +se(tstars).toFixed(1),
    median: pct(0.5), p90: pct(0.9), max: sorted[sorted.length - 1]
  }));
}
main();
