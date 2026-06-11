// Faithful 1-boid endgame sandbox. Ports the EXACT dynamics from js/ (boid flee +
// update + wrap @border10, predator seek + update + wrap @border20, triangle-SAT
// catch) so we can iterate interceptor designs locally (1 boid = microseconds) and
// measure the REALISTIC floor (best achievable given the predator's bounded turn
// rate + the boid's flee), then port the winner into predator_cheap.js.
//
//   node dev/endgame_lab.js --W 390 --H 844 --speed 6 --seeds 512 --pol lead
'use strict';

const MAX_SPEED = 6, render_size = 10, PREDATOR_TURN_FACTOR = 0.3, BMAX_FORCE = 0.1;
const PREDATOR_MAX_SPEED = 2.5, PREDATOR_MAX_FORCE = 0.05, PREDATOR_SIZE = 12;
const B_BORDER = 10, P_BORDER = 20, PREDATOR_RANGE = 80, EPSILON = 1e-7;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function limit(vx, vy, max) { const m = Math.hypot(vx, vy); return m > max ? [vx / m * max, vy / m * max] : [vx, vy]; }
function catchTriangle(cx, cy, vx, vy, fwd, half) {
  const m = Math.hypot(vx, vy) || 1, ux = vx / m, uy = vy / m;
  return [cx + ux * fwd, cy + uy * fwd, cx - uy * half, cy + ux * half, cx + uy * half, cy - ux * half];
}
function axisSep(A, C) {
  for (let i = 0; i < 6; i += 2) {
    const nx = A[i + 1] - A[(i + 3) % 6], ny = A[(i + 2) % 6] - A[i];
    let a0 = Infinity, a1 = -Infinity, c0 = Infinity, c1 = -Infinity, d;
    for (let j = 0; j < 6; j += 2) { d = A[j] * nx + A[j + 1] * ny; if (d < a0) a0 = d; if (d > a1) a1 = d; }
    for (let k = 0; k < 6; k += 2) { d = C[k] * nx + C[k + 1] * ny; if (d < c0) c0 = d; if (d > c1) c1 = d; }
    if (a1 < c0 || c1 < a0) return true;
  }
  return false;
}
function catches(px, py, pvx, pvy, psize, bx, by, bvx, bvy) {
  const reach = psize * 1.2 + render_size, dx = bx - px, dy = by - py;
  if (dx * dx + dy * dy > reach * reach) return false;
  const P = catchTriangle(px, py, pvx, pvy, psize * 1.2, psize * 0.3);
  const B = catchTriangle(bx, by, bvx, bvy, render_size, render_size / 3);
  return !axisSep(P, B) && !axisSep(B, P);
}
function minimg(d, P) { return d - P * Math.round(d / P); }

// ---- interceptor policies: given world state, return a desired AIM vector (dx,dy
//      a direction to seek at full speed). Persistent state in `st`. ----
function earliestReach(W, H, px, py, bx, by, bvx, bvy, slack, tmax, dt) {
  const Px = W + 2 * B_BORDER, Py = H + 2 * B_BORDER, vP = PREDATOR_MAX_SPEED;
  for (let t = dt; t <= tmax; t += dt) {
    const fx = bx + bvx * t, fy = by + bvy * t;
    const dx = minimg(fx - px, Px), dy = minimg(fy - py, Py);
    if (Math.hypot(dx, dy) - render_size <= vP * t * slack) return [dx, dy, t];
  }
  // fallback: perpendicular cutoff onto the line
  const bs = Math.hypot(bvx, bvy) || 1e-6, ux = bvx / bs, uy = bvy / bs;
  const cdx = minimg(bx - px, Px), cdy = minimg(by - py, Py);
  const along = cdx * ux + cdy * uy;
  return [cdx - along * ux, cdy - along * uy, tmax + 1];
}

const POLICIES = {
  // aim straight at the boid (min-image) — naive tail chase
  pure(s, st) { const Px = s.W + 2 * B_BORDER, Py = s.H + 2 * B_BORDER; return [minimg(s.bx - s.px, Px), minimg(s.by - s.py, Py)]; },
  // re-aim every frame at the earliest-reachable intercept point on the straight track
  lead(s, st) { const r = earliestReach(s.W, s.H, s.px, s.py, s.bx, s.by, s.bvx, s.bvy, st.slack ?? 1.0, 1400, 4); return [r[0], r[1]]; },
  // TERI: earliest-reachable + FREEZE the aim once inside the bubble (commit & ram)
  teri(s, st) {
    const Px = s.W + 2 * B_BORDER, Py = s.H + 2 * B_BORDER;
    const cdx = minimg(s.bx - s.px, Px), cdy = minimg(s.by - s.py, Py);
    const curdist = Math.hypot(cdx, cdy);
    const FREEZE_R = st.freeze ?? 110, SLACK = st.slack ?? 0.97;
    if (st.frozen && curdist < FREEZE_R) return st.aim;
    const r = earliestReach(s.W, s.H, s.px, s.py, s.bx, s.by, s.bvx, s.bvy, SLACK, 1400, 4);
    st.aim = [r[0], r[1]];
    if (curdist < FREEZE_R) st.frozen = true;
    return st.aim;
  },
};

function runEpisode(W, H, speed, seed, polName, polOpts, noFlee, maxFrames) {
  const rnd = mulberry32((seed) ^ 0x9e3779b9);
  // boid scattered; predator at center (matches endgame_fasteval intent)
  let bx = rnd() * W, by = rnd() * H, ang = rnd() * Math.PI * 2;
  let bvx = Math.cos(ang) * speed, bvy = Math.sin(ang) * speed;
  let px = W / 2, py = H / 2, pvx = rnd() * 2 - 1, pvy = rnd() * 2 - 1;
  const pol = POLICIES[polName], st = Object.assign({}, polOpts);
  for (let f = 0; f < maxFrames; f++) {
    // ---- predator decides + moves (seek aim at full speed) ----
    const s = { W, H, px, py, bx, by, bvx, bvy };
    let [ax, ay] = pol(s, st);
    const am = Math.hypot(ax, ay) || 1e-9;
    const desx = ax / am * PREDATOR_MAX_SPEED, desy = ay / am * PREDATOR_MAX_SPEED;
    let [sfx, sfy] = limit(desx - pvx, desy - pvy, PREDATOR_MAX_FORCE);
    [pvx, pvy] = limit(pvx + sfx, pvy + sfy, PREDATOR_MAX_SPEED);
    px += pvx; py += pvy;
    if (px > W + P_BORDER) px = -P_BORDER; if (px < -P_BORDER) px = W + P_BORDER;
    if (py > H + P_BORDER) py = -P_BORDER; if (py < -P_BORDER) py = H + P_BORDER;
    // catch check happens in predator.update() each frame (after predator moves)
    if (catches(px, py, pvx, pvy, PREDATOR_SIZE, bx, by, bvx, bvy)) return f + 1;
    // ---- boid flees + moves ----
    let acx = 0, acy = 0;
    if (!noFlee) {
      const dist = Math.hypot(bx - px, by - py) + EPSILON;
      if (dist < PREDATOR_RANGE) {
        let avx = (bx - px), avy = (by - py); const an = Math.hypot(avx, avy) || 1e-9;
        avx /= an; avy /= an;
        const strength = (PREDATOR_RANGE - dist) / PREDATOR_RANGE;
        avx *= strength * PREDATOR_TURN_FACTOR; avy *= strength * PREDATOR_TURN_FACTOR;
        [avx, avy] = limit(avx, avy, BMAX_FORCE * 1.5);
        acx += avx; acy += avy;
      }
    }
    bvx += acx; bvy += acy; [bvx, bvy] = limit(bvx, bvy, MAX_SPEED);
    bx += bvx; by += bvy;
    if (bx > W + B_BORDER) bx = -B_BORDER; if (bx < -B_BORDER) bx = W + B_BORDER;
    if (by > H + B_BORDER) by = -B_BORDER; if (by < -B_BORDER) by = H + B_BORDER;
  }
  return maxFrames;
}

function parse(argv) {
  const a = { W: 390, H: 844, speed: 6, seeds: 512, seedStart: 300000, pol: 'teri',
              maxFrames: 4000, noFlee: false, freeze: 110, slack: 0.97 };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i];
    if (k === '--W') a.W = +argv[++i]; else if (k === '--H') a.H = +argv[++i];
    else if (k === '--speed') a.speed = +argv[++i]; else if (k === '--seeds') a.seeds = +argv[++i];
    else if (k === '--seedStart') a.seedStart = +argv[++i]; else if (k === '--pol') a.pol = argv[++i];
    else if (k === '--maxFrames') a.maxFrames = +argv[++i]; else if (k === '--noFlee') a.noFlee = true;
    else if (k === '--freeze') a.freeze = +argv[++i]; else if (k === '--slack') a.slack = +argv[++i];
  }
  return a;
}
function main() {
  const o = parse(process.argv);
  const ttc = [];
  for (let s = 0; s < o.seeds; s++) {
    ttc.push(runEpisode(o.W, o.H, o.speed, o.seedStart + s, o.pol,
      { freeze: o.freeze, slack: o.slack }, o.noFlee, o.maxFrames));
  }
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const cleared = ttc.filter(t => t < o.maxFrames).length / ttc.length;
  const sorted = [...ttc].sort((a, b) => a - b);
  console.log(JSON.stringify({
    pol: o.pol, W: o.W, H: o.H, speed: o.speed, noFlee: o.noFlee,
    freeze: o.freeze, slack: o.slack, seeds: o.seeds,
    meanTTC: +mean(ttc).toFixed(1), median: sorted[Math.floor(0.5 * sorted.length)],
    p90: sorted[Math.floor(0.9 * sorted.length)], clearRate: +cleared.toFixed(3)
  }));
}
main();
