// device_matrix.js — THE SPEC §5 device matrix, with each cell's engine
// parameters derived exactly as the live page derives them (simulation.js
// isMobileDevice(): UA regex OR innerWidth<=768).
//
//   * NUM_BOIDS:  60 if mobile else 120. The stepper derives it from width
//     only (its UA stub is 'Node'), so cells that are mobile BY UA REGEX
//     (820x1180 iPad Air) must force startBoids=60 — identical RNG draw
//     sequence to a real iPad page load (NUM_BOIDS is read once, in
//     initialize()).
//   * frameMs: REFRESH_INTERVAL_IN_MS = 18 mobile / 12 desktop. Causally dead
//     for the policy (simNow feeds lastFeed/nowMs fields that nothing reads
//     back — re-proven by oracle_logger --selftest), but kept faithful so the
//     virtual clock matches the browser bit-for-bit.
//   * PREDATOR_RANGE: 80 on EVERY device, including real phones/iPads. Load
//     order quirk: index.html runs boid.js (line 20) before simulation.js
//     (line 26) defines isMobileDevice, so `var PREDATOR_RANGE =
//     getBoidPredatorRange()` always takes the desktop fallback. The 60
//     branch is dead code at 6dce76f. buildHarness pins 80 (re-loads within
//     one process would otherwise flip it); shard headers log the value
//     as-evaluated.
//
// seedBase blocks are disjoint per cell; training allocation stays below the
// 270000 held-out boundary (verification seeds are >=270000).
'use strict';

const CELLS = [
    { id: 'iphone_390x844', W: 390,  H: 844,  mobile: true,  startBoids: 0,  frameMs: 18,
      seedBase: 100000, maxFrames: 24000, gamesPerShard: 8,
      note: 'mobile by width (<=768): NUM_BOIDS=60 auto' },
    { id: 'ipad_820x1180',  W: 820,  H: 1180, mobile: true,  startBoids: 60, frameMs: 18,
      seedBase: 110000, maxFrames: 60000, gamesPerShard: 6,
      note: 'mobile by UA regex (iPad): width rule alone would wrongly give 120 boids' },
    { id: 'desk_1024x768',  W: 1024, H: 768,  mobile: false, startBoids: 0,  frameMs: 12,
      seedBase: 120000, maxFrames: 90000, gamesPerShard: 4 },
    { id: 'desk_1512x982',  W: 1512, H: 982,  mobile: false, startBoids: 0,  frameMs: 12,
      seedBase: 130000, maxFrames: 90000, gamesPerShard: 4 },
    { id: 'desk_1680x1050', W: 1680, H: 1050, mobile: false, startBoids: 0,  frameMs: 12,
      seedBase: 140000, maxFrames: 90000, gamesPerShard: 4 },
    { id: 'desk_2560x1440', W: 2560, H: 1440, mobile: false, startBoids: 0,  frameMs: 12,
      seedBase: 150000, maxFrames: 90000, gamesPerShard: 2 },
];

const HELD_OUT_SEED = 270000;   // >=this: verification only, never trained on

module.exports = { CELLS, HELD_OUT_SEED };
