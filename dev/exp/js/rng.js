// Deterministic RNG + virtual clock shared by the live page, the headless
// oracle, and the NN runtime. Same seed -> identical simulation trace.
//
// This is the only file the predator NN training pipeline assumes about
// non-determinism. Math.random() / Date.now() must never be called from
// simulation code; use simRandom() / simNow() instead.

(function (root) {
    // mulberry32: 32-bit state, fast, good distribution for this use case.
    function mulberry32(seed) {
        var state = seed >>> 0;
        return function () {
            state = (state + 0x6D2B79F5) >>> 0;
            var t = state;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    var _rng = null;
    var _frame = 0;
    var _frameMs = 12; // matches REFRESH_INTERVAL_IN_MS default; reset by setSimSeed

    function setSimSeed(seed, frameMs) {
        _rng = mulberry32(seed >>> 0);
        _frame = 0;
        if (typeof frameMs === 'number') {
            _frameMs = frameMs;
        }
    }

    function simRandom() {
        if (_rng === null) {
            // Live page bootstrap: pick a fresh seed from real entropy on first
            // call so the user sees a different sim each visit. Headless
            // callers always call setSimSeed first.
            var bootSeed = (typeof Date !== 'undefined' ? Date.now() : 1) ^
                           Math.floor(Math.random() * 0x7fffffff);
            setSimSeed(bootSeed);
        }
        return _rng();
    }

    function simNow() {
        return _frame * _frameMs;
    }

    function simTick() {
        _frame += 1;
    }

    function getSimFrame() {
        return _frame;
    }

    function setFrameMs(ms) {
        _frameMs = ms;
    }

    var api = {
        setSimSeed: setSimSeed,
        simRandom: simRandom,
        simNow: simNow,
        simTick: simTick,
        getSimFrame: getSimFrame,
        setFrameMs: setFrameMs,
        mulberry32: mulberry32,
    };

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    }
    // Always expose on global so the rest of the JS sees simRandom/simNow.
    // We attach to BOTH globalThis (works in Node vm sandboxes) and window
    // (the live browser page) so either runtime resolves the names.
    var globals = [];
    if (typeof globalThis !== 'undefined') globals.push(globalThis);
    if (typeof window !== 'undefined' && window !== globalThis) globals.push(window);
    for (var i = 0; i < globals.length; i++) {
        var g = globals[i];
        g.setSimSeed = setSimSeed;
        g.simRandom = simRandom;
        g.simNow = simNow;
        g.simTick = simTick;
        g.getSimFrame = getSimFrame;
        g.setFrameMs = setFrameMs;
        g.mulberry32 = mulberry32;
    }
})(typeof globalThis !== 'undefined' ? globalThis : (typeof window !== 'undefined' ? window : this));
