// TEMPORARY, URL-flag-gated planner policy for visual inspection only.
//
// Activates ONLY when the page URL carries ?policy=planner. When absent (the
// default, i.e. production), window.__planner is never created and the hook in
// predator.js (getAutonomousForce) is a no-op — prod behaviour is byte-identical.
// This whole file + the 6-line hook are one-commit reversible.
//
// What it does: runs the ~21-catch receding-horizon planner ("teacher") that we
// distilled from. The expensive K-candidate × H-frame rollout runs OFF the main
// thread in predator_planner_worker.js (which importScripts the REAL sim math,
// so its flocking + predator-avoidance is identical to the live frame). Here on
// the main thread we only: (1) ship a state snapshot to the worker every D
// frames, (2) steer the live predator toward the latest committed target using
// the SAME analytic chase/seek the production CHASE branch uses.
//
// Tunable via URL: ?policy=planner&K=16&H=120&D=8  (defaults shown).
'use strict';

(function () {
    var params = new URLSearchParams(location.search);
    if (params.get('policy') !== 'planner') return;   // prod path: do nothing

    var K = parseInt(params.get('K'), 10) || 16;
    var H = parseInt(params.get('H'), 10) || 120;
    var D = parseInt(params.get('D'), 10) || 8;        // re-plan cadence (frames)

    var worker = new Worker('js/predator_planner_worker.js');
    var target = null;          // latest committed patrol target {x,y}
    var busy = false;           // a plan request is in flight
    var frame = 0;              // frames since last plan request
    var configured = false;

    worker.onmessage = function (e) {
        if (e.data && e.data.type === 'target') {
            target = { x: e.data.x, y: e.data.y };
            busy = false;
        }
    };
    worker.onerror = function (err) {
        console.error('[planner worker]', err.message || err);
        busy = false;
    };

    function configure(sim) {
        var POLICY_R = (window.__predatorModel && window.__predatorModel.POLICY_R) || 80;
        worker.postMessage({
            type: 'config', K: K, H: H, POLICY_R: POLICY_R,
            W: sim.canvasWidth, Hc: sim.canvasHeight,
            predRange: (typeof PREDATOR_RANGE !== 'undefined') ? PREDATOR_RANGE : 80
        });
        configured = true;
    }

    function snapshot(pred, boids, sim) {
        var n = boids.length;
        var bx = new Float64Array(n), by = new Float64Array(n);
        var bvx = new Float64Array(n), bvy = new Float64Array(n);
        for (var i = 0; i < n; i++) {
            bx[i] = boids[i].position.x; by[i] = boids[i].position.y;
            bvx[i] = boids[i].velocity.x; bvy[i] = boids[i].velocity.y;
        }
        return {
            type: 'plan',
            snapshot: {
                bx: bx, by: by, bvx: bvx, bvy: bvy,
                px: pred.position.x, py: pred.position.y,
                pvx: pred.velocity.x, pvy: pred.velocity.y,
                psize: pred.currentSize, lastFeed: pred.lastFeedTime,
                nowMs: simNow()
            }
        };
    }

    // Steer toward the committed target, mirroring predatorStep in the worker and
    // the production CHASE branch: chase the nearest boid if one is within
    // POLICY_R, else seek the target. Returns a steering force (acceleration).
    function steer(pred, boids) {
        var POLICY_R = (window.__predatorModel && window.__predatorModel.POLICY_R) || 80;
        var px = pred.position.x, py = pred.position.y;
        var bestD2 = Infinity, nx = 0, ny = 0;
        for (var i = 0, n = boids.length; i < n; i++) {
            var dx = boids[i].position.x - px, dy = boids[i].position.y - py;
            var d2 = dx * dx + dy * dy;
            if (d2 < bestD2) { bestD2 = d2; nx = dx; ny = dy; }
        }
        var desired;
        if (bestD2 < POLICY_R * POLICY_R) {
            desired = new Vector(nx, ny);
        } else if (target) {
            desired = new Vector(target.x - px, target.y - py);
        } else {
            desired = new Vector(nx, ny);   // no plan yet: fall back to nearest
        }
        desired.iFastSetMagnitude(PREDATOR_MAX_SPEED);
        var st = desired.subtract(pred.velocity);
        st.iFastLimit(PREDATOR_MAX_FORCE);
        return st;
    }

    window.__planner = {
        active: true,
        force: function (pred, boids) {
            var sim = pred.simulation;
            if (!configured && sim) configure(sim);
            if (boids.length === 0) return new Vector(0, 0);
            // Re-plan as soon as the worker is free (its ~0.4s latency already
            // exceeds the teacher's D-frame commit, so we minimise staleness by
            // dispatching a fresh snapshot the instant the previous plan lands;
            // D is only a floor so we never out-pace the teacher's cadence).
            frame++;
            if (configured && !busy && frame >= D) {
                busy = true;
                frame = 0;
                worker.postMessage(snapshot(pred, boids, sim));
            }
            return steer(pred, boids);
        }
    };

    console.info('[planner] active (K=' + K + ' H=' + H + ' D=' + D +
        ') — TEMPORARY policy=planner override; remove ?policy=planner for prod.');
})();
