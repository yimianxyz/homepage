document.addEventListener('DOMContentLoaded', function() {
	if (!isCanvasSupported()) {
		return;
	}

	resizeCanvas();

	var initialize_canvas_simulation = function (name, use_obstacle) {
		var simulation = new Simulation(name);
		simulation.initialize(use_obstacle);
		simulation.run();
		return simulation;
	};

	// The predator's policy is a trained neural network; wait for its
	// weights to load before starting the simulation, so the NN drives
	// the predator from frame 1.
	window.__predatorReady.then(function () {
		// Expose on window so the interactive layer (below) can reach it.
		window.__sim = initialize_canvas_simulation('boids1', false);
	});

	// Tap / click anywhere on the page spawns a new boid at the pointer
	// position. We use pointer events instead of `click` because iOS Safari
	// only fires `click` on elements it considers "clickable" (cursor:pointer,
	// onclick handler, or interactive descendants) — a tap on the bare body
	// background of this page simply never bubbles a click event up to the
	// document. `pointerdown` + `pointerup` fire reliably on any element,
	// any input type, on every browser since 2019.
	//
	// Click-vs-drag detection is rolled by hand: we remember where the
	// pointer went down, and only spawn if it came up within an 8px radius
	// — far enough that desktop drag-to-select and touch-swipe-scroll
	// gestures naturally skip the spawn, close enough that any reasonable
	// tap registers. The text-selection guard catches the remaining edge
	// case where the user drags within a single paragraph (down and up on
	// the same element, browser fires click anyway, but selection is now
	// non-empty).
	var pdX = 0, pdY = 0, pdId = null;
	document.addEventListener('pointerdown', function (e) {
		if (e.button !== undefined && e.button !== 0) { pdId = null; return; }
		pdX = e.clientX;
		pdY = e.clientY;
		pdId = e.pointerId;
	});
	document.addEventListener('pointerup', function (e) {
		if (pdId !== e.pointerId) return;
		pdId = null;
		if (e.target && e.target.closest && e.target.closest('a')) return;
		var dx = e.clientX - pdX;
		var dy = e.clientY - pdY;
		if (dx * dx + dy * dy > 64) return;             // >8px = drag, not tap
		var sel = window.getSelection && window.getSelection();
		if (sel && sel.toString().length > 0) return;   // user selected text
		if (window.__sim) window.__sim.spawnBoid(e.clientX, e.clientY);
	});
	document.addEventListener('pointercancel', function () { pdId = null; });
});