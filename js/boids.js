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
	// position. We use `click`, not `pointerdown`, so touch swipe / scroll
	// gestures do not fire and the synthetic click on touchend collapses
	// the touchstart→touchend→click flow into one event. `click` does still
	// fire after a desktop drag-to-select if mousedown/mouseup happen on
	// the same element, so we also skip when there's an active text
	// selection — that's the only case where the user clearly intended
	// something other than spawning.
	document.addEventListener('click', function (e) {
		if (e.button !== undefined && e.button !== 0) return;
		if (e.target && e.target.closest && e.target.closest('a')) return;
		var sel = window.getSelection && window.getSelection();
		if (sel && sel.toString().length > 0) return;
		if (window.__sim) window.__sim.spawnBoid(e.clientX, e.clientY);
	});
});