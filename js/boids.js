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
		initialize_canvas_simulation('boids1', false);
	});
}); 