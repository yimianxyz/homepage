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

	initialize_canvas_simulation('boids1', false);
}); 