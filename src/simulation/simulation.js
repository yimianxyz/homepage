// Simple fixed configuration for consistent behavior across all devices
var NUM_BOIDS = window.SIMULATION_CONSTANTS.NUM_BOIDS;

function Simulation(name) {
	var canvas = document.getElementById(name);
	this.ctx = canvas.getContext('2d');
	this.canvasHeight = canvas.height;
	this.canvasWidth = canvas.width;
	this.separationMultiplier = 2;
	this.cohesionMultiplier = 1;
	this.alignmentMultiplier = 1;
}

Simulation.prototype = {
	initialize: function (skipPredator) {
		this.boids = [];
		var start_x = Math.floor(Math.random() * this.canvasWidth);
		var start_y = Math.floor(Math.random() * this.canvasHeight);
		for (var i = 0; i < NUM_BOIDS; i++) {
			var boid = new Boid(start_x, start_y, this);
			this.boids.push(boid);
		}
		
		if (!skipPredator) {
			var predator_x = this.canvasWidth / 2;
			var predator_y = this.canvasHeight / 2;
			this.predator = new NeuralPredator(predator_x, predator_y, this);
		}
	},
	
	render: function () {
		this.ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);

		for (var i = 0; i < this.boids.length; i++) {
			this.boids[i].run(this.boids);
		}
		
		if (this.predator) {
			this.predator.update(this.boids);
			
			var caughtBoids = this.predator.checkForPrey(this.boids);
			for (var i = caughtBoids.length - 1; i >= 0; i--) {
				this.boids.splice(caughtBoids[i], 1);
			}
			
			this.predator.render();
		}
	}
}; 