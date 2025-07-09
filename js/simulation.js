// Detect mobile devices and adjust performance accordingly
function isMobileDevice() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
           window.innerWidth <= 768;
}

// Adjust number of boids based on device capabilities
// Increased slightly due to performance improvements from Cornell optimizations
var NUM_BOIDS = isMobileDevice() ? 60 : 120;
var REFRESH_INTERVAL_IN_MS = isMobileDevice() ? 18 : 12;

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
	initialize: function (use_obstacle) {
		this.obstacles = [];
		this.boids = [];
		var start_x = Math.floor(Math.random() * this.canvasWidth);
		var start_y = Math.floor(Math.random() * this.canvasHeight);
		for (var i = 0; i < NUM_BOIDS; i++) {
			var boid = new Boid(start_x, start_y, this);
			this.addBoid(boid);
		}
		
		// Initialize neural predator in center of canvas
		var predator_x = this.canvasWidth / 2;
		var predator_y = this.canvasHeight / 2;
		this.predator = new NeuralPredator(predator_x, predator_y, this);
		
		// Connect predator to neural visualization
		if (typeof connectNeuralViz === 'function') {
			connectNeuralViz(this.predator);
		}
	},
	addBoid: function (boid) {
		this.boids.push(boid);
	},
	addObstacle: function (obstacle) {
		this.obstacles.push(obstacle);
	},
	render: function () {
		this.ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);

		for (var bi in this.boids) {
			this.boids[bi].run(this.boids);
		}
		for (var ob in this.obstacles) {
			this.obstacles[ob].render(this.obstacles);
		}
		
		// Update predator and handle predator-prey interactions
		this.predator.update(this.boids);
		
		// Check for caught boids (remove them smoothly)
		var caughtBoids = this.predator.checkForPrey(this.boids);
		for (var i = caughtBoids.length - 1; i >= 0; i--) {
			this.boids.splice(caughtBoids[i], 1);
			
			// If using neural predator, provide additional learning signal for successful catch
			if (this.predator.calculateReward) {
				var catchReward = this.predator.calculateReward(this.boids, true);
				this.predator.updateWeights(catchReward);
			}
		}
		
		// Render predator
		this.predator.render();
		
		// Optional: Log learning progress every few seconds (for debugging)
		if (this.predator.getLearningStats && !this.lastStatsLog) {
			this.lastStatsLog = Date.now();
			this.statsLogInterval = 15000; // Log every 15 seconds
		}
		if (this.predator.getLearningStats && Date.now() - this.lastStatsLog > this.statsLogInterval) {
			var stats = this.predator.getLearningStats();
			console.log('Neural Predator Learning Stats:', {
				avgReward: stats.avgReward.toFixed(3),
				efficiency: (stats.framesSinceLastFeed / 60).toFixed(1) + 's since last feed',
				size: stats.currentSize.toFixed(1),
				learningIntensity: stats.learningIntensity.toFixed(3),
				boidsRemaining: this.boids.length
			});
			this.lastStatsLog = Date.now();
		}
	},
	tick: function () {
		for (var bi in this.boids) {
			var boid = this.boids[bi];
			if (boid.death_throws == 0) {
				boid.flock(this.boids);
				for (var ob in this.obstacles) {
					var obstacle = this.obstacles[ob];
					if (boid.position.getDistance(obstacle.position) < boid.render_size + obstacle.radius) {
						this.boids[bi].set_death_throws();
					}
				}
			} else if (boid.death_throws == 1) {
				this.boids.splice(bi, 1);
			} else {
				this.boids[bi].decrease_death_throws();
			}
		}
	},
	run: function () {
		var self = this;
		self.tick();
		setInterval(function () {
			self.tick();
			self.render();
		}, REFRESH_INTERVAL_IN_MS);
	},

	update_sabateurs: function (v) {
		for (var bi in this.boids) {
			this.boids[bi].set_sabateur(parseInt(bi) < parseInt(v))
		}
	},

	update_separationMultiplier: function (value) {
		this.separationMultiplier = value;
	},

	update_cohesionMultiplier: function (value) {
		this.cohesionMultiplier = value;
	},

	update_alignmentMultiplier: function (value) {
		this.alignmentMultiplier = value;
	},
}; 