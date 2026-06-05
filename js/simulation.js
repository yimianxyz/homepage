// Detect mobile devices and adjust performance accordingly
function isMobileDevice() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
           window.innerWidth <= 768;
}

// Adjust number of boids based on device capabilities
// Increased slightly due to performance improvements from Cornell optimizations
var NUM_BOIDS = isMobileDevice() ? 60 : 120;
var REFRESH_INTERVAL_IN_MS = isMobileDevice() ? 18 : 12;

// Predator performance settings
var PREDATOR_MOBILE_RANGE = 60;  // Smaller range on mobile for better performance
var PREDATOR_DESKTOP_RANGE = 80;

function Simulation(name) {
	var canvas = document.getElementById(name);
	this.ctx = canvas.getContext('2d');
	this.canvasHeight = canvas.height;
	this.canvasWidth = canvas.width;
	this.separationMultiplier = 2;
	this.cohesionMultiplier = 1;
	this.alignmentMultiplier = 1;
	// Cumulative count of boids the predator has caught since page load —
	// surfaced by the activation viz caption ("N left · M eaten").
	this.boidsEaten = 0;
}

Simulation.prototype = {
	initialize: function (use_obstacle) {
		this.obstacles = [];
		this.boids = [];
		var start_x = Math.floor(simRandom() * this.canvasWidth);
		var start_y = Math.floor(simRandom() * this.canvasHeight);
		for (var i = 0; i < NUM_BOIDS; i++) {
			var boid = new Boid(start_x, start_y, this);
			this.addBoid(boid);
		}
		
		// Initialize predator in center of canvas
		var predator_x = this.canvasWidth / 2;
		var predator_y = this.canvasHeight / 2;
		this.predator = new Predator(predator_x, predator_y, this);
	},
	addBoid: function (boid) {
		this.boids.push(boid);
	},
	addObstacle: function (obstacle) {
		this.obstacles.push(obstacle);
	},
	// Interactive spawn: user clicked/tapped at (x, y) in canvas coordinates.
	// Adds a boid at that point with the default unit-vector velocity from
	// the Boid constructor, plus a short-lived ripple entry that renderSpawns
	// draws as a fading expanding circle for visual confirmation.
	spawnBoid: function (x, y) {
		this.addBoid(new Boid(x, y, this));
		if (!this.spawns) this.spawns = [];
		this.spawns.push({ x: x, y: y, t0: simNow() });
	},
	// 400ms expanding/fading stroke ring at each recent spawn point. Drawn
	// behind the boids (called first in render) so the new arrival appears
	// on top of its own welcome glow. Palette matches the gray-ladder used
	// by the activation viz and the h1.
	renderSpawns: function () {
		if (!this.spawns || !this.spawns.length) return;
		var DUR = 400;
		var now = simNow();
		var remaining = [];
		for (var i = 0; i < this.spawns.length; i++) {
			var s = this.spawns[i];
			var t = (now - s.t0) / DUR;
			if (t >= 1) continue;
			var r = 4 + t * 28;
			var alpha = (1 - t) * 0.5;
			this.ctx.beginPath();
			this.ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
			this.ctx.strokeStyle = 'rgba(85, 85, 85, ' + alpha.toFixed(3) + ')';
			this.ctx.lineWidth = 1.2;
			this.ctx.stroke();
			remaining.push(s);
		}
		this.spawns = remaining;
	},
	render: function () {
		this.ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);
		this.renderSpawns();

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
		}
		this.boidsEaten += caughtBoids.length;
		
		// Render predator
		this.predator.render();

		// Real-time neural activation diagram, drawn on top of the
		// simulation but underneath the centered text content.
		renderActivationViz(this.ctx, this);
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
		if (typeof setFrameMs === 'function') {
			setFrameMs(REFRESH_INTERVAL_IN_MS);
		}
		// Single-pass boids: each frame flocks once (render() runs flock()+update()
		// per boid). The predator's distilled policy was trained/tuned in this same
		// single-pass regime, so this is the dynamics it expects — and it's also a
		// ~30-40% speedup. (tick() is the obstacle/death-throw lifecycle, used only
		// when obstacles are present, which the homepage has none of.)
		setInterval(function () {
			if (typeof simTick === 'function') {
				simTick();
			}
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