var MAX_SPEED = 6;
var MAX_FORCE = 0.1;
var DESIRED_SEPARATION = 40;
var OBSTACLE_DESIRED_SEPARATION = 200;
var NEIGHBOR_DISTANCE = 60;
var BORDER_OFFSET = 10;
var EPSILON = 0.0000001;
var render_size = 10;
var death_throws = 0;
var avoidanceMultiplier = 2;

// Predator avoidance parameters (from Cornell ECE 5730 implementation)
// Fixed range for all devices to ensure consistent training behavior
var PREDATOR_RANGE = 75; // Device-independent value for consistent neural network training
var PREDATOR_TURN_FACTOR = 0.3;

function Boid(x, y, simulation) {
	var randomAngle = Math.random() * 2 * Math.PI;
	this.velocity = new Vector(Math.cos(randomAngle), Math.sin(randomAngle));
	this.position = new Vector(x, y);
	this.acceleration = new Vector(0, 0);
	this.simulation = simulation;
	this.render_size = render_size;
	this.death_throws = death_throws;
	this.sabateur = false;
}

Boid.prototype = {

	render: function () {
		var directionVector = this.velocity.normalize().multiplyBy(this.render_size);
		var inverseVector1 = new Vector(- directionVector.y, directionVector.x);
		var inverseVector2 = new Vector(directionVector.y, - directionVector.x);
		inverseVector1 = inverseVector1.divideBy(3);
		inverseVector2 = inverseVector2.divideBy(3);

		this.simulation.ctx.beginPath();
		this.simulation.ctx.moveTo(this.position.x, this.position.y);
		this.simulation.ctx.lineTo(this.position.x + inverseVector1.x, this.position.y + inverseVector1.y);
		this.simulation.ctx.lineTo(this.position.x + directionVector.x, this.position.y + directionVector.y);
		this.simulation.ctx.lineTo(this.position.x + inverseVector2.x, this.position.y + inverseVector2.y);
		this.simulation.ctx.lineTo(this.position.x, this.position.y);
		this.simulation.ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
		this.simulation.ctx.stroke();
		if (this.sabateur) {
			this.simulation.ctx.fillStyle = 'rgba(22, 236, 22, 0.8)';
		} else if (this.death_throws == 0) {
			this.simulation.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
		} else {
			this.simulation.ctx.fillStyle = 'rgba(22, 72, 236, 0.8)';
		}
		this.simulation.ctx.fill();
	},

	getCohesionVector: function (boids) {
		var totalPosition = new Vector(0, 0);
		var neighborCount = 0;
		for (var bi in boids) {
			var boid = boids[bi];
			if (this == boid) {
				continue;
			}

			var distance = this.position.getDistance(boid.position) + EPSILON;
			if (distance <= NEIGHBOR_DISTANCE) {
				totalPosition = totalPosition.add(boid.position);
				neighborCount++;
			}
		}

		if (neighborCount > 0) {
			var averagePosition = totalPosition.divideBy(neighborCount);
			return this.seek(averagePosition);
		} else {
			return new Vector(0, 0);
		}
	},

	seek: function (targetPosition) {
		var desiredVector = targetPosition.subtract(this.position);
		desiredVector.iFastSetMagnitude(MAX_SPEED);
		var steeringVector = desiredVector.subtract(this.velocity);
		steeringVector.iFastLimit(MAX_FORCE);
		return steeringVector;
	},

	getSeparationVector: function (boids) {
		var steeringVector = new Vector(0, 0);
		var neighborCount = 0;

		for (var bi in boids) {
			var boid = boids[bi];
			if (this == boid) {
				continue;
			}

			var distance = this.position.getDistance(boid.position) + EPSILON;
			if (distance > 0 && distance < DESIRED_SEPARATION) {
				var deltaVector = this.position.subtract(boid.position);
				deltaVector.iNormalize();
				deltaVector.iDivideBy(distance);
				steeringVector.iAdd(deltaVector);
				neighborCount++;
			}
		}

		if (neighborCount > 0) {
			var averageSteeringVector = steeringVector.divideBy(neighborCount);
		} else {
			var averageSteeringVector = new Vector(0, 0);
		}

		for (var ob in this.simulation.obstacles) {
			var obstacle = this.simulation.obstacles[ob];
			var distance = this.position.getDistance(obstacle.position) + EPSILON;
			if (distance > 0 && distance < OBSTACLE_DESIRED_SEPARATION) {
				var deltaVector = this.position.subtract(obstacle.position);
				deltaVector.iFastNormalize();
				deltaVector.iDivideBy(distance);
				steeringVector.iAdd(deltaVector);
				if (this.sabateur) {
					deltaVector.iMultiplyBy(-obstacle.repulsion);
				} else {
					deltaVector.iMultiplyBy(obstacle.repulsion);
				}
				averageSteeringVector.iAdd(deltaVector);
			}
		}

		if (averageSteeringVector.getFastMagnitude() > 0) {
			averageSteeringVector.iFastSetMagnitude(MAX_SPEED);
			averageSteeringVector.iSubtract(this.velocity);
			averageSteeringVector.iFastLimit(MAX_FORCE);
		}

		return averageSteeringVector;
	},

	getAlignmentVector: function (boids) {
		var perceivedFlockVelocity = new Vector(0, 0);
		var neighborCount = 0;

		for (var bi in boids) {
			var boid = boids[bi];
			if (this == boid) {
				continue;
			}

			var distance = this.position.getDistance(boid.position) + EPSILON;
			if (distance > 0 && distance < NEIGHBOR_DISTANCE) {
				perceivedFlockVelocity.iAdd(boid.velocity);
				neighborCount++;
			}
		}

		if (neighborCount > 0) {
			var averageVelocity = perceivedFlockVelocity.divideBy(neighborCount);
			averageVelocity.iFastSetMagnitude(MAX_SPEED);
			var steeringVector = averageVelocity.subtract(this.velocity);
			steeringVector.iFastLimit(MAX_FORCE);
			return steeringVector;
		} else {
			return new Vector(0, 0);
		}
	},

	// Predator avoidance behavior (based on Cornell ECE 5730 implementation)
	getPredatorAvoidanceVector: function(predator) {
		if (!predator) {
			return new Vector(0, 0);
		}
		
		var distance = this.position.getDistance(predator.position) + EPSILON;
		if (distance > 0 && distance < PREDATOR_RANGE) {
			// Calculate avoidance vector away from predator
			var avoidanceVector = this.position.subtract(predator.position);
			avoidanceVector.iFastNormalize();
			
			// Stronger avoidance when predator is closer
			var avoidanceStrength = (PREDATOR_RANGE - distance) / PREDATOR_RANGE;
			avoidanceVector.iMultiplyBy(avoidanceStrength * PREDATOR_TURN_FACTOR);
			
			// Apply maximum force limit - subtle avoidance
			avoidanceVector.iFastLimit(MAX_FORCE * 1.5); // Gentle avoidance, not too dramatic
			
			return avoidanceVector;
		}
		
		return new Vector(0, 0);
	},

	flock: function (boids) {
		var cohesionVector = this.getCohesionVector(boids);
		var separationVector = this.getSeparationVector(boids);
		var alignmentVector = this.getAlignmentVector(boids);

		separationVector.iMultiplyBy(this.simulation.separationMultiplier);
		cohesionVector.iMultiplyBy(this.simulation.cohesionMultiplier);
		alignmentVector.iMultiplyBy(this.simulation.alignmentMultiplier);

		this.acceleration.iAdd(cohesionVector);
		this.acceleration.iAdd(separationVector);
		this.acceleration.iAdd(alignmentVector);
		
		// Add predator avoidance if predator exists
		if (this.simulation.predator) {
			var predatorAvoidanceVector = this.getPredatorAvoidanceVector(this.simulation.predator);
			this.acceleration.iAdd(predatorAvoidanceVector);
		}
	},

	// Wrap-around boundary handling
	bound: function () {
		if (this.position.x > this.simulation.canvasWidth + BORDER_OFFSET) {
			this.position.x = -BORDER_OFFSET;
		}
		if (this.position.x < -BORDER_OFFSET) {
			this.position.x = this.simulation.canvasWidth + BORDER_OFFSET;
		}
		if (this.position.y > this.simulation.canvasHeight + BORDER_OFFSET) {
			this.position.y = -BORDER_OFFSET;
		}
		if (this.position.y < -BORDER_OFFSET) {
			this.position.y = this.simulation.canvasHeight + BORDER_OFFSET;
		}
	},

	update: function () {
		if (this.death_throws == 0) {
			this.velocity.iAdd(this.acceleration);
			this.velocity.iFastLimit(MAX_SPEED);
			this.position.iAdd(this.velocity);
			this.bound();
			this.acceleration.iMultiplyBy(0);
		}
	},

	run: function (boids) {
		this.flock(boids);
		this.update();
		this.render();
	},

	set_death_throws: function () {
		this.death_throws = 50;
	},

	decrease_death_throws: function () {
		this.death_throws = this.death_throws - 1;
	},

	set_sabateur: function (b) {
		this.sabateur = b;
	},
}; 