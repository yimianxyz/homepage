var MAX_SPEED = window.SIMULATION_CONSTANTS.BOID_MAX_SPEED;
var MAX_FORCE = window.SIMULATION_CONSTANTS.BOID_MAX_FORCE;
var DESIRED_SEPARATION = window.SIMULATION_CONSTANTS.BOID_DESIRED_SEPARATION;
var NEIGHBOR_DISTANCE = window.SIMULATION_CONSTANTS.BOID_NEIGHBOR_DISTANCE;
var BORDER_OFFSET = window.SIMULATION_CONSTANTS.BOID_BORDER_OFFSET;
var EPSILON = window.SIMULATION_CONSTANTS.EPSILON;
var render_size = window.SIMULATION_CONSTANTS.BOID_RENDER_SIZE;

// Predator avoidance parameters
var PREDATOR_RANGE = window.SIMULATION_CONSTANTS.PREDATOR_RANGE;
var PREDATOR_TURN_FACTOR = window.SIMULATION_CONSTANTS.PREDATOR_TURN_FACTOR;

function Boid(x, y, simulation) {
	var randomAngle = Math.random() * 2 * Math.PI;
	this.velocity = new Vector(Math.cos(randomAngle), Math.sin(randomAngle));
	this.position = new Vector(x, y);
	this.acceleration = new Vector(0, 0);
	this.simulation = simulation;
	this.render_size = render_size;
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
		this.simulation.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
		this.simulation.ctx.fill();
	},

	getCohesionVector: function (boids) {
		var totalPosition = new Vector(0, 0);
		var neighborCount = 0;
		for (var i = 0; i < boids.length; i++) {
			var boid = boids[i];
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

		for (var i = 0; i < boids.length; i++) {
			var boid = boids[i];
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
			averageSteeringVector.iFastSetMagnitude(MAX_SPEED);
			averageSteeringVector.iSubtract(this.velocity);
			averageSteeringVector.iFastLimit(MAX_FORCE);
			return averageSteeringVector;
		} else {
			return new Vector(0, 0);
		}
	},

	getAlignmentVector: function (boids) {
		var perceivedFlockVelocity = new Vector(0, 0);
		var neighborCount = 0;

		for (var i = 0; i < boids.length; i++) {
			var boid = boids[i];
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

	getPredatorAvoidanceVector: function(predator) {
		if (!predator) {
			return new Vector(0, 0);
		}
		
		var distance = this.position.getDistance(predator.position) + EPSILON;
		if (distance > 0 && distance < PREDATOR_RANGE) {
			var avoidanceVector = this.position.subtract(predator.position);
			avoidanceVector.iFastNormalize();
			
			var avoidanceStrength = (PREDATOR_RANGE - distance) / PREDATOR_RANGE;
			avoidanceVector.iMultiplyBy(avoidanceStrength * PREDATOR_TURN_FACTOR);
			avoidanceVector.iFastLimit(MAX_FORCE * 1.5);
			
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
		
		if (this.simulation.predator) {
			var predatorAvoidanceVector = this.getPredatorAvoidanceVector(this.simulation.predator);
			this.acceleration.iAdd(predatorAvoidanceVector);
		}
	},

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
		this.velocity.iAdd(this.acceleration);
		this.velocity.iFastLimit(MAX_SPEED);
		this.position.iAdd(this.velocity);
		this.bound();
		this.acceleration.iMultiplyBy(0);
	},

	run: function (boids) {
		this.flock(boids);
		this.update();
		this.render();
	}
}; 