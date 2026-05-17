function Vector(x, y) {
	this.x = x;
	this.y = y;
}

Vector.prototype = {

	add: function(vector) {
		return new Vector(this.x + vector.x, this.y + vector.y);
	},

	update:function(x,y){
		this.x = x
		this.y = y
	},

	iAdd: function(vector) {
		this.x += vector.x;
		this.y += vector.y;
	},

	subtract: function(vector) {
		return new Vector(this.x - vector.x, this.y - vector.y);
	},

	iSubtract: function(vector) {
		this.x -= vector.x;
		this.y -= vector.y;
	},

	divideBy: function(factor) {
		return new Vector(this.x / factor, this.y / factor);
	},

	iDivideBy: function(factor) {
		this.x /= factor;
		this.y /= factor;
	},

	multiplyBy: function(factor) {
		return new Vector(this.x * factor, this.y * factor);
	},

	iMultiplyBy: function(factor) {
		this.x *= factor;
		this.y *= factor;
	},

	multiplyYBy: function(factor) {
		return new Vector(this.x, this.y * factor);
	},

	normalize: function() {
		if (this.getMagnitude() > 0) {
			return this.divideBy(this.getMagnitude());
		} else {
			return new Vector(0, 0);
		}
	},

	iNormalize: function() {
		if (this.getMagnitude() > 0) {
			this.iDivideBy(this.getMagnitude());
		}
	},

	round: function() {
		return new Vector(Math.round(this.x), Math.round(this.y));
	},

	setMagnitude: function(max) {
		return this.normalize().multiplyBy(max);
	},

	iSetMagnitude: function(max) {
		var unit = this.normalize();
		this.x = unit.x * max;
		this.y = unit.y * max;
	},

	limit: function(max) {
		if (this.getMagnitude() > max) {
			return this.setMagnitude(max);
		} else {
			return this;
		}
	},

	iLimit: function(max) {
		if (this.getMagnitude() > max) {
			this.iSetMagnitude(max);
		}
	},

	toString: function() {
		return '(' + this.x + ', ' + this.y + ')';
	},

	getAngle: function() {
		return Math.atan2(this.y, this.x);
	},
	getAngleInDegrees: function() {
		return this.getAngle() * (180 / Math.PI);
	},

	getMagnitude: function() {
		var origin = new Vector(0, 0);
		return origin.getDistance(this);
	},

	// Fast magnitude approximation using alpha-max-beta-min (from Cornell ECE 5730)
	// This avoids expensive sqrt operations: ~3x faster than true magnitude
	getFastMagnitude: function() {
		var alpha = 0.96;  // approximately 1
		var beta = 0.398;  // approximately 1/4
		var absX = Math.abs(this.x);
		var absY = Math.abs(this.y);
		return Math.max(absX, absY) * alpha + Math.min(absX, absY) * beta;
	},

	// Optimized limit function using fast magnitude
	iFastLimit: function(max) {
		var fastMag = this.getFastMagnitude();
		if (fastMag > max) {
			var scale = max / fastMag;
			this.x *= scale;
			this.y *= scale;
		}
	},

	// Optimized normalize function using fast magnitude
	iFastNormalize: function() {
		var fastMag = this.getFastMagnitude();
		if (fastMag > 0) {
			this.x /= fastMag;
			this.y /= fastMag;
		}
	},

	// Optimized set magnitude function using fast magnitude
	iFastSetMagnitude: function(max) {
		var fastMag = this.getFastMagnitude();
		if (fastMag > 0) {
			var scale = max / fastMag;
			this.x *= scale;
			this.y *= scale;
		}
	},

	getDistance: function(vector) {
		return Math.sqrt(Math.pow(this.x - vector.x, 2) + Math.pow(this.y - vector.y, 2));
	},

	iSetAngle: function(angle) {
		this.x = Math.cos(angle);
		this.y = Math.sin(angle);
	}

}; 