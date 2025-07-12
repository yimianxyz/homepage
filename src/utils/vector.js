function Vector(x, y) {
	this.x = x;
	this.y = y;
}

Vector.prototype = {

	add: function(vector) {
		return new Vector(this.x + vector.x, this.y + vector.y);
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

	getMagnitude: function() {
		var origin = new Vector(0, 0);
		return origin.getDistance(this);
	},

	getFastMagnitude: function() {
		var alpha = 0.96;
		var beta = 0.398;
		var absX = Math.abs(this.x);
		var absY = Math.abs(this.y);
		return Math.max(absX, absY) * alpha + Math.min(absX, absY) * beta;
	},

	iFastLimit: function(max) {
		var fastMag = this.getFastMagnitude();
		if (fastMag > max) {
			var scale = max / fastMag;
			this.x *= scale;
			this.y *= scale;
		}
	},

	iFastNormalize: function() {
		var fastMag = this.getFastMagnitude();
		if (fastMag > 0) {
			this.x /= fastMag;
			this.y /= fastMag;
		}
	},

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
	}
}; 