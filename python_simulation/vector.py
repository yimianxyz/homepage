"""
Vector class - Exact port of JavaScript src/utils/vector.js

This MUST match exactly with the JavaScript implementation to ensure
identical simulation behavior.
"""

import math
from typing import Union

class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def add(self, vector: 'Vector') -> 'Vector':
        """Return new vector with sum"""
        return Vector(self.x + vector.x, self.y + vector.y)
    
    def iAdd(self, vector: 'Vector') -> None:
        """In-place add"""
        self.x += vector.x
        self.y += vector.y
    
    def subtract(self, vector: 'Vector') -> 'Vector':
        """Return new vector with difference"""
        return Vector(self.x - vector.x, self.y - vector.y)
    
    def iSubtract(self, vector: 'Vector') -> None:
        """In-place subtract"""
        self.x -= vector.x
        self.y -= vector.y
    
    def divideBy(self, factor: float) -> 'Vector':
        """Return new vector divided by factor"""
        return Vector(self.x / factor, self.y / factor)
    
    def iDivideBy(self, factor: float) -> None:
        """In-place divide"""
        self.x /= factor
        self.y /= factor
    
    def multiplyBy(self, factor: float) -> 'Vector':
        """Return new vector multiplied by factor"""
        return Vector(self.x * factor, self.y * factor)
    
    def iMultiplyBy(self, factor: float) -> None:
        """In-place multiply"""
        self.x *= factor
        self.y *= factor
    
    def normalize(self) -> 'Vector':
        """Return normalized vector"""
        magnitude = self.getMagnitude()
        if magnitude > 0:
            return self.divideBy(magnitude)
        else:
            return Vector(0, 0)
    
    def iNormalize(self) -> None:
        """In-place normalize"""
        magnitude = self.getMagnitude()
        if magnitude > 0:
            self.iDivideBy(magnitude)
    
    def setMagnitude(self, max_mag: float) -> 'Vector':
        """Return vector with specified magnitude"""
        return self.normalize().multiplyBy(max_mag)
    
    def iSetMagnitude(self, max_mag: float) -> None:
        """In-place set magnitude"""
        unit = self.normalize()
        self.x = unit.x * max_mag
        self.y = unit.y * max_mag
    
    def limit(self, max_mag: float) -> 'Vector':
        """Return vector limited to max magnitude"""
        if self.getMagnitude() > max_mag:
            return self.setMagnitude(max_mag)
        else:
            return Vector(self.x, self.y)  # Return copy
    
    def iLimit(self, max_mag: float) -> None:
        """In-place limit magnitude"""
        if self.getMagnitude() > max_mag:
            self.iSetMagnitude(max_mag)
    
    def getMagnitude(self) -> float:
        """Get magnitude using distance from origin"""
        origin = Vector(0, 0)
        return origin.getDistance(self)
    
    def getFastMagnitude(self) -> float:
        """Fast magnitude approximation - matches JavaScript exactly"""
        alpha = 0.96
        beta = 0.398
        abs_x = abs(self.x)
        abs_y = abs(self.y)
        return max(abs_x, abs_y) * alpha + min(abs_x, abs_y) * beta
    
    def iFastLimit(self, max_mag: float) -> None:
        """In-place fast limit using fast magnitude"""
        fast_mag = self.getFastMagnitude()
        if fast_mag > max_mag:
            scale = max_mag / fast_mag
            self.x *= scale
            self.y *= scale
    
    def iFastNormalize(self) -> None:
        """In-place fast normalize"""
        fast_mag = self.getFastMagnitude()
        if fast_mag > 0:
            self.x /= fast_mag
            self.y /= fast_mag
    
    def iFastSetMagnitude(self, max_mag: float) -> None:
        """In-place fast set magnitude"""
        fast_mag = self.getFastMagnitude()
        if fast_mag > 0:
            scale = max_mag / fast_mag
            self.x *= scale
            self.y *= scale
    
    def getDistance(self, vector: 'Vector') -> float:
        """Get distance to another vector"""
        return math.sqrt((self.x - vector.x) ** 2 + (self.y - vector.y) ** 2)
    
    def __str__(self) -> str:
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self) -> str:
        return self.__str__() 