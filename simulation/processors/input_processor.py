"""
Input Processor - Transformer-ready format for neural network inputs

This is a data conversion layer that transforms game state into structured inputs
for the neural network. This MUST match exactly with the JavaScript implementation.
"""

import math
from typing import List, Dict, Any

from config.constants import CONSTANTS

class InputProcessor:
    def __init__(self):
        # Use centralized constants (matches JavaScript)
        self.max_distance = CONSTANTS.MAX_DISTANCE
        
        # Unified velocity normalization - use the maximum of boid and predator speeds
        self.unified_max_velocity = max(
            CONSTANTS.BOID_MAX_SPEED,
            CONSTANTS.PREDATOR_MAX_SPEED
        )
    
    def process_inputs(self, boids: List[Dict], predator_pos: Dict, predator_vel: Dict, 
                      canvas_width: float, canvas_height: float) -> Dict[str, Any]:
        """
        Convert game state to transformer-friendly format
        
        Args:
            boids: List of boid objects with position and velocity
            predator_pos: Predator position {'x': float, 'y': float}
            predator_vel: Predator velocity {'x': float, 'y': float}
            canvas_width: Canvas width
            canvas_height: Canvas height
            
        Returns:
            Structured input with context, predator, and boids arrays
        """
        # Context information for world boundary awareness
        context = {
            'canvasWidth': canvas_width / self.max_distance,  # Normalized canvas width
            'canvasHeight': canvas_height / self.max_distance  # Normalized canvas height
        }
        
        # Predator velocity information
        predator = {
            'velX': self.clamp(predator_vel['x'] / self.unified_max_velocity),
            'velY': self.clamp(predator_vel['y'] / self.unified_max_velocity)
        }
        
        # Dynamic array of boids with relative positions and velocities
        boid_array = []
        for boid in boids:
            boid_data = self.encode_boid(boid, predator_pos, canvas_width, canvas_height)
            boid_array.append({
                'relX': boid_data[0],
                'relY': boid_data[1],
                'velX': boid_data[2],
                'velY': boid_data[3]
            })
        
        return {
            'context': context,
            'predator': predator,
            'boids': boid_array
        }
    
    def encode_boid(self, boid: Dict, predator_pos: Dict, canvas_width: float, canvas_height: float) -> List[float]:
        """
        Encode a single boid as relative position and velocity
        
        Args:
            boid: Boid object with position and velocity
            predator_pos: Predator position for relative calculation
            canvas_width: Canvas width for edge wrapping calculation
            canvas_height: Canvas height for edge wrapping calculation
            
        Returns:
            [rel_x, rel_y, vel_x, vel_y] - normalized values
        """
        # Calculate relative position with edge wrapping support
        relative_pos = self.calculate_relative_position(
            boid['position'], predator_pos, canvas_width, canvas_height
        )
        
        # Normalize relative position by fixed maximum distance
        rel_x = relative_pos['x'] / self.max_distance
        rel_y = relative_pos['y'] / self.max_distance
        
        # Normalize and clamp velocity by unified max velocity
        vel_x = self.clamp(boid['velocity']['x'] / self.unified_max_velocity)
        vel_y = self.clamp(boid['velocity']['y'] / self.unified_max_velocity)
        
        return [rel_x, rel_y, vel_x, vel_y]
    
    def calculate_relative_position(self, boid_pos: Dict, predator_pos: Dict, 
                                  canvas_width: float, canvas_height: float) -> Dict[str, float]:
        """
        Calculate relative position with edge wrapping support
        
        Args:
            boid_pos: Boid position {'x': float, 'y': float}
            predator_pos: Predator position {'x': float, 'y': float}
            canvas_width: Canvas width
            canvas_height: Canvas height
            
        Returns:
            Relative position {'x': float, 'y': float}
        """
        # Calculate basic relative position
        dx = boid_pos['x'] - predator_pos['x']
        dy = boid_pos['y'] - predator_pos['y']
        
        # Handle edge wrapping (toroidal topology)
        # Choose the shorter distance considering wrapping
        if abs(dx) > canvas_width / 2:
            dx = dx - canvas_width if dx > 0 else dx + canvas_width
        if abs(dy) > canvas_height / 2:
            dy = dy - canvas_height if dy > 0 else dy + canvas_height
        
        return {'x': dx, 'y': dy}
    
    def clamp(self, value: float) -> float:
        """Clamp value to [-1, 1] range"""
        return max(-1, min(1, value)) 