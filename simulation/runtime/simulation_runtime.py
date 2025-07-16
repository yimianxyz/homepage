"""
Pure Simulation Runtime - Core simulation logic abstracted for exact JS/Python matching

This module contains the pure simulation runtime that takes current states and actions,
and returns the next states. It's designed to be identical line-by-line between
JavaScript and Python implementations.

Interface:
- Input: boids_states, predator_state, predator_action, canvas_width, canvas_height
- Output: {boids_states, predator_state, caught_boids}
"""

import math
from typing import List, Dict, Any, Tuple

# Import centralized constants
from config.constants import CONSTANTS

# === VECTOR MATH UTILITIES ===

def vector_add(v1: Dict[str, float], v2: Dict[str, float]) -> Dict[str, float]:
    """Add two vectors"""
    return {'x': v1['x'] + v2['x'], 'y': v1['y'] + v2['y']}

def vector_subtract(v1: Dict[str, float], v2: Dict[str, float]) -> Dict[str, float]:
    """Subtract two vectors"""
    return {'x': v1['x'] - v2['x'], 'y': v1['y'] - v2['y']}

def vector_multiply(v: Dict[str, float], scalar: float) -> Dict[str, float]:
    """Multiply vector by scalar"""
    return {'x': v['x'] * scalar, 'y': v['y'] * scalar}

def vector_divide(v: Dict[str, float], scalar: float) -> Dict[str, float]:
    """Divide vector by scalar"""
    return {'x': v['x'] / scalar, 'y': v['y'] / scalar}

def vector_distance(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Calculate distance between two vectors"""
    dx = v1['x'] - v2['x']
    dy = v1['y'] - v2['y']
    return math.sqrt(dx * dx + dy * dy)

def vector_fast_magnitude(v: Dict[str, float]) -> float:
    """Fast magnitude approximation - matches JavaScript exactly"""
    alpha = 0.96
    beta = 0.398
    abs_x = abs(v['x'])
    abs_y = abs(v['y'])
    return max(abs_x, abs_y) * alpha + min(abs_x, abs_y) * beta

def vector_normalize(v: Dict[str, float]) -> Dict[str, float]:
    """Normalize vector using exact magnitude"""
    magnitude = vector_distance(v, {'x': 0, 'y': 0})
    if magnitude > 0:
        return vector_divide(v, magnitude)
    else:
        return {'x': 0, 'y': 0}

def vector_fast_normalize(v: Dict[str, float]) -> Dict[str, float]:
    """Fast normalize using fast magnitude"""
    fast_mag = vector_fast_magnitude(v)
    if fast_mag > 0:
        return vector_divide(v, fast_mag)
    else:
        return {'x': 0, 'y': 0}

def vector_fast_set_magnitude(v: Dict[str, float], magnitude: float) -> Dict[str, float]:
    """Set magnitude using fast approximation"""
    fast_mag = vector_fast_magnitude(v)
    if fast_mag > 0:
        scale = magnitude / fast_mag
        return vector_multiply(v, scale)
    else:
        return {'x': 0, 'y': 0}

def vector_fast_limit(v: Dict[str, float], max_magnitude: float) -> Dict[str, float]:
    """Limit magnitude using fast approximation"""
    fast_mag = vector_fast_magnitude(v)
    if fast_mag > max_magnitude:
        scale = max_magnitude / fast_mag
        return vector_multiply(v, scale)
    else:
        return v

# === BOID BEHAVIOR FUNCTIONS ===

def boid_get_cohesion_vector(boid_index: int, boids_states: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate cohesion force - move towards average position of neighbors"""
    current_boid = boids_states[boid_index]
    total_position = {'x': 0, 'y': 0}
    neighbor_count = 0
    
    for i, other_boid in enumerate(boids_states):
        if i == boid_index:
            continue
        
        distance = vector_distance(current_boid['position'], other_boid['position']) + CONSTANTS.EPSILON
        if distance <= CONSTANTS.BOID_NEIGHBOR_DISTANCE:
            total_position = vector_add(total_position, other_boid['position'])
            neighbor_count += 1
    
    if neighbor_count > 0:
        average_position = vector_divide(total_position, neighbor_count)
        return boid_seek(current_boid, average_position)
    else:
        return {'x': 0, 'y': 0}

def boid_seek(boid_state: Dict[str, Any], target_position: Dict[str, float]) -> Dict[str, float]:
    """Seek towards target position"""
    desired_vector = vector_subtract(target_position, boid_state['position'])
    desired_vector = vector_fast_set_magnitude(desired_vector, CONSTANTS.BOID_MAX_SPEED)
    steering_vector = vector_subtract(desired_vector, boid_state['velocity'])
    steering_vector = vector_fast_limit(steering_vector, CONSTANTS.BOID_MAX_FORCE)
    return steering_vector

def boid_get_separation_vector(boid_index: int, boids_states: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate separation force - avoid crowding neighbors"""
    current_boid = boids_states[boid_index]
    steering_vector = {'x': 0, 'y': 0}
    neighbor_count = 0
    
    for i, other_boid in enumerate(boids_states):
        if i == boid_index:
            continue
        
        distance = vector_distance(current_boid['position'], other_boid['position']) + CONSTANTS.EPSILON
        if distance > 0 and distance < CONSTANTS.BOID_DESIRED_SEPARATION:
            delta_vector = vector_subtract(current_boid['position'], other_boid['position'])
            delta_vector = vector_normalize(delta_vector)
            delta_vector = vector_divide(delta_vector, distance)
            steering_vector = vector_add(steering_vector, delta_vector)
            neighbor_count += 1
    
    if neighbor_count > 0:
        average_steering_vector = vector_divide(steering_vector, neighbor_count)
        average_steering_vector = vector_fast_set_magnitude(average_steering_vector, CONSTANTS.BOID_MAX_SPEED)
        average_steering_vector = vector_subtract(average_steering_vector, current_boid['velocity'])
        average_steering_vector = vector_fast_limit(average_steering_vector, CONSTANTS.BOID_MAX_FORCE)
        return average_steering_vector
    else:
        return {'x': 0, 'y': 0}

def boid_get_alignment_vector(boid_index: int, boids_states: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate alignment force - steer towards average heading of neighbors"""
    current_boid = boids_states[boid_index]
    perceived_flock_velocity = {'x': 0, 'y': 0}
    neighbor_count = 0
    
    for i, other_boid in enumerate(boids_states):
        if i == boid_index:
            continue
        
        distance = vector_distance(current_boid['position'], other_boid['position']) + CONSTANTS.EPSILON
        if distance > 0 and distance < CONSTANTS.BOID_NEIGHBOR_DISTANCE:
            perceived_flock_velocity = vector_add(perceived_flock_velocity, other_boid['velocity'])
            neighbor_count += 1
    
    if neighbor_count > 0:
        average_velocity = vector_divide(perceived_flock_velocity, neighbor_count)
        average_velocity = vector_fast_set_magnitude(average_velocity, CONSTANTS.BOID_MAX_SPEED)
        steering_vector = vector_subtract(average_velocity, current_boid['velocity'])
        steering_vector = vector_fast_limit(steering_vector, CONSTANTS.BOID_MAX_FORCE)
        return steering_vector
    else:
        return {'x': 0, 'y': 0}

def boid_get_predator_avoidance_vector(boid_state: Dict[str, Any], predator_state: Dict[str, Any]) -> Dict[str, float]:
    """Calculate predator avoidance force"""
    distance = vector_distance(boid_state['position'], predator_state['position']) + CONSTANTS.EPSILON
    if distance > 0 and distance < CONSTANTS.PREDATOR_RANGE:
        avoidance_vector = vector_subtract(boid_state['position'], predator_state['position'])
        avoidance_vector = vector_fast_normalize(avoidance_vector)
        
        avoidance_strength = (CONSTANTS.PREDATOR_RANGE - distance) / CONSTANTS.PREDATOR_RANGE
        avoidance_vector = vector_multiply(avoidance_vector, avoidance_strength * CONSTANTS.PREDATOR_TURN_FACTOR)
        avoidance_vector = vector_fast_limit(avoidance_vector, CONSTANTS.BOID_MAX_FORCE * 1.5)
        
        return avoidance_vector
    
    return {'x': 0, 'y': 0}

def boid_bound(position: Dict[str, float], canvas_width: float, canvas_height: float) -> Dict[str, float]:
    """Handle boundary wrapping for boid"""
    result = {'x': position['x'], 'y': position['y']}
    
    if result['x'] > canvas_width + CONSTANTS.BOID_BORDER_OFFSET:
        result['x'] = -CONSTANTS.BOID_BORDER_OFFSET
    if result['x'] < -CONSTANTS.BOID_BORDER_OFFSET:
        result['x'] = canvas_width + CONSTANTS.BOID_BORDER_OFFSET
    if result['y'] > canvas_height + CONSTANTS.BOID_BORDER_OFFSET:
        result['y'] = -CONSTANTS.BOID_BORDER_OFFSET
    if result['y'] < -CONSTANTS.BOID_BORDER_OFFSET:
        result['y'] = canvas_height + CONSTANTS.BOID_BORDER_OFFSET
    
    return result

# === PREDATOR BEHAVIOR FUNCTIONS ===

def predator_bound(position: Dict[str, float], canvas_width: float, canvas_height: float) -> Dict[str, float]:
    """Handle boundary wrapping for predator"""
    result = {'x': position['x'], 'y': position['y']}
    
    if result['x'] > canvas_width + CONSTANTS.PREDATOR_BORDER_OFFSET:
        result['x'] = -CONSTANTS.PREDATOR_BORDER_OFFSET
    if result['x'] < -CONSTANTS.PREDATOR_BORDER_OFFSET:
        result['x'] = canvas_width + CONSTANTS.PREDATOR_BORDER_OFFSET
    if result['y'] > canvas_height + CONSTANTS.PREDATOR_BORDER_OFFSET:
        result['y'] = -CONSTANTS.PREDATOR_BORDER_OFFSET
    if result['y'] < -CONSTANTS.PREDATOR_BORDER_OFFSET:
        result['y'] = canvas_height + CONSTANTS.PREDATOR_BORDER_OFFSET
    
    return result

def predator_check_for_prey(predator_state: Dict[str, Any], boids_states: List[Dict[str, Any]]) -> List[int]:
    """Check for caught boids and return their indices"""
    caught_boids = []
    catch_radius = CONSTANTS.PREDATOR_SIZE * 0.7
    
    for i, boid_state in enumerate(boids_states):
        distance = vector_distance(predator_state['position'], boid_state['position'])
        if distance < catch_radius:
            caught_boids.append(i)
    
    return caught_boids

# === MAIN SIMULATION RUNTIME ===

def simulation_step(
    boids_states: List[Dict[str, Any]],
    predator_state: Dict[str, Any],
    predator_action: Dict[str, float],
    canvas_width: float,
    canvas_height: float,
) -> Dict[str, Any]:
    """
    Pure simulation runtime - compute next state from current state and action
    
    Args:
        boids_states: List of boid states [{position: {x, y}, velocity: {x, y}}]
        predator_state: Predator state {position: {x, y}, velocity: {x, y}}
        predator_action: Predator action {force_x, force_y}
        canvas_width: Canvas width
        canvas_height: Canvas height
    
    Returns:
        {
            boids_states: [...],      # Updated boids states
            predator_state: {...},    # Updated predator state
            caught_boids: [...]       # Indices of caught boids
        }
    """
    # Deep copy states to avoid mutation
    new_boids_states = []
    for boid_state in boids_states:
        new_boids_states.append({
            'position': {'x': boid_state['position']['x'], 'y': boid_state['position']['y']},
            'velocity': {'x': boid_state['velocity']['x'], 'y': boid_state['velocity']['y']}
        })
    
    new_predator_state = {
        'position': {'x': predator_state['position']['x'], 'y': predator_state['position']['y']},
        'velocity': {'x': predator_state['velocity']['x'], 'y': predator_state['velocity']['y']}
    }
    
    # Step 1: Update all boids
    for i in range(len(new_boids_states)):
        boid_state = new_boids_states[i]
        
        # Calculate flocking forces
        cohesion_vector = boid_get_cohesion_vector(i, new_boids_states)
        separation_vector = boid_get_separation_vector(i, new_boids_states)
        alignment_vector = boid_get_alignment_vector(i, new_boids_states)
        
        # Apply multipliers
        separation_vector = vector_multiply(separation_vector, CONSTANTS.SEPARATION_MULTIPLIER)
        cohesion_vector = vector_multiply(cohesion_vector, CONSTANTS.COHESION_MULTIPLIER)
        alignment_vector = vector_multiply(alignment_vector, CONSTANTS.ALIGNMENT_MULTIPLIER)
        
        # Combine forces
        acceleration = vector_add(cohesion_vector, separation_vector)
        acceleration = vector_add(acceleration, alignment_vector)
        
        # Add predator avoidance
        predator_avoidance = boid_get_predator_avoidance_vector(boid_state, new_predator_state)
        acceleration = vector_add(acceleration, predator_avoidance)
        
        # Update physics
        boid_state['velocity'] = vector_add(boid_state['velocity'], acceleration)
        boid_state['velocity'] = vector_fast_limit(boid_state['velocity'], CONSTANTS.BOID_MAX_SPEED)
        boid_state['position'] = vector_add(boid_state['position'], boid_state['velocity'])
        boid_state['position'] = boid_bound(boid_state['position'], canvas_width, canvas_height)
    
    # Step 2: Update predator
    # Apply predator action to acceleration
    predator_acceleration = {'x': predator_action['force_x'], 'y': predator_action['force_y']}
    
    # Update predator physics
    new_predator_state['velocity'] = vector_add(new_predator_state['velocity'], predator_acceleration)
    new_predator_state['velocity'] = vector_fast_limit(new_predator_state['velocity'], CONSTANTS.PREDATOR_MAX_SPEED)
    new_predator_state['position'] = vector_add(new_predator_state['position'], new_predator_state['velocity'])
    new_predator_state['position'] = predator_bound(new_predator_state['position'], canvas_width, canvas_height)
    
    # Step 3: Check for caught boids
    caught_boids = predator_check_for_prey(new_predator_state, new_boids_states)
    
    return {
        'boids_states': new_boids_states,
        'predator_state': new_predator_state,
        'caught_boids': caught_boids
    } 