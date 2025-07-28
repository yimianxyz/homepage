"""
Simple Reward Processor - Instant reward calculation for RL training

This module calculates instant rewards for each simulation step based on:
1. Catch reward - dominant reward for catching boids (instant)
2. Approaching reward - small reward for getting closer to boids (instant)

Design principle: Occam's razor - simple, clean, single-step processing

This MUST match exactly with the JavaScript implementation.
"""

import math
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import CONSTANTS

class RewardProcessor:
    """Simple reward processor for RL training with instant rewards"""
    
    def __init__(self):
        # Base catch reward per boid caught (dominant reward)
        self.base_catch_reward = 10.0
        
        # Use centralized constants for normalization
        self.max_distance = CONSTANTS.MAX_DISTANCE
        self.unified_max_velocity = max(
            CONSTANTS.BOID_MAX_SPEED,
            CONSTANTS.PREDATOR_MAX_SPEED
        )
    
    def calculate_step_reward(self, step_input: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate instant reward for a single simulation step
        
        Args:
            step_input: Single step input object:
                {
                    'state': {context, predator, boids},      # Policy standard input format
                    'action': [x, y],                         # Policy standard output format  
                    'caughtBoids': [boid_id1, boid_id2, ...] # IDs of boids caught this step
                }
                
        Returns:
            Reward object:
                {
                    'total': float,        # Total reward for this step
                    'approaching': float,  # Approaching reward component
                    'catch': float         # Catch reward component
                }
        """
        if not step_input or 'state' not in step_input or 'action' not in step_input:
            return {
                'total': 0.0,
                'approaching': 0.0,
                'catch': 0.0
            }
        
        # Calculate approaching reward (same logic as before, simplified)
        approaching_reward = self._calculate_approaching_reward(
            step_input['state'], 
            step_input['action']
        )
        
        # Calculate catch reward (simple: count * base reward)
        catch_reward = self._calculate_catch_reward(
            step_input.get('caughtBoids', [])
        )
        
        # Combine rewards
        total_reward = approaching_reward + catch_reward
        
        return {
            'total': total_reward,
            'approaching': approaching_reward,
            'catch': catch_reward
        }
    
    def _calculate_approaching_reward(self, state: Dict[str, Any], action: List[float]) -> float:
        """
        Calculate approaching reward - rewards getting closer to the nearest boid
        
        Args:
            state: Policy standard input format
            action: Policy output [x, y]
            
        Returns:
            Approaching reward value
        """
        if not state['boids'] or len(state['boids']) == 0:
            return 0.0
        
        # Find closest boid
        closest_boid = None
        min_distance = float('inf')
        
        for boid in state['boids']:
            distance = math.sqrt(boid['relX'] ** 2 + boid['relY'] ** 2)
            
            if distance < min_distance:
                min_distance = distance
                closest_boid = boid
        
        if not closest_boid or min_distance < 0.001:
            return 0.0
        
        # Calculate approach direction (normalized)
        approach_dir_x = closest_boid['relX'] / min_distance
        approach_dir_y = closest_boid['relY'] / min_distance
        
        # Action alignment reward - how well action aligns with approach direction
        action_alignment = action[0] * approach_dir_x + action[1] * approach_dir_y
        alignment_reward = max(0, action_alignment) * 0.1
        
        # Proximity reward - closer is better (max reward at distance 0)
        proximity_reward = max(0, 1 - min_distance) * 0.05
        
        # Velocity convergence reward - predator and boid moving toward each other
        predator_vel = state['predator']
        relative_vel_x = closest_boid['velX'] - predator_vel['velX']
        relative_vel_y = closest_boid['velY'] - predator_vel['velY']
        convergence_rate = -(relative_vel_x * approach_dir_x + relative_vel_y * approach_dir_y)
        velocity_reward = max(0, convergence_rate) * 0.03
        
        return proximity_reward + velocity_reward + alignment_reward
    
    def _calculate_catch_reward(self, caught_boids: List[int]) -> float:
        """
        Calculate catch reward - simple count-based reward
        
        Args:
            caught_boids: List of caught boid IDs
            
        Returns:
            Catch reward value
        """
        if not caught_boids or len(caught_boids) == 0:
            return 0.0
        
        # Simple: base reward per boid caught
        return len(caught_boids) * self.base_catch_reward

def create_reward_processor():
    """Create reward processor instance"""
    processor = RewardProcessor()
    print(f"Created Simple RewardProcessor:")
    print(f"  Base catch reward: {processor.base_catch_reward}")
    print(f"  Max distance: {processor.max_distance}")
    print(f"  Unified max velocity: {processor.unified_max_velocity}")
    return processor

if __name__ == "__main__":
    # Test reward processor
    processor = create_reward_processor()
    
    # Test with dummy data - no catch
    test_input_no_catch = {
        'state': {
            'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
            'predator': {'velX': 0.1, 'velY': -0.2},
            'boids': [
                {'id': 1, 'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
                {'id': 2, 'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
            ]
        },
        'action': [0.5, -0.8],
        'caughtBoids': []
    }
    
    # Test with dummy data - with catch
    test_input_with_catch = {
        'state': {
            'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
            'predator': {'velX': 0.2, 'velY': -0.3},
            'boids': [
                {'id': 2, 'relX': -0.15, 'relY': 0.05, 'velX': -0.2, 'velY': 0.3}
            ]
        },
        'action': [0.3, -0.6],
        'caughtBoids': [1]  # Caught boid 1
    }
    
    reward_no_catch = processor.calculate_step_reward(test_input_no_catch)
    reward_with_catch = processor.calculate_step_reward(test_input_with_catch)
    
    print(f"\nTest results:")
    print(f"  No catch: total={reward_no_catch['total']:.3f}, approaching={reward_no_catch['approaching']:.3f}, catch={reward_no_catch['catch']:.3f}")
    print(f"  With catch: total={reward_with_catch['total']:.3f}, approaching={reward_with_catch['approaching']:.3f}, catch={reward_with_catch['catch']:.3f}") 