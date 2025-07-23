"""
Reward Processor - RL reward calculation with retroactive catch rewards

This module calculates rewards for reinforcement learning training based on:
1. Approaching reward - immediate reward for getting closer to boids
2. Catch retro reward - retroactive reward for actions that lead to catches

The processor looks ahead up to MAX_RETRO_REWARD_STEPS to assign retroactive
rewards when catches occur in the future.

This MUST match exactly with the JavaScript implementation.
"""

import math
from typing import List, Dict, Any, Union
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import CONSTANTS

class RewardProcessor:
    """Reward processor for RL training with approaching and catch retro rewards"""
    
    def __init__(self):
        # Use centralized constants
        self.max_retro_steps = CONSTANTS.MAX_RETRO_REWARD_STEPS
        self.max_distance = CONSTANTS.MAX_DISTANCE
        self.unified_max_velocity = max(
            CONSTANTS.BOID_MAX_SPEED,
            CONSTANTS.PREDATOR_MAX_SPEED
        )
    
    def process_rewards(self, reward_inputs: List[Dict[str, Any]], is_episode_end: bool) -> List[Dict[str, float]]:
        """
        Process rewards for a sequence of steps
        
        Args:
            reward_inputs: Array of reward input objects:
                [{
                    'caughtBoids': [boid_id1, boid_id2, ...],  # IDs of boids caught this step
                    'state': {context, predator, boids},        # Policy standard input format
                    'action': [x, y]                            # Policy standard output format
                }, ...]
            is_episode_end: True if the last step is episode end
            
        Returns:
            Array of reward objects:
                [{
                    'total': float,        # Total reward for this step
                    'approaching': float,  # Approaching reward component
                    'catchRetro': float   # Catch retro reward component
                }, ...]
        """
        if not reward_inputs or len(reward_inputs) == 0:
            return []
        
        # Determine output length based on episode end
        if is_episode_end:
            output_length = len(reward_inputs)
        else:
            output_length = max(0, len(reward_inputs) - self.max_retro_steps)
        
        if output_length == 0:
            return []
        
        rewards = []
        
        # Calculate rewards for each output step
        for i in range(output_length):
            approaching_reward = self._calculate_approaching_reward(reward_inputs[i])
            catch_retro_reward = self._calculate_catch_retro_reward(reward_inputs, i, is_episode_end)
            
            total_reward = approaching_reward + catch_retro_reward
            
            rewards.append({
                'total': total_reward,
                'approaching': approaching_reward,
                'catchRetro': catch_retro_reward
            })
        
        return rewards
    
    def _calculate_approaching_reward(self, reward_input: Dict[str, Any]) -> float:
        """
        Calculate approaching reward for a single step
        Rewards getting closer to the nearest boid and moving toward it
        
        Args:
            reward_input: Single reward input object
            
        Returns:
            Approaching reward value
        """
        state = reward_input['state']
        action = reward_input['action']
        
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
        
        # Calculate how well the action aligns with approach direction
        action_alignment = action[0] * approach_dir_x + action[1] * approach_dir_y
        
        # Calculate proximity reward (closer = better, max reward at distance 0)
        proximity_reward = max(0, 1 - min_distance)
        
        # Calculate relative velocity reward (boid and predator moving toward each other)
        predator_vel = state['predator']
        relative_vel_x = closest_boid['velX'] - predator_vel['velX']
        relative_vel_y = closest_boid['velY'] - predator_vel['velY']
        convergence_rate = -(relative_vel_x * approach_dir_x + relative_vel_y * approach_dir_y)
        velocity_reward = max(0, convergence_rate) * 0.5
        
        # Combine rewards (keep approaching reward much smaller than catch reward)
        base_reward = proximity_reward * 0.05 + velocity_reward * 0.03
        alignment_bonus = max(0, action_alignment) * base_reward * 0.1
        
        return base_reward + alignment_bonus
    
    def _calculate_catch_retro_reward(self, reward_inputs: List[Dict[str, Any]], 
                                     current_step: int, is_episode_end: bool) -> float:
        """
        Calculate catch retro reward for a single step
        Looks ahead for catches and assigns retroactive rewards
        
        Args:
            reward_inputs: All reward input objects
            current_step: Index of current step
            is_episode_end: True if episode ends
            
        Returns:
            Catch retro reward value
        """
        total_retro_reward = 0.0
        
        # Determine how far ahead we can look
        if is_episode_end:
            look_ahead_limit = min(len(reward_inputs), current_step + self.max_retro_steps + 1)
        else:
            look_ahead_limit = current_step + self.max_retro_steps + 1
        
        # Look ahead for catches
        for future_step in range(current_step + 1, look_ahead_limit):
            caught_boids = reward_inputs[future_step]['caughtBoids']
            
            if caught_boids and len(caught_boids) > 0:
                # For each caught boid, find when they started approaching each other
                for caught_boid_id in caught_boids:
                    first_approach_step = self._find_first_approach_step(
                        reward_inputs, 
                        current_step, 
                        future_step, 
                        caught_boid_id
                    )
                    
                    if first_approach_step != -1:
                        # Calculate retro reward for this catch
                        steps_from_first = current_step - first_approach_step
                        total_steps = future_step - first_approach_step
                        
                        if current_step >= first_approach_step and total_steps > 0:
                            # Linear increase: closer to catch = higher reward
                            progress_ratio = (steps_from_first + 1) / (total_steps + 1)
                            catch_reward = progress_ratio * 10.0  # Base catch reward
                            total_retro_reward += catch_reward
        
        return total_retro_reward
    
    def _find_first_approach_step(self, reward_inputs: List[Dict[str, Any]], 
                                 start_step: int, catch_step: int, boid_id: int) -> int:
        """
        Find the first step where predator and boid started approaching each other
        
        Args:
            reward_inputs: All reward input objects
            start_step: Start looking from this step
            catch_step: Step where catch occurred
            boid_id: ID of the caught boid
            
        Returns:
            Step index where approach started, or -1 if not found
        """
        # Look backwards from catch step to find first approach
        max_lookback = max(0, catch_step - self.max_retro_steps)
        search_start = max(start_step, max_lookback)
        
        for step in range(search_start, catch_step + 1):
            state = reward_inputs[step]['state']
            
            # Find the target boid in this step's state
            target_boid = None
            for boid in state['boids']:
                if boid['id'] == boid_id:
                    target_boid = boid
                    break
            
            if target_boid:
                # Check if they're moving toward each other
                is_approaching = self._are_moving_toward_each_other(state['predator'], target_boid)
                if is_approaching:
                    return step
        
        return -1
    
    def _are_moving_toward_each_other(self, predator: Dict[str, float], boid: Dict[str, float]) -> bool:
        """
        Check if predator and boid are moving toward each other
        
        Args:
            predator: Predator state {velX, velY}
            boid: Boid state {relX, relY, velX, velY}
            
        Returns:
            True if moving toward each other
        """
        # Calculate relative position vector (from predator to boid)
        distance = math.sqrt(boid['relX'] ** 2 + boid['relY'] ** 2)
        if distance < 0.001:
            return False
        
        dir_to_boid_x = boid['relX'] / distance
        dir_to_boid_y = boid['relY'] / distance
        
        # Calculate relative velocity (boid velocity - predator velocity)
        rel_vel_x = boid['velX'] - predator['velX']
        rel_vel_y = boid['velY'] - predator['velY']
        
        # Dot product: negative means moving toward each other
        velocity_dot_product = rel_vel_x * dir_to_boid_x + rel_vel_y * dir_to_boid_y
        
        return velocity_dot_product < 0

def create_reward_processor():
    """Create reward processor instance"""
    processor = RewardProcessor()
    print(f"Created RewardProcessor:")
    print(f"  Max retro steps: {processor.max_retro_steps}")
    print(f"  Max distance: {processor.max_distance}")
    print(f"  Unified max velocity: {processor.unified_max_velocity}")
    return processor

if __name__ == "__main__":
    # Test reward processor
    processor = create_reward_processor()
    
    # Test with dummy data
    test_inputs = [
        {
            'caughtBoids': [],
            'state': {
                'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
                'predator': {'velX': 0.1, 'velY': -0.2},
                'boids': [
                    {'id': 1, 'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
                    {'id': 2, 'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
                ]
            },
            'action': [0.5, -0.8]
        },
        {
            'caughtBoids': [1],  # Catch boid 1
            'state': {
                'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
                'predator': {'velX': 0.2, 'velY': -0.3},
                'boids': [
                    {'id': 2, 'relX': -0.15, 'relY': 0.05, 'velX': -0.2, 'velY': 0.3}
                ]
            },
            'action': [0.3, -0.6]
        }
    ]
    
    rewards = processor.process_rewards(test_inputs, is_episode_end=True)
    
    print(f"\nTest results:")
    for i, reward in enumerate(rewards):
        print(f"  Step {i}: total={reward['total']:.3f}, approaching={reward['approaching']:.3f}, catchRetro={reward['catchRetro']:.3f}") 