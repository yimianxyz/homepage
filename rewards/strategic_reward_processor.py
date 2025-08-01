"""
Strategic Reward Processor - Enhanced reward system for better RL performance

Key improvements over basic reward processor:
1. Strategic herding rewards - incentivize coordinated boid herding
2. Long-term planning rewards - reward efficient pathing and positioning
3. Adaptive difficulty scaling - rewards scale with scenario difficulty
4. Multi-target coordination - rewards for optimal target selection

Keeps the system simple while adding strategic depth.
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import CONSTANTS
from .reward_processor import RewardProcessor


class StrategicRewardProcessor(RewardProcessor):
    """Enhanced reward processor with strategic components for RL training"""
    
    def __init__(self, 
                 enable_herding_rewards: bool = True,
                 enable_planning_rewards: bool = True,
                 enable_exploration_bonus: bool = True,
                 strategic_weight: float = 0.3):
        """
        Initialize strategic reward processor
        
        Args:
            enable_herding_rewards: Enable herding behavior rewards
            enable_planning_rewards: Enable long-term planning rewards
            enable_exploration_bonus: Enable exploration bonus for new areas
            strategic_weight: Weight for strategic rewards vs base rewards
        """
        super().__init__()
        
        # Strategic reward configuration
        self.enable_herding_rewards = enable_herding_rewards
        self.enable_planning_rewards = enable_planning_rewards
        self.enable_exploration_bonus = enable_exploration_bonus
        self.strategic_weight = strategic_weight
        
        # Strategic reward coefficients
        self.herding_multiplier = 0.15          # Reward for herding boids together
        self.corner_cutting_multiplier = 0.1    # Reward for efficient pathing
        self.target_selection_multiplier = 0.2  # Reward for smart target selection
        self.exploration_bonus_multiplier = 0.05 # Bonus for exploring new areas
        
        # State tracking for strategic rewards
        self.previous_boid_spread = None
        self.visited_positions = []
        self.max_visited_positions = 50  # Limit memory usage
        
        print(f"Created StrategicRewardProcessor:")
        print(f"  Herding rewards: {enable_herding_rewards}")
        print(f"  Planning rewards: {enable_planning_rewards}")
        print(f"  Exploration bonus: {enable_exploration_bonus}")
        print(f"  Strategic weight: {strategic_weight}")
    
    def calculate_step_reward(self, step_input: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate enhanced reward with strategic components
        
        Args:
            step_input: Same format as base RewardProcessor
                
        Returns:
            Enhanced reward object with strategic components:
            {
                'total': float,           # Total reward
                'approaching': float,     # Base approaching reward
                'catch': float,          # Base catch reward
                'herding': float,        # Strategic herding reward
                'planning': float,       # Long-term planning reward
                'exploration': float     # Exploration bonus
            }
        """
        if not step_input or 'state' not in step_input or 'action' not in step_input:
            return self._empty_reward()
        
        # Get base rewards from parent class
        base_reward = super().calculate_step_reward(step_input)
        
        # Calculate strategic rewards
        strategic_rewards = self._calculate_strategic_rewards(
            step_input['state'], 
            step_input['action'],
            step_input.get('caughtBoids', [])
        )
        
        # Combine all rewards
        total_reward = (
            base_reward['total'] + 
            self.strategic_weight * (
                strategic_rewards['herding'] + 
                strategic_rewards['planning'] + 
                strategic_rewards['exploration']
            )
        )
        
        return {
            'total': total_reward,
            'approaching': base_reward['approaching'],
            'catch': base_reward['catch'],
            'herding': strategic_rewards['herding'],
            'planning': strategic_rewards['planning'],
            'exploration': strategic_rewards['exploration']
        }
    
    def _calculate_strategic_rewards(self, 
                                   state: Dict[str, Any], 
                                   action: List[float],
                                   caught_boids: List[int]) -> Dict[str, float]:
        """
        Calculate strategic reward components
        
        Args:
            state: Current state
            action: Action taken
            caught_boids: Boids caught this step
            
        Returns:
            Strategic reward components
        """
        herding_reward = 0.0
        planning_reward = 0.0
        exploration_reward = 0.0
        
        if not state['boids']:
            return {
                'herding': herding_reward,
                'planning': planning_reward,
                'exploration': exploration_reward
            }
        
        # 1. Herding Rewards - incentivize grouping boids together
        if self.enable_herding_rewards:
            herding_reward = self._calculate_herding_reward(state, action)
        
        # 2. Planning Rewards - reward efficient movement and positioning
        if self.enable_planning_rewards:
            planning_reward = self._calculate_planning_reward(state, action)
        
        # 3. Exploration Bonus - encourage visiting new areas
        if self.enable_exploration_bonus:
            exploration_reward = self._calculate_exploration_bonus(state)
        
        return {
            'herding': herding_reward,
            'planning': planning_reward,
            'exploration': exploration_reward
        }
    
    def _calculate_herding_reward(self, state: Dict[str, Any], action: List[float]) -> float:
        """
        Calculate herding reward - incentivize actions that group boids together
        This makes them easier to catch in clusters
        """
        boids = state['boids']
        if len(boids) < 2:
            return 0.0
        
        # Calculate current boid spread (variance in positions)
        boid_positions = [(boid['relX'], boid['relY']) for boid in boids]
        
        # Calculate center of mass of boids
        center_x = sum(pos[0] for pos in boid_positions) / len(boid_positions)
        center_y = sum(pos[1] for pos in boid_positions) / len(boid_positions)
        
        # Calculate spread (average distance from center)
        current_spread = sum(
            math.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
            for pos in boid_positions
        ) / len(boid_positions)
        
        # Reward for reducing spread (herding boids together)
        herding_reward = 0.0
        if self.previous_boid_spread is not None:
            spread_reduction = self.previous_boid_spread - current_spread
            if spread_reduction > 0:  # Boids got closer together
                herding_reward = spread_reduction * self.herding_multiplier
        
        # Update state for next step
        self.previous_boid_spread = current_spread
        
        return max(0.0, herding_reward)
    
    def _calculate_planning_reward(self, state: Dict[str, Any], action: List[float]) -> float:
        """
        Calculate planning reward - reward strategic positioning and target selection
        """
        boids = state['boids']
        if not boids:
            return 0.0
        
        planning_reward = 0.0
        
        # 1. Smart target selection reward
        # Reward for targeting isolated or slower boids (easier catches)
        closest_boid, closest_dist = self._find_closest_boid(boids)
        if closest_boid and closest_dist > 0.001:
            
            # Calculate boid "vulnerability" (isolation + velocity mismatch)
            vulnerability = self._calculate_boid_vulnerability(closest_boid, boids, state['predator'])
            
            # Reward for targeting more vulnerable boids
            target_selection_reward = vulnerability * self.target_selection_multiplier
            planning_reward += target_selection_reward
        
        # 2. Corner cutting reward - efficient pathing
        # Reward for taking more direct paths vs simple pursuit
        if closest_boid and closest_dist > 0.001:
            # Direct pursuit direction
            direct_dir_x = closest_boid['relX'] / closest_dist
            direct_dir_y = closest_boid['relY'] / closest_dist
            
            # Predicted interception point (simple prediction)
            predicted_x = closest_boid['relX'] + closest_boid['velX'] * 2.0  # 2 steps ahead
            predicted_y = closest_boid['relY'] + closest_boid['velY'] * 2.0
            predicted_dist = max(0.001, math.sqrt(predicted_x**2 + predicted_y**2))
            
            # Interception direction
            intercept_dir_x = predicted_x / predicted_dist
            intercept_dir_y = predicted_y / predicted_dist
            
            # Action alignment with interception vs direct pursuit
            direct_alignment = action[0] * direct_dir_x + action[1] * direct_dir_y
            intercept_alignment = action[0] * intercept_dir_x + action[1] * intercept_dir_y
            
            # Reward for choosing interception over direct pursuit (smarter)
            if intercept_alignment > direct_alignment:
                corner_cutting_reward = (intercept_alignment - direct_alignment) * self.corner_cutting_multiplier
                planning_reward += max(0.0, corner_cutting_reward)
        
        return planning_reward
    
    def _calculate_exploration_bonus(self, state: Dict[str, Any]) -> float:
        """
        Calculate exploration bonus - encourage visiting new areas of the environment
        This prevents getting stuck in local minima
        """
        # Discretize current position for tracking
        # Since predator is at origin in relative coordinates, use boid center as proxy
        boids = state['boids']
        if not boids:
            return 0.0
        
        # Calculate average boid position (approximates predator's area of operation)
        avg_x = sum(boid['relX'] for boid in boids) / len(boids)
        avg_y = sum(boid['relY'] for boid in boids) / len(boids)
        
        # Discretize position to reduce noise
        discrete_x = round(avg_x * 10) / 10  # 0.1 resolution
        discrete_y = round(avg_y * 10) / 10
        current_area = (discrete_x, discrete_y)
        
        # Check if this is a new area
        exploration_bonus = 0.0
        if current_area not in self.visited_positions:
            exploration_bonus = self.exploration_bonus_multiplier
            
            # Add to visited positions
            self.visited_positions.append(current_area)
            
            # Limit memory usage
            if len(self.visited_positions) > self.max_visited_positions:
                self.visited_positions.pop(0)  # Remove oldest
        
        return exploration_bonus
    
    def _find_closest_boid(self, boids: List[Dict[str, Any]]) -> tuple:
        """Find closest boid and its distance"""
        if not boids:
            return None, float('inf')
        
        closest_boid = None
        min_distance = float('inf')
        
        for boid in boids:
            distance = math.sqrt(boid['relX']**2 + boid['relY']**2)
            if distance < min_distance:
                min_distance = distance
                closest_boid = boid
        
        return closest_boid, min_distance
    
    def _calculate_boid_vulnerability(self, 
                                    target_boid: Dict[str, Any], 
                                    all_boids: List[Dict[str, Any]], 
                                    predator: Dict[str, Any]) -> float:
        """
        Calculate how "vulnerable" a boid is (easier to catch)
        
        Factors:
        1. Isolation from other boids (less flocking protection)
        2. Velocity mismatch with predator (easier interception)
        3. Distance from predator (closer = more vulnerable)
        
        Returns:
            Vulnerability score [0, 1] where 1 = most vulnerable
        """
        # 1. Isolation score
        target_x, target_y = target_boid['relX'], target_boid['relY']
        nearby_boids = 0
        min_boid_distance = float('inf')
        
        for other_boid in all_boids:
            if other_boid == target_boid:
                continue
                
            other_x, other_y = other_boid['relX'], other_boid['relY']
            distance = math.sqrt((target_x - other_x)**2 + (target_y - other_y)**2)
            
            if distance < 0.3:  # Close proximity threshold (normalized)
                nearby_boids += 1
            
            min_boid_distance = min(min_boid_distance, distance)
        
        # Isolation score: fewer nearby boids = higher vulnerability
        isolation_score = 1.0 / (1.0 + nearby_boids)
        
        # 2. Velocity mismatch score
        predator_vel_mag = math.sqrt(predator['velX']**2 + predator['velY']**2)
        boid_vel_mag = math.sqrt(target_boid['velX']**2 + target_boid['velY']**2)
        
        # Velocity alignment (negative = moving toward each other)
        if predator_vel_mag > 0.001 and boid_vel_mag > 0.001:
            vel_dot = (predator['velX'] * target_boid['velX'] + 
                      predator['velY'] * target_boid['velY']) / (predator_vel_mag * boid_vel_mag)
            velocity_score = max(0.0, -vel_dot)  # Higher when moving toward each other
        else:
            velocity_score = 0.5
        
        # 3. Distance score
        target_distance = math.sqrt(target_x**2 + target_y**2)
        distance_score = max(0.0, 1.0 - target_distance)  # Closer = higher score
        
        # Combine scores with weights
        vulnerability = (
            0.4 * isolation_score + 
            0.3 * velocity_score + 
            0.3 * distance_score
        )
        
        return min(1.0, max(0.0, vulnerability))
    
    def _empty_reward(self) -> Dict[str, float]:
        """Return empty reward structure"""
        return {
            'total': 0.0,
            'approaching': 0.0,
            'catch': 0.0,
            'herding': 0.0,
            'planning': 0.0,
            'exploration': 0.0
        }
    
    def reset_episode_state(self):
        """Reset episode-specific state tracking"""
        self.previous_boid_spread = None
        self.visited_positions = []
    
    def get_strategic_stats(self) -> Dict[str, Any]:
        """Get statistics about strategic behavior"""
        return {
            'areas_explored': len(self.visited_positions),
            'max_areas': self.max_visited_positions,
            'strategic_weight': self.strategic_weight,
            'herding_enabled': self.enable_herding_rewards,
            'planning_enabled': self.enable_planning_rewards,
            'exploration_enabled': self.enable_exploration_bonus
        }


def create_strategic_reward_processor(**kwargs):
    """Create strategic reward processor with configuration"""
    processor = StrategicRewardProcessor(**kwargs)
    print(f"Created StrategicRewardProcessor with configuration:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")
    return processor


if __name__ == "__main__":
    # Test strategic reward processor
    processor = create_strategic_reward_processor(
        enable_herding_rewards=True,
        enable_planning_rewards=True,
        enable_exploration_bonus=True,
        strategic_weight=0.3
    )
    
    # Test with complex scenario
    test_input = {
        'state': {
            'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
            'predator': {'velX': 0.2, 'velY': -0.1},
            'boids': [
                {'relX': 0.1, 'relY': 0.2, 'velX': 0.3, 'velY': -0.2},   # Close, moving away
                {'relX': 0.3, 'relY': 0.1, 'velX': -0.1, 'velY': 0.4},   # Further, moving perpendicular
                {'relX': 0.15, 'relY': 0.25, 'velX': 0.2, 'velY': -0.1},  # Close to first boid
                {'relX': -0.4, 'relY': -0.2, 'velX': 0.1, 'velY': 0.3}    # Isolated
            ]
        },
        'action': [0.6, -0.3],  # Strategic action toward intercept
        'caughtBoids': []
    }
    
    reward_result = processor.calculate_step_reward(test_input)
    
    print(f"\nTest results:")
    print(f"  Total: {reward_result['total']:.4f}")
    print(f"  Base approaching: {reward_result['approaching']:.4f}")
    print(f"  Base catch: {reward_result['catch']:.4f}")
    print(f"  Strategic herding: {reward_result['herding']:.4f}")
    print(f"  Strategic planning: {reward_result['planning']:.4f}")
    print(f"  Exploration bonus: {reward_result['exploration']:.4f}")
    
    stats = processor.get_strategic_stats()
    print(f"\nStrategic stats: {stats}")