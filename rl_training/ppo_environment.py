"""
PPO Environment - Gym wrapper for the existing boids simulation system

This environment wraps our existing simulation components to create a Gym-compatible
interface for PPO training while maintaining full compatibility with the existing
evaluation system.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Any, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.state_manager import StateManager
from simulation.random_state_generator import RandomStateGenerator
from simulation.processors import ActionProcessor
from rewards.reward_processor import RewardProcessor
from config.constants import CONSTANTS

class BoidsEnvironment(gym.Env):
    """
    Gym environment wrapper for the boids predator-prey simulation
    
    Observation Space: Variable-length structured input (handled by transformer)
    Action Space: Continuous 2D actions in [-1, 1] range
    Reward: Instant reward from existing reward system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # Environment configuration
        self.config = config or {}
        self.canvas_width = self.config.get('canvas_width', 800)
        self.canvas_height = self.config.get('canvas_height', 600)
        self.initial_boids = self.config.get('initial_boids', 20)
        self.max_steps = self.config.get('max_steps', 1500)
        self.seed_value = self.config.get('seed', None)
        
        # Action space: 2D continuous actions in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # Observation space: Will be structured input, but we define bounds for SB3
        # Note: The actual observation will be the structured dict format
        max_boids = self.config.get('max_boids', 50)
        
        # Define observation space bounds (for SB3 compatibility)
        # Structure: [canvas_width, canvas_height, pred_velX, pred_velY, boid1_relX, boid1_relY, boid1_velX, boid1_velY, ...]
        obs_size = 4 + max_boids * 4  # context(2) + predator(2) + boids(4 each)
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Initialize simulation components (matching existing evaluation system)
        self.state_manager = StateManager()
        self.state_generator = RandomStateGenerator(seed=self.seed_value)
        self.action_processor = ActionProcessor()
        self.reward_processor = RewardProcessor()
        
        # Environment state
        self.current_step = 0
        self.initial_boid_count = 0
        self.total_reward = 0.0
        self.episode_catches = 0
        self.current_policy_action = None
        
        print(f"BoidsEnvironment initialized:")
        print(f"  Canvas: {self.canvas_width}×{self.canvas_height}")
        print(f"  Initial boids: {self.initial_boids}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Action space: {self.action_space}")
        print(f"  Observation space: {self.observation_space}")
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, Any], Dict]:
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Structured observation dict
            info: Info dict
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed_value = seed
            self.state_generator = RandomStateGenerator(seed=seed)
        
        # Generate random initial state
        initial_state = self.state_generator.generate_scattered_state(
            self.initial_boids, self.canvas_width, self.canvas_height
        )
        
        # Initialize state manager with environment as policy
        self.state_manager.init(initial_state, self)
        
        # Reset episode tracking
        self.current_step = 0
        self.initial_boid_count = len(initial_state['boids_states'])
        self.total_reward = 0.0
        self.episode_catches = 0
        self.current_policy_action = None
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: 2D action array in [-1, 1] range
            
        Returns:
            observation, reward, done, info
        """
        self.current_step += 1
        
        # Store raw action for StateManager (ActionProcessor will be called by StateManager)
        # This matches the direct simulation where policy returns raw action
        self.current_policy_action = action.tolist()
        
        # Execute simulation step
        step_result = self.state_manager.step()
        
        # Calculate reward using existing reward system (use original unprocessed action)
        reward_input = {
            'state': self._get_structured_inputs(),
            'action': action.tolist(),  # Use original action for reward calculation
            'caughtBoids': step_result.get('caught_boids', [])
        }
        
        reward_output = self.reward_processor.calculate_step_reward(reward_input)
        reward = reward_output['total']
        
        # Update episode tracking
        self.total_reward += reward
        self.episode_catches += len(step_result.get('caught_boids', []))
        
        # Check termination conditions
        done = self._is_done(step_result)
        
        # Get next observation
        observation = self._get_observation()
        
        # Info dict for logging
        info = {
            'step': self.current_step,
            'total_reward': self.total_reward,
            'episode_catches': self.episode_catches,
            'boids_remaining': len(step_result['boids_states']),
            'reward_components': reward_output
        }
        
        return observation, reward, done, False, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation in structured format"""
        current_state = self.state_manager.get_state()
        structured_inputs = self.state_manager._convert_state_to_structured_inputs(current_state)
        return structured_inputs
    
    def _get_structured_inputs(self) -> Dict[str, Any]:
        """Get structured inputs for reward calculation"""
        return self._get_observation()
    
    def _is_done(self, step_result: Dict) -> bool:
        """Check if episode should terminate"""
        # Episode ends if:
        # 1. Max steps reached
        # 2. All boids caught
        # 3. No boids remaining
        
        if self.current_step >= self.max_steps:
            return True
        
        if len(step_result['boids_states']) == 0:
            return True
        
        return False
    
    def render(self, mode='human'):
        """Optional rendering (not implemented)"""
        pass
    
    def get_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Policy interface for StateManager - returns current action from PPO
        
        Args:
            structured_inputs: Structured inputs (ignored, we use stored action)
            
        Returns:
            Current action from PPO step
        """
        if self.current_policy_action is None:
            # Should not happen during normal operation
            raise RuntimeError("No action available - step() must be called first")
        return self.current_policy_action
    
    def close(self):
        """Clean up environment"""
        pass
    
    def seed(self, seed=None):
        """Set random seed"""
        self.seed_value = seed
        self.state_generator = RandomStateGenerator(seed=seed)
        return [seed]

def create_boids_environment(config: Optional[Dict] = None) -> BoidsEnvironment:
    """Factory function to create boids environment"""
    env = BoidsEnvironment(config)
    print(f"✓ Created BoidsEnvironment with config: {config}")
    return env

if __name__ == "__main__":
    # Test environment
    env = create_boids_environment({
        'canvas_width': 800,
        'canvas_height': 600,
        'initial_boids': 10,
        'max_steps': 100
    })
    
    obs = env.reset()
    print(f"Initial observation structure: {list(obs.keys())}")
    print(f"Boids count: {len(obs['boids'])}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, done={done}, boids={len(obs['boids'])}")
        
        if done:
            break
    
    print("✓ Environment test completed")