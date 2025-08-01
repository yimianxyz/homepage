"""
Boid Environment - Gym wrapper for the boid simulation system

This environment wraps the existing StateManager and RewardProcessor to create
a Gym-compatible environment for RL training. It handles episode management,
reward calculation, and observation/action space conversion.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulation.state_manager import StateManager
from simulation.random_state_generator import RandomStateGenerator
from rewards.reward_processor import RewardProcessor
from config.constants import CONSTANTS


class BoidEnvironment(gym.Env):
    """
    Gym environment wrapper for boid simulation
    
    This environment provides a standard RL interface for training predator policies
    to catch boids in a flocking simulation. It integrates with the existing
    simulation infrastructure while providing Gym-compatible interfaces.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 num_boids: int = 20,
                 canvas_width: float = 800,
                 canvas_height: float = 600,
                 max_steps: int = 1000,
                 seed: Optional[int] = None):
        """
        Initialize the boid environment
        
        Args:
            num_boids: Number of boids in the simulation
            canvas_width: Simulation canvas width
            canvas_height: Simulation canvas height
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Environment parameters
        self.num_boids = num_boids
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.max_steps = max_steps
        self.seed_value = seed
        
        # Initialize core components
        self.state_manager = StateManager()
        self.reward_processor = RewardProcessor()
        self.state_generator = RandomStateGenerator(seed=seed)
        
        # Episode tracking
        self.current_step = 0
        self.total_reward = 0
        self.boids_caught = 0
        self.episode_count = 0
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # State tracking
        self.last_structured_inputs = None
        self.last_action = None
        
        print(f"Created BoidEnvironment:")
        print(f"  Boids: {self.num_boids}")
        print(f"  Canvas: {self.canvas_width}x{self.canvas_height}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Observation space: {self.observation_space}")
        print(f"  Action space: {self.action_space}")
    
    def _setup_spaces(self):
        """Setup Gym observation and action spaces"""
        
        # Action space: normalized policy outputs in [-1, 1] range
        # ActionProcessor will scale these to game forces
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # Observation space: flattened structured inputs
        # Format: [context(2) + predator(2) + boids(4*max_boids)]
        # We need to define a maximum number of boids for fixed observation size
        self.max_obs_boids = self.num_boids  # Start with initial number
        
        obs_size = (
            2 +  # context: [canvasWidth, canvasHeight] 
            2 +  # predator: [velX, velY]
            4 * self.max_obs_boids  # boids: [relX, relY, velX, velY] * max_boids
        )
        
        # All values are normalized to [-1, 1] or [0, 1] ranges
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode
        
        Args:
            seed: Optional seed for this episode
            options: Optional reset options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Set seed if provided
        if seed is not None:
            self.seed_value = seed
            self.state_generator = RandomStateGenerator(seed=seed)
        
        # Generate initial state
        initial_state = self.state_generator.generate_random_state(
            self.num_boids, self.canvas_width, self.canvas_height
        )
        
        # Create a dummy policy for state manager initialization
        # We'll replace the policy's get_action method in step()
        dummy_policy = DummyPolicy()
        
        # Initialize state manager
        self.state_manager.init(initial_state, dummy_policy)
        
        # Reset episode tracking
        self.current_step = 0
        self.total_reward = 0
        self.boids_caught = 0
        self.episode_count += 1
        
        # Get initial observation
        current_state = self.state_manager.get_state()
        observation = self._state_to_observation(current_state)
        
        # Reset state tracking
        self.last_structured_inputs = None
        self.last_action = None
        
        info = {
            'episode': self.episode_count,
            'boids_remaining': len(current_state['boids_states']),
            'total_boids': self.num_boids
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Action from the policy [x, y] in [-1, 1] range
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Validate and convert action to proper format
        if np.isscalar(action):
            # Handle scalar input by converting to 2D array
            action = np.array([action, 0.0], dtype=np.float32)
        else:
            # Ensure action is numpy array
            action = np.asarray(action, dtype=np.float32)
        
        # Ensure action has correct shape
        if action.shape == ():
            action = np.array([float(action), 0.0], dtype=np.float32)
        elif len(action) == 1:
            action = np.array([float(action[0]), 0.0], dtype=np.float32)
        elif len(action) > 2:
            action = action[:2]  # Take first 2 elements
        
        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Get current state for reward calculation
        current_state = self.state_manager.get_state()
        structured_inputs = self.state_manager._convert_state_to_structured_inputs(current_state)
        
        # Replace dummy policy's action with our action
        self.state_manager.policy._last_action = action.tolist()
        
        # Execute simulation step
        try:
            step_result = self.state_manager.step()
            caught_boids = step_result.get('caught_boids', [])
            
            # Calculate reward using RewardProcessor
            # Ensure action is properly formatted as list
            action_list = action.tolist() if hasattr(action, 'tolist') else list(action)
            step_input = {
                'state': structured_inputs,
                'action': action_list,
                'caughtBoids': caught_boids
            }
            reward_result = self.reward_processor.calculate_step_reward(step_input)
            reward = reward_result['total']
            
        except Exception as e:
            print(f"Error in step: {e}")
            reward = 0.0
            step_result = current_state
            caught_boids = []
            reward_result = {'total': 0.0, 'approaching': 0.0, 'catch': 0.0}
        
        # Update tracking
        self.current_step += 1
        self.total_reward += reward
        self.boids_caught += len(caught_boids)
        
        # Get next observation
        observation = self._state_to_observation(step_result)
        
        # Check termination conditions
        boids_remaining = len(step_result['boids_states'])
        terminated = boids_remaining == 0  # All boids caught
        truncated = self.current_step >= self.max_steps  # Max steps reached
        
        # Store state for next reward calculation
        self.last_structured_inputs = structured_inputs
        self.last_action = action.tolist()
        
        info = {
            'step': self.current_step,
            'total_reward': self.total_reward,
            'boids_remaining': boids_remaining,
            'boids_caught_this_step': len(caught_boids),
            'total_boids_caught': self.boids_caught,
            'reward_breakdown': reward_result
        }
        
        return observation, reward, terminated, truncated, info
    
    def _state_to_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Convert simulation state to observation vector
        
        Args:
            state: Simulation state from StateManager
            
        Returns:
            observation: Flattened observation vector
        """
        # Get structured inputs from state manager
        structured_inputs = self.state_manager._convert_state_to_structured_inputs(state)
        
        # Build observation vector
        obs = []
        
        # Context: [canvasWidth, canvasHeight] (already normalized)
        obs.extend([
            structured_inputs['context']['canvasWidth'],
            structured_inputs['context']['canvasHeight']
        ])
        
        # Predator: [velX, velY] (already normalized)
        obs.extend([
            structured_inputs['predator']['velX'],
            structured_inputs['predator']['velY']
        ])
        
        # Boids: [relX, relY, velX, velY] for each boid (already normalized)
        boids = structured_inputs['boids']
        for i in range(self.max_obs_boids):
            if i < len(boids):
                boid = boids[i]
                obs.extend([boid['relX'], boid['relY'], boid['velX'], boid['velY']])
            else:
                # Pad with zeros for missing boids
                obs.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(obs, dtype=np.float32)
    
    def render(self, mode='human'):
        """Render the environment (placeholder for now)"""
        if mode == 'human':
            current_state = self.state_manager.get_state()
            print(f"Step {self.current_step}: {len(current_state['boids_states'])} boids remaining")
    
    def close(self):
        """Clean up resources"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get detailed environment information"""
        current_state = self.state_manager.get_state()
        return {
            'environment': 'BoidEnvironment',
            'episode': self.episode_count,
            'step': self.current_step,
            'max_steps': self.max_steps,
            'boids_remaining': len(current_state['boids_states']),
            'total_boids': self.num_boids,
            'boids_caught': self.boids_caught,
            'total_reward': self.total_reward,
            'canvas_size': [self.canvas_width, self.canvas_height]
        }


class DummyPolicy:
    """
    Dummy policy for StateManager initialization
    The actual action will be provided externally in step()
    """
    
    def __init__(self):
        self._last_action = [0.0, 0.0]
    
    def get_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """Return the last action set externally"""
        return self._last_action