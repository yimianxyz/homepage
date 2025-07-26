"""
RL Environment - Wrapper for boids predator-prey simulation

This environment wraps the existing simulation system to provide a standard
RL interface (similar to OpenAI Gym) while using all existing components:
- Simulation runtime and state management
- Input/output processors
- Reward processor with approaching + catch retro rewards
- Random state generation

The environment supports both single episodes and batch environments for efficiency.
"""

import torch
import numpy as np
import sys
import os
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import copy
from collections import deque
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.constants import CONSTANTS
from simulation.state_manager import StateManager
from simulation.random_state_generator import RandomStateGenerator
from rewards.reward_processor import RewardProcessor

class BoidsEnvironment:
    """
    RL Environment for boids predator-prey simulation
    
    This environment provides a standard RL interface while using all existing
    simulation components. It handles episode management, reward calculation,
    and state transitions.
    
    Features:
    - Compatible with existing simulation system
    - Supports variable episode lengths
    - Uses sophisticated reward system with retro rewards
    - Extensive debugging and monitoring
    - Batch environment support for efficiency
    """
    
    def __init__(self, 
                 num_boids_range: Tuple[int, int] = (10, 50),
                 canvas_size_range: Tuple[Tuple[int, int], Tuple[int, int]] = ((800, 600), (1920, 1080)),
                 max_episode_steps: int = 1000,
                 reward_lookahead_steps: int = 50,
                 state_type: str = 'scattered',  # 'scattered' or 'clustered'
                 seed: Optional[int] = None,
                 debug: bool = True):
        """
        Initialize the boids RL environment
        
        Args:
            num_boids_range: (min, max) number of boids to spawn
            canvas_size_range: ((min_w, min_h), (max_w, max_h)) canvas dimensions
            max_episode_steps: Maximum steps per episode
            reward_lookahead_steps: Steps to look ahead for retro rewards
            state_type: Type of initial state ('scattered' or 'clustered')
            seed: Random seed for reproducibility
            debug: Enable debug logging
        """
        self.num_boids_range = num_boids_range
        self.canvas_size_range = canvas_size_range
        self.max_episode_steps = max_episode_steps
        self.reward_lookahead_steps = reward_lookahead_steps
        self.state_type = state_type
        self.seed = seed
        self.debug = debug
        
        if self.debug:
            print(f"🌍 Initializing BoidsEnvironment:")
            print(f"   Boids range: {num_boids_range}")
            print(f"   Canvas range: {canvas_size_range}")
            print(f"   Max episode steps: {max_episode_steps}")
            print(f"   Reward lookahead: {reward_lookahead_steps}")
            print(f"   State type: {state_type}")
            print(f"   Seed: {seed}")
        
        # Initialize components
        self.state_generator = RandomStateGenerator(seed)
        self.state_manager = StateManager()
        self.reward_processor = RewardProcessor()
        
        # Use original MAX_RETRO_REWARD_STEPS from config for proper long-range learning
        if debug:
            print(f"🔧 RL Environment: Using MAX_RETRO_REWARD_STEPS={self.reward_processor.max_retro_steps} for long-range behavior learning")
        
        # Episode state
        self.current_episode_step = 0
        self.episode_reward_history = []  # Store reward inputs for retro calculation
        self.episode_states = []  # Store states for debugging
        self.total_episodes = 0
        self.total_steps = 0
        
        # Episode statistics
        self.episode_stats = {
            'total_reward': 0.0,
            'approaching_reward': 0.0,
            'catch_reward': 0.0,
            'boids_caught': 0,
            'boids_remaining': 0,
            'steps': 0
        }
        
        # Current state
        self.current_structured_input = None
        self.is_episode_active = False
        
        if self.debug:
            print(f"   ✅ Environment initialized successfully")

    def reset(self, **kwargs) -> Dict[str, Any]:
        """
        Reset environment and start new episode
        
        Args:
            **kwargs: Override parameters (num_boids, canvas_width, canvas_height)
            
        Returns:
            Initial structured input for the policy
        """
        if self.debug:
            print(f"\n🔄 Resetting environment (episode {self.total_episodes + 1})")
        
        # Determine episode parameters
        num_boids = kwargs.get('num_boids', 
                              np.random.randint(self.num_boids_range[0], self.num_boids_range[1] + 1))
        
        # Handle both pixel coordinates and normalized coordinates
        min_width, max_width = self.canvas_size_range[0][0], self.canvas_size_range[1][0]
        min_height, max_height = self.canvas_size_range[0][1], self.canvas_size_range[1][1]
        
        # If values are < 1, assume normalized coordinates and scale to pixels
        scale_factor = 1000 if max_width < 1 and max_height < 1 else 1
        
        canvas_width = kwargs.get('canvas_width',
                                 np.random.uniform(min_width, max_width) * scale_factor)
        canvas_height = kwargs.get('canvas_height', 
                                  np.random.uniform(min_height, max_height) * scale_factor)
        
        if self.debug:
            print(f"   Episode params: {num_boids} boids, {canvas_width}x{canvas_height} canvas")
        
        # Generate initial state
        if self.state_type == 'clustered':
            initial_state = self.state_generator.generate_clustered_state(num_boids, canvas_width, canvas_height)
        else:
            initial_state = self.state_generator.generate_scattered_state(num_boids, canvas_width, canvas_height)
        
        # Create dummy policy for state manager initialization (we'll replace with actual policy)
        class DummyPolicy:
            def get_action(self, structured_inputs):
                return [0.0, 0.0]  # Will be replaced by actual policy
        
        # Initialize state manager
        self.state_manager.init(initial_state, DummyPolicy())
        
        # Reset episode tracking
        self.current_episode_step = 0
        self.episode_reward_history = []
        self.episode_states = []
        self.episode_stats = {
            'boids_caught': 0,
            'boids_remaining': len(initial_state['boids_states']),
            'steps': 0
        }
        self.is_episode_active = True
        
        # Get initial structured input
        current_state = self.state_manager.get_state()
        self.current_structured_input = self.state_manager._convert_state_to_structured_inputs(current_state)
        
        if self.debug:
            print(f"   ✅ Reset complete - {len(self.current_structured_input['boids'])} boids active")
        
        return copy.deepcopy(self.current_structured_input)

    def step(self, action: Union[torch.Tensor, np.ndarray, List[float]]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one environment step
        
        Args:
            action: Policy action [x, y] in [-1, 1] range
            
        Returns:
            Tuple of (next_observation, reward, done, info)
            Note: reward is always 0.0 - rewards calculated after complete episodes
        """
        if not self.is_episode_active:
            raise RuntimeError("Environment not active. Call reset() first.")
        
        # Convert action to list if needed
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy().tolist()
        elif isinstance(action, np.ndarray):
            action = action.tolist()
        
        # Store current state and action for reward calculation
        current_state = self.state_manager.get_state()
        
        if self.debug and self.current_episode_step < 10:  # Debug first 10 steps
            print(f"\n🌍 Environment Step {self.current_episode_step + 1}:")
            print(f"   Action received: [{action[0]:.3f}, {action[1]:.3f}]")
            print(f"   Boids remaining: {len(current_state['boids_states'])}")
        
        current_structured_input = copy.deepcopy(self.current_structured_input)
        
        # Update state manager policy with current action
        class ActionPolicy:
            def __init__(self, action):
                self.action = action
            def get_action(self, structured_inputs):
                return self.action
        
        self.state_manager.policy = ActionPolicy(action)
        
        # Execute simulation step
        step_result = self.state_manager.step()
        
        # Update structured input for next step
        self.current_structured_input = self.state_manager._convert_state_to_structured_inputs(step_result)
        
        # Store reward input for later reward calculation (no immediate calculation!)
        reward_input = {
            'caughtBoids': step_result.get('caught_boids', []),
            'state': current_structured_input,
            'action': action
        }
        self.episode_reward_history.append(reward_input)
        
        # Update episode statistics (no reward calculation yet)
        caught_count = len(step_result.get('caught_boids', []))
        
        if self.debug and self.current_episode_step < 10:
            if caught_count > 0:
                print(f"   🎯 BOIDS CAUGHT: {caught_count}!")
            print(f"   ─" * 40)
        
        self.episode_stats['boids_caught'] += caught_count
        self.episode_stats['boids_remaining'] = len(step_result['boids_states'])
        self.episode_stats['steps'] = self.current_episode_step + 1
        
        # Check episode termination conditions
        self.current_episode_step += 1
        self.total_steps += 1
        
        boids_remaining = len(step_result['boids_states'])
        max_steps_reached = self.current_episode_step >= self.max_episode_steps
        
        # Clear termination logic
        if boids_remaining == 0:
            # SUCCESS: All boids caught
            done = True
            episode_end_type = 'success'
            episode_end = True  # Full success - use full retroactive rewards
        elif max_steps_reached:
            # TIMEOUT: Max steps reached with boids remaining
            done = True
            episode_end_type = 'timeout'
            episode_end = False  # Partial episode - limited retroactive rewards
        else:
            # CONTINUING: Episode still active
            done = False
            episode_end_type = 'continuing'
            episode_end = False
        
        if done:
            self.is_episode_active = False
            self.total_episodes += 1
            
            if self.debug:
                success_rate = self.episode_stats['boids_caught'] / (
                    self.episode_stats['boids_caught'] + self.episode_stats['boids_remaining']
                ) * 100 if (self.episode_stats['boids_caught'] + self.episode_stats['boids_remaining']) > 0 else 0
                
                print(f"\n🏁 Episode {self.total_episodes} Complete ({episode_end_type.upper()})!")
                print(f"   Duration: {self.episode_stats['steps']} steps")
                print(f"   Boids caught: {self.episode_stats['boids_caught']} / {self.episode_stats['boids_caught'] + self.episode_stats['boids_remaining']}")
                print(f"   Success rate: {success_rate:.1f}%")
                print(f"   Episode end flag: {episode_end} ({'full rewards' if episode_end else 'limited rewards'})")
                print(f"   Reward history length: {len(self.episode_reward_history)}")
                print(f"   ═" * 50)
        
        # Create info dict with clear episode information
        info = {
            'episode_step': self.current_episode_step,
            'boids_caught_this_step': caught_count,
            'boids_remaining': boids_remaining,
            'episode_end_type': episode_end_type,
            'episode_end': episode_end,  # Key flag for reward calculation!
            'episode_stats': copy.deepcopy(self.episode_stats),
            'reward_history': copy.deepcopy(self.episode_reward_history),  # Provide access to full history
            'caught_boids': step_result.get('caught_boids', [])  # For reward calculation
        }
        
        # Always return 0.0 reward - rewards calculated after complete episodes
        return copy.deepcopy(self.current_structured_input), 0.0, done, info
    
    def get_reward_history(self) -> List[Dict[str, Any]]:
        """Get current episode's reward history for rollout recalculation"""
        return copy.deepcopy(self.episode_reward_history)



    def get_observation_space_info(self) -> Dict[str, Any]:
        """Get information about the observation space"""
        return {
            'type': 'structured_input',
            'context_dims': 2,  # canvas_width, canvas_height
            'predator_dims': 2,  # velX, velY
            'boid_dims': 4,  # relX, relY, velX, velY
            'max_boids': self.num_boids_range[1],
            'value_ranges': {
                'context': '(0, 1) - normalized canvas dimensions',
                'predator_velocity': '(-1, 1) - normalized velocity',
                'boid_relative_position': '(-1, 1) - normalized relative position',
                'boid_velocity': '(-1, 1) - normalized velocity'
            }
        }

    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about the action space"""
        return {
            'type': 'continuous',
            'dims': 2,  # [force_x, force_y]
            'range': '(-1, 1) - normalized forces',
            'description': 'Predator steering forces in normalized coordinates'
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics"""
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'current_episode_step': self.current_episode_step,
            'is_episode_active': self.is_episode_active,
            'last_episode_stats': copy.deepcopy(self.episode_stats)
        }

    def render(self, mode: str = 'text') -> Optional[str]:
        """Render environment state (text mode only)"""
        if mode != 'text':
            print(f"⚠️  Render mode '{mode}' not supported, using 'text'")
        
        if not self.current_structured_input:
            return "Environment not initialized"
        
        state = self.current_structured_input
        lines = [
            f"🎮 Boids Environment - Step {self.current_episode_step}",
            f"   Predator vel: ({state['predator']['velX']:.3f}, {state['predator']['velY']:.3f})",
            f"   Boids active: {len(state['boids'])}",
            f"   Canvas: ({state['context']['canvasWidth']:.3f}, {state['context']['canvasHeight']:.3f})",
            f"   Episode reward: {self.episode_stats['total_reward']:.3f}",
            f"   Boids caught: {self.episode_stats['boids_caught']}"
        ]
        
        result = '\n'.join(lines)
        print(result)
        return result

    def close(self):
        """Clean up environment resources"""
        if self.debug:
            print(f"🛑 Closing environment after {self.total_episodes} episodes, {self.total_steps} steps")
        
        self.is_episode_active = False


class BatchBoidsEnvironment:
    """
    Batch environment wrapper for parallel episode execution
    
    This runs multiple environments in parallel for more efficient training.
    """
    
    def __init__(self, 
                 num_envs: int = 4,
                 env_kwargs: Optional[Dict[str, Any]] = None,
                 debug: bool = True):
        """
        Initialize batch environment
        
        Args:
            num_envs: Number of parallel environments
            env_kwargs: Keyword arguments for each environment
            debug: Enable debug logging
        """
        self.num_envs = num_envs
        self.debug = debug
        
        if env_kwargs is None:
            env_kwargs = {}
        
        if self.debug:
            print(f"🏭 Creating batch environment with {num_envs} parallel envs")
        
        # Create environments with different seeds for diversity
        self.envs = []
        for i in range(num_envs):
            env_kwargs_copy = env_kwargs.copy()
            if 'seed' not in env_kwargs_copy:
                env_kwargs_copy['seed'] = i * 1000  # Different seeds
            env_kwargs_copy['debug'] = False  # Disable individual env debug
            
            env = BoidsEnvironment(**env_kwargs_copy)
            self.envs.append(env)
        
        if self.debug:
            print(f"   ✅ Batch environment created")

    def reset(self) -> List[Dict[str, Any]]:
        """Reset all environments"""
        observations = []
        for env in self.envs:
            obs = env.reset()
            observations.append(obs)
        return observations

    def step(self, actions: List[Union[torch.Tensor, np.ndarray, List[float]]]) -> Tuple[List[Dict[str, Any]], List[float], List[bool], List[Dict[str, Any]]]:
        """Step all environments - no auto-reset, handled by trainer"""
        observations, rewards, dones, infos = [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return observations, rewards, dones, infos

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all environments"""
        all_stats = []
        for i, env in enumerate(self.envs):
            stats = env.get_stats()
            stats['env_id'] = i
            all_stats.append(stats)
        
        return {
            'num_envs': self.num_envs,
            'total_episodes': sum(s['total_episodes'] for s in all_stats),
            'total_steps': sum(s['total_steps'] for s in all_stats),
            'env_stats': all_stats
        }



    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()


if __name__ == "__main__":
    # Test the environment
    print("🧪 Testing BoidsEnvironment...")
    
    # Test single environment
    env = BoidsEnvironment(
        num_boids_range=(5, 10),
        canvas_size_range=((400, 300), (800, 600)),
        max_episode_steps=20,  # Short for testing
        debug=True
    )
    
    print("\n🔍 Testing single environment...")
    obs = env.reset()
    print(f"   Initial obs: {len(obs['boids'])} boids")
    
    # Run a few steps
    for step in range(5):
        action = [0.5, -0.3]  # Test action
        obs, reward, done, info = env.step(action)
        print(f"   Step {step}: reward={reward:.3f}, done={done}, boids={len(obs['boids'])}")
        
        if done:
            print("   Episode finished early")
            break
    
    env.close()
    
    print("\n🔍 Testing batch environment...")
    batch_env = BatchBoidsEnvironment(
        num_envs=2,
        env_kwargs={
            'num_boids_range': (3, 6),
            'max_episode_steps': 10
        },
        debug=True
    )
    
    batch_obs = batch_env.reset()
    print(f"   Batch reset: {len(batch_obs)} environments")
    
    # Test batch step
    batch_actions = [[0.2, 0.4], [-0.1, 0.8]]
    batch_obs, batch_rewards, batch_dones, batch_infos = batch_env.step(batch_actions)
    print(f"   Batch step: rewards={batch_rewards}, dones={batch_dones}")
    
    batch_env.close()
    
    print("\n✅ BoidsEnvironment tests passed!") 