"""
PPO Experience Buffer - Simple rollout collection for PPO training

This module provides a clean interface for collecting and processing experience
from the simulation environment for PPO training. It integrates seamlessly with
StateManager and RewardProcessor.

Features:
- Stores rollout trajectories (states, actions, rewards, values, log_probs)
- Computes advantages using GAE (Generalized Advantage Estimation)
- Provides batched data for PPO training
- Simple, efficient implementation focused on single environment
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class PPOExperience:
    """Single step experience for PPO"""
    structured_input: Dict[str, Any]  # State in policy format
    action: torch.Tensor              # Action taken [x, y]
    log_prob: torch.Tensor           # Log probability of action
    value: torch.Tensor              # Value estimate
    reward: float                    # Reward received
    done: bool                       # Episode termination flag


class PPOExperienceBuffer:
    """
    Experience buffer for PPO rollout collection
    
    Collects experiences from environment interaction and processes them
    for PPO training with GAE advantage computation.
    """
    
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Initialize experience buffer
        
        Args:
            gamma: Discount factor for future rewards
            gae_lambda: GAE lambda parameter for advantage estimation
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Storage for current rollout
        self.experiences: List[PPOExperience] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        
        # Current episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        print(f"PPO Experience Buffer initialized:")
        print(f"  Discount factor (gamma): {self.gamma}")
        print(f"  GAE lambda: {self.gae_lambda}")
    
    def add_experience(self, experience: PPOExperience):
        """
        Add a single experience to the buffer
        
        Args:
            experience: PPOExperience object
        """
        self.experiences.append(experience)
        self.current_episode_reward += experience.reward
        self.current_episode_length += 1
        
        # Track episode completion
        if experience.done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
    
    def compute_advantages_and_returns(self, next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using GAE
        
        Args:
            next_value: Value estimate for the state after the last experience
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns (targets for value function)
        """
        if len(self.experiences) == 0:
            return torch.tensor([]), torch.tensor([])
        
        # Extract values and rewards
        values = torch.stack([exp.value for exp in self.experiences])
        rewards = torch.tensor([exp.reward for exp in self.experiences], dtype=torch.float32)
        dones = torch.tensor([exp.done for exp in self.experiences], dtype=torch.bool)
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        advantages_list = []
        
        # Bootstrap value (value of next state)
        next_value = torch.tensor(next_value, dtype=torch.float32)
        
        # Work backwards through the trajectory
        last_advantage = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_values = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            
            # GAE advantage
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            advantages_list.insert(0, last_advantage)
        
        advantages = torch.tensor(advantages_list, dtype=torch.float32)
        
        # Compute returns (advantages + values)
        returns = advantages + values
        
        return advantages, returns
    
    def get_batch_data(self) -> Dict[str, Any]:
        """
        Get all experiences as batched data for training
        
        Returns:
            Dictionary with batched experiences
        """
        if len(self.experiences) == 0:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages_and_returns()
        
        # Normalize advantages (standard PPO practice)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'structured_inputs': [exp.structured_input for exp in self.experiences],
            'actions': torch.stack([exp.action for exp in self.experiences]),
            'log_probs': torch.stack([exp.log_prob for exp in self.experiences]),
            'values': torch.stack([exp.value for exp in self.experiences]),
            'advantages': advantages,
            'returns': returns,
            'rewards': torch.tensor([exp.reward for exp in self.experiences], dtype=torch.float32)
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get rollout statistics for logging
        
        Returns:
            Dictionary with rollout statistics
        """
        if len(self.experiences) == 0:
            return {}
        
        rewards = [exp.reward for exp in self.experiences]
        
        stats = {
            'rollout_length': len(self.experiences),
            'mean_reward': np.mean(rewards),
            'total_reward': sum(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
        }
        
        if self.episode_rewards:
            stats.update({
                'episode_count': len(self.episode_rewards),
                'mean_episode_reward': np.mean(self.episode_rewards),
                'mean_episode_length': np.mean(self.episode_lengths),
                'best_episode_reward': max(self.episode_rewards),
            })
        
        return stats
    
    def clear(self):
        """Clear all stored experiences"""
        self.experiences.clear()
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def __len__(self):
        """Return number of experiences stored"""
        return len(self.experiences)


class PPORolloutCollector:
    """
    High-level interface for collecting PPO rollouts using existing simulation infrastructure
    
    Integrates StateManager, RewardProcessor, and PPO policy for seamless experience collection.
    """
    
    def __init__(self, 
                 state_manager,
                 reward_processor, 
                 ppo_policy,
                 max_episode_steps: int = 1000):
        """
        Initialize rollout collector
        
        Args:
            state_manager: StateManager instance
            reward_processor: RewardProcessor instance  
            ppo_policy: PPOTransformerPolicy instance
            max_episode_steps: Maximum steps per episode
        """
        self.state_manager = state_manager
        self.reward_processor = reward_processor
        self.ppo_policy = ppo_policy
        self.max_episode_steps = max_episode_steps
        
        self.current_step = 0
        
        print(f"PPO Rollout Collector initialized:")
        print(f"  Max episode steps: {max_episode_steps}")
        print(f"  Using existing StateManager and RewardProcessor")
    
    def collect_rollout(self, 
                       initial_state: Dict[str, Any], 
                       rollout_steps: int) -> PPOExperienceBuffer:
        """
        Collect a rollout of experiences
        
        Args:
            initial_state: Initial simulation state
            rollout_steps: Number of steps to collect
            
        Returns:
            PPOExperienceBuffer with collected experiences
        """
        buffer = PPOExperienceBuffer()
        
        # Initialize environment
        self.state_manager.init(initial_state, self.ppo_policy)
        self.current_step = 0
        
        # Set policy to training mode for stochastic actions
        self.ppo_policy.train()
        
        print(f"Collecting rollout of {rollout_steps} steps...")
        
        for step in range(rollout_steps):
            # Get current state in structured format
            current_state = self.state_manager.get_state()
            structured_input = self.state_manager._convert_state_to_structured_inputs(current_state)
            
            # Get action and value from policy
            action, log_prob, value = self.ppo_policy.get_action_and_value(structured_input)
            
            # Take action in environment
            step_result = self.state_manager.step()
            
            # Calculate reward
            reward_input = {
                'state': structured_input,
                'action': action.detach().cpu().numpy().tolist(),
                'caughtBoids': step_result.get('caught_boids', [])
            }
            reward_data = self.reward_processor.calculate_step_reward(reward_input)
            reward = reward_data['total']
            
            # Check if episode is done
            done = (len(step_result['boids_states']) == 0 or  # All boids caught
                   self.current_step >= self.max_episode_steps)  # Max steps reached
            
            # Create experience
            experience = PPOExperience(
                structured_input=structured_input,
                action=action.detach(),
                log_prob=log_prob.detach(),
                value=value.detach(),
                reward=reward,
                done=done
            )
            
            # Add to buffer
            buffer.add_experience(experience)
            
            self.current_step += 1
            
            # Print progress
            if step % 100 == 0:
                print(f"  Step {step}/{rollout_steps}, reward: {reward:.3f}, boids: {len(step_result['boids_states'])}")
            
            # Reset if episode done
            if done:
                print(f"  Episode finished at step {step}, total reward: {buffer.current_episode_reward:.2f}")
                if step < rollout_steps - 1:  # If not the last step
                    # Reset environment for new episode
                    self.state_manager.init(initial_state, self.ppo_policy)
                    self.current_step = 0
        
        stats = buffer.get_statistics()
        print(f"✅ Rollout complete:")
        print(f"  Steps collected: {len(buffer)}")
        print(f"  Episodes: {stats.get('episode_count', 'Incomplete')}")
        print(f"  Mean reward: {stats.get('mean_reward', 0):.3f}")
        print(f"  Total reward: {stats.get('total_reward', 0):.2f}")
        
        return buffer


if __name__ == "__main__":
    # Test experience buffer
    print("Testing PPO Experience Buffer...")
    
    buffer = PPOExperienceBuffer()
    
    # Add some dummy experiences
    for i in range(5):
        exp = PPOExperience(
            structured_input={'test': f'state_{i}'},
            action=torch.tensor([0.1 * i, -0.1 * i]),
            log_prob=torch.tensor(-0.5),
            value=torch.tensor(float(i)),
            reward=1.0 if i % 2 == 0 else -0.1,
            done=(i == 4)
        )
        buffer.add_experience(exp)
    
    # Test advantage computation
    advantages, returns = buffer.compute_advantages_and_returns()
    print(f"Advantages: {advantages}")
    print(f"Returns: {returns}")
    
    # Test batch data
    batch_data = buffer.get_batch_data()
    print(f"Batch data keys: {batch_data.keys()}")
    
    # Test statistics
    stats = buffer.get_statistics()
    print(f"Statistics: {stats}")
    
    print("✅ Experience buffer test complete!")