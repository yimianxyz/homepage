"""
RL Training Package - PPO reinforcement learning for transformer policy

This package provides a complete PPO implementation that integrates seamlessly 
with the existing simulation infrastructure. It includes:

- PPOTransformerModel: Extends SL transformer with value head
- PPOExperienceBuffer: Efficient rollout collection and GAE computation  
- PPOTrainer: Complete training pipeline with evaluation integration
- PPORolloutCollector: Bridge between PPO and simulation systems

Key features:
- Loads supervised learning baselines from best_model.pt
- Reuses StateManager, RewardProcessor, PolicyEvaluator
- Maintains same policy interface for evaluation compatibility
- Clean, readable implementation following RL best practices
"""

from .ppo_transformer_model import (
    PPOTransformerModel,
    PPOTransformerPolicy, 
    create_ppo_policy_from_sl
)

from .ppo_experience_buffer import (
    PPOExperience,
    PPOExperienceBuffer,
    PPORolloutCollector
)

from .ppo_trainer import PPOTrainer

__all__ = [
    'PPOTransformerModel',
    'PPOTransformerPolicy',
    'create_ppo_policy_from_sl',
    'PPOExperience', 
    'PPOExperienceBuffer',
    'PPORolloutCollector',
    'PPOTrainer'
]

__version__ = "1.0.0"