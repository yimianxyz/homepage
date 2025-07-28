"""
Rewards Module - Simple Instant Reward Calculation

This module provides instant reward calculation for RL training:
- Catch rewards: Dominant rewards for catching boids (instant)
- Approaching rewards: Small rewards for getting closer to boids (instant)

Design principle: Occam's razor - simple, clean, single-step processing
"""

from .reward_processor import RewardProcessor, create_reward_processor

__all__ = ['RewardProcessor', 'create_reward_processor'] 