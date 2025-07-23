"""
Rewards Module - Reinforcement Learning Reward Calculation

This module provides reward calculation for RL training, including:
- Approaching rewards based on current state
- Catch retro rewards based on future episode data
- Lookback logic for retroactive reward assignment
"""

from .reward_processor import RewardProcessor, create_reward_processor

__all__ = ['RewardProcessor', 'create_reward_processor'] 