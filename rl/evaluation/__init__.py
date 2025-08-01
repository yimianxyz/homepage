"""
RL Evaluation Module - Comprehensive performance evaluation and statistical analysis

This module provides rigorous evaluation tools to verify that RL training
produces meaningful improvements over supervised learning baselines.
"""

from .performance_evaluator import (
    PerformanceEvaluator,
    EpisodeMetrics,
    ModelPerformance,
    create_evaluation_suite
)

__all__ = [
    'PerformanceEvaluator',
    'EpisodeMetrics', 
    'ModelPerformance',
    'create_evaluation_suite'
]