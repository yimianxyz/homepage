"""
Evaluation module for policy performance assessment.

Simple usage:
    from evaluation import evaluate
    
    # Get catch rate for a policy
    catch_rate = evaluate(policy)
    
    # Or use the Evaluator class for more control
    from evaluation import Evaluator
    evaluator = Evaluator(num_episodes=50)
    results = evaluator.evaluate(policy, detailed=True)
"""

# Simple interface
from .evaluator import Evaluator, evaluate, compare, create_evaluator

# Advanced interface (for detailed statistical analysis)
from .practical_evaluator import PracticalEvaluator
from .statistical_analyzer import StatisticalAnalyzer

__all__ = [
    # Simple interface (recommended)
    'Evaluator',
    'evaluate', 
    'compare',
    'create_evaluator',
    
    # Advanced interface
    'PracticalEvaluator',
    'StatisticalAnalyzer'
]