"""
Experiments Package - Systematic RL Validation Framework

This package provides a comprehensive experimental framework for systematically
proving that our PPO RL system improves upon the SL baseline.

Key Components:
- ExperimentRunner: Core experimental framework with statistical validation
- ExperimentSuite: Predefined experiment collections for different validation needs
- Automated execution with progress tracking and resumption capabilities
- Statistical significance testing and effect size analysis
- Comprehensive reporting and visualization

Usage:
    # Quick validation (1-2 hours)
    python run_experiments.py --suite quick
    
    # Critical path experiments (3-4 hours)
    python run_experiments.py --suite critical
    
    # Full validation suite (12-16 hours)
    python run_experiments.py --suite full

Experiment Categories:
1. Core Validation - Fundamental RL vs SL comparison
2. Ablation Studies - Component contribution analysis
3. Sensitivity Analysis - Hyperparameter robustness
4. Generalization Tests - Performance across conditions
5. Efficiency Analysis - Sample efficiency validation
"""

from .experimental_framework import (
    ExperimentConfig,
    TrialResult,
    ExperimentResult,
    ExperimentRunner
)

from .experiment_definitions import ExperimentSuite

__all__ = [
    'ExperimentConfig',
    'TrialResult', 
    'ExperimentResult',
    'ExperimentRunner',
    'ExperimentSuite'
]

__version__ = "1.0.0"