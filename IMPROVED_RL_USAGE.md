# Improved RL System Usage Guide

## Overview

The improved RL system addresses key issues in the original implementation and provides a systematic approach to training transformer policies that significantly outperform baselines.

## Key Improvements

### 1. **Proper Stochastic Policy Architecture**
- Fixed deterministic policy issue that prevented proper PPO training
- Shared transformer backbone for both actor and critic
- Learnable action standard deviation for controlled exploration
- Strategic attention mechanisms

### 2. **Enhanced Strategic Reward System**
- **Herding Rewards**: Incentivize grouping boids together for easier catching
- **Planning Rewards**: Reward strategic positioning and smart target selection
- **Exploration Bonus**: Encourage visiting new areas to avoid local minima
- **Adaptive Difficulty**: Rewards scale appropriately with scenario complexity

### 3. **Systematic Curriculum Learning**
- 6-stage progressive difficulty: tutorial → basic → intermediate → advanced → expert → master
- Performance-based advancement criteria
- Automatic hyperparameter adaptation per stage
- Transfer learning between stages

### 4. **Evaluation System Integration**
- Uses existing evaluation system as single source of truth
- Systematic performance monitoring and validation
- Automatic baseline comparison and improvement tracking
- Clear performance targets and achievement metrics

## Quick Start

### Method 1: Use the Complete Improved Pipeline (Recommended)

```bash
# Run with small-scale configuration
python3 rl/training/improved_trainer.py --config small_scale

# Run with curriculum learning disabled (if needed)
python3 rl/training/improved_trainer.py --config small_scale --no-curriculum

# Run with custom validation frequency
python3 rl/training/improved_trainer.py --config full_scale --validation-freq 25000
```

### Method 2: Use Individual Components

```python
from rl.training.improved_trainer import ImprovedTrainingPipeline
from rl.training.config import get_small_scale_config

# Create configuration
config = get_small_scale_config()
config.model_checkpoint = "checkpoints/best_model.pt"  # Your pretrained model

# Create and run improved pipeline
pipeline = ImprovedTrainingPipeline(
    config=config,
    use_curriculum=True,
    enable_strategic_rewards=True,
    validation_freq=50000
)

# Run training
results = pipeline.train()

# Check results
if results.get('improvements'):
    best_improvement = max(results['improvements'].values())
    print(f"Achieved {best_improvement:+.1f}% improvement over baseline!")
```

## Configuration Options

### Training Configurations

- **`quick_test`**: Fast testing (50K timesteps, 2 envs)
- **`small_scale`**: Small-scale training (500K timesteps, 4 envs) 
- **`full_scale`**: Full training (2M timesteps, 8 envs)

### Pipeline Options

- **`use_curriculum`**: Enable progressive difficulty curriculum (recommended: True)
- **`enable_strategic_rewards`**: Use enhanced reward system (recommended: True)
- **`validation_freq`**: How often to validate using eval system (default: 50K timesteps)

## Expected Performance

Based on our systematic design, the improved system should achieve:

- **Target**: 15%+ improvement over ClosestPursuit baseline (~27.7% → 32%+ catch rate)
- **Progressive Learning**: Each curriculum stage builds strategic capabilities
- **Consistent Performance**: Lower variance due to strategic reward guidance
- **Better Generalization**: Curriculum training improves adaptability across scenarios

## Monitoring Training

The system provides comprehensive monitoring:

### During Training
- **Stage Progression**: Automatic advancement through curriculum stages
- **Performance Tracking**: Regular validation using evaluation system
- **Baseline Comparison**: Continuous comparison with ClosestPursuit
- **Best Model Saving**: Automatic saving of best-performing models

### After Training
- **Training Summary**: Comprehensive results and improvement analysis
- **Performance Breakdown**: Per-scenario and per-stage performance
- **Model Artifacts**: Saved models and training history
- **Evaluation Integration**: Direct comparison using existing eval system

## Testing the System

Before running full training, validate the system:

```bash
# Test all components
python3 test_improved_rl.py

# Test specific components
python3 test_improved_rl.py --component rewards
python3 test_improved_rl.py --component policy
python3 test_improved_rl.py --component curriculum
python3 test_improved_rl.py --component integration
```

All tests should pass before running training.

## Architecture Details

### Improved Policy Architecture
- **Backbone**: Transformer with attention mechanisms
- **Actor Head**: Gaussian distribution with learnable std
- **Critic Head**: Shared backbone with value function
- **Strategic Attention**: Learnable attention weighting

### Curriculum Learning Stages
1. **Tutorial** (3 boids, 300×200): Learn basic catching (40% target)
2. **Basic** (5 boids, 400×300): Multi-target selection (30% target)  
3. **Intermediate** (8 boids, 500×400): Strategic positioning (25% target)
4. **Advanced** (12 boids, 600×450): Flocking coordination (22% target)
5. **Expert** (15 boids, 700×500): Complex multi-agent strategy (20% target)
6. **Master** (20 boids, 800×600): Full complexity (18% target)

### Strategic Reward Components
- **Base Catch Reward**: 10.0 per boid (dominant signal)
- **Approaching Reward**: Proximity + velocity + alignment
- **Herding Reward**: 0.15 × spread reduction 
- **Planning Reward**: 0.2 × vulnerability targeting + 0.1 × efficient pathing
- **Exploration Bonus**: 0.05 × new area visits

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from project root directory
2. **GPU Memory**: Reduce `n_envs` or `batch_size` if GPU memory issues
3. **Slow Training**: Use `quick_test` config for initial validation
4. **Performance Plateaus**: Check if curriculum advancement criteria are too strict

### Performance Issues

If the system doesn't achieve target performance:
1. **Check Baseline**: Ensure ClosestPursuit baseline is ~27%
2. **Verify Components**: Run `test_improved_rl.py` to check all components
3. **Monitor Curriculum**: Check if stages are advancing properly
4. **Adjust Rewards**: Consider tuning strategic reward weights
5. **Increase Training**: May need more timesteps for complex scenarios

## Files Overview

### Core Components
- `rl/models/improved_transformer_policy.py`: Enhanced stochastic policy
- `rewards/strategic_reward_processor.py`: Strategic reward system
- `rl/training/curriculum_trainer.py`: Curriculum learning implementation
- `rl/training/improved_trainer.py`: Complete training pipeline

### Usage Files
- `test_improved_rl.py`: Comprehensive testing script
- `IMPROVED_RL_USAGE.md`: This usage guide
- Configuration files in `rl/training/config.py`

## Success Metrics

The system is successful if it achieves:
- ✅ **Performance Target**: >15% improvement over ClosestPursuit
- ✅ **Curriculum Completion**: Successfully progresses through stages
- ✅ **Evaluation Integration**: Uses eval system as single source of truth
- ✅ **Systematic Improvement**: Clear, measurable, and reproducible gains

This improved RL system provides a systematic, principled approach to training transformer policies that should significantly outperform the original SL transformer and baseline policies.