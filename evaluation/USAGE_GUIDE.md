# Policy Evaluation Usage Guide

## Overview

The evaluation system provides a standardized way to compare different policies for the boid catching simulation. It evaluates policies across multiple scenarios and provides statistical analysis of their performance.

## Quick Start

```bash
# From project root directory:
python3 evaluation/evaluate_policies.py
```

## Command Line Options

```bash
# Run with more episodes for better statistical significance
python3 evaluation/evaluate_policies.py --episodes 50

# Test specific scenarios
python3 evaluation/evaluate_policies.py --scenarios easy,medium,hard

# Show detailed progress
python3 evaluation/evaluate_policies.py --verbose

# Use a different transformer checkpoint
python3 evaluation/evaluate_policies.py --checkpoint path/to/model.pt
```

## What Gets Evaluated

1. **Random Policy** - Baseline that outputs random actions
2. **Closest Pursuit Policy** - Greedy strategy pursuing nearest boid
3. **Transformer Policy** - Neural network trained on closest pursuit data

## Performance Benchmarks

Expected catch rates from testing:
- Random: 15-20%
- Closest Pursuit: 25-30%
- Transformer: 25-35%

## Understanding Output

### Overall Performance Table
Shows catch rate (% ± std), efficiency (boids/step), and success rate for each policy.

### Relative Performance
Compares each policy against the random baseline and against each other.

### Scenario Breakdown
Shows performance on each scenario type (easy, medium, hard, dense, sparse).

### Performance Profiles
- **Adaptability**: How consistent the policy is across scenarios (0-1)
- **Early game**: Performance in first 25% of episodes

## For Future Agents

### Adding a New Policy

1. Create a policy class with a `get_action(structured_inputs)` method:
```python
class MyPolicy:
    def get_action(self, structured_inputs):
        # structured_inputs contains:
        # - context: {'canvasWidth': float, 'canvasHeight': float}
        # - predator: {'velX': float, 'velY': float}  
        # - boids: [{'relX': float, 'relY': float, 'velX': float, 'velY': float}, ...]
        
        # Return action as [x, y] in range [-1, 1]
        return [0.5, -0.3]
```

2. Add it to the evaluation script:
```python
policies['MyPolicy'] = MyPolicy()
```

### Evaluating RL-Trained Models

To evaluate a model trained with RL:

1. Create a wrapper that implements `get_action()`:
```python
class RLPolicyWrapper:
    def __init__(self, model_path):
        self.model = load_your_model(model_path)
    
    def get_action(self, structured_inputs):
        # Convert structured_inputs to your model's format
        # Return action in [-1, 1] range
        return self.model.predict(structured_inputs)
```

2. Compare against baselines:
```bash
python3 evaluation/evaluate_policies.py --episodes 100
```

### Success Criteria for RL

An RL-trained policy should:
1. **Minimum**: Beat random baseline (>20%)
2. **Good**: Match closest pursuit (~30%)
3. **Excellent**: Exceed closest pursuit (>30%)
4. **Outstanding**: Achieve >40% catch rate

### Common Issues

**Low performance (<15%)**
- Check action output is in [-1, 1] range
- Verify input preprocessing matches training
- Ensure model weights loaded correctly

**High variance between runs**
- Increase episodes per scenario (use --episodes 50+)
- Check if policy has stochastic elements

**Crashes during evaluation**
- Ensure get_action() handles edge cases (no boids, etc.)
- Check for proper error handling

## Implementation Details

The evaluation system:
- Uses fixed random seeds for reproducibility
- Tests across 5 scenario types by default
- Provides confidence intervals for catch rates
- Measures multiple performance metrics
- Supports custom scenario definitions

For detailed implementation, see `evaluation/practical_evaluator.py`.