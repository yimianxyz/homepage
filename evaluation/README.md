# Policy Evaluation System

A simple, unified evaluation system for the boid simulation that provides a single source of truth for policy performance.

## Quick Start - Python Interface (Recommended)

```python
from evaluation import evaluate

# Simplest usage - get catch rate
catch_rate = evaluate(my_policy)
print(f"Policy achieves {catch_rate:.1%} catch rate")

# Detailed evaluation
from evaluation import Evaluator
evaluator = Evaluator(num_episodes=50)
results = evaluator.evaluate(my_policy, detailed=True)
print(f"Catch rate: {results['catch_rate']:.1%}")
print(f"Success rate: {results['success_rate']:.1%}")

# Compare multiple policies
from evaluation import compare
compare({
    'baseline': RandomPolicy(),
    'learned': MyRLPolicy()
})
```

## Quick Start - Command Line

```bash
# Evaluate standard policies (Random, Closest Pursuit, Transformer)
python evaluation/evaluate_policies.py

# Run thorough evaluation with 50 episodes per scenario
python evaluation/evaluate_policies.py --episodes 50

# Evaluate on specific scenarios only
python evaluation/evaluate_policies.py --scenarios easy,medium,dense
```

## Available Policies

1. **Random Policy**: Baseline that outputs random actions in [-1, 1]
2. **Closest Pursuit Policy**: Greedy strategy that always pursues the nearest boid
3. **Transformer Policy**: Neural network trained via supervised learning on closest pursuit data

## Evaluation Scenarios

- **easy**: 5 boids, 400×300 arena (high catch rate expected)
- **medium**: 10 boids, 600×400 arena (standard difficulty)
- **hard**: 20 boids, 800×600 arena (challenging)
- **dense**: 15 boids, 400×300 arena (high density)
- **sparse**: 10 boids, 1000×800 arena (low density)

## Key Metrics

1. **Catch Rate**: Percentage of boids caught (primary metric)
2. **Efficiency**: Boids caught per simulation step
3. **Success Rate**: Percentage of episodes where all boids are caught
4. **Adaptability**: Consistency across different scenarios

## Expected Performance

Based on extensive testing:
- **Random**: ~15-20% catch rate (baseline)
- **Closest Pursuit**: ~25-30% catch rate (greedy baseline)
- **Transformer**: ~25-35% catch rate (should match or exceed closest pursuit)

## Understanding the Results

### Why Complete Success is Rare
- Boids are 1.75× faster than the predator
- Flocking behavior helps boids evade capture
- Success requires cornering or herding strategies

### Performance Factors
- **Arena size**: Smaller arenas → higher catch rates
- **Boid density**: Higher density → easier catching
- **Early game**: Most policies struggle in first 25% of episode

## For RL Training Comparison

Any RL-trained policy should aim to:
1. Beat the **Random baseline** (15-20%)
2. Match or exceed **Closest Pursuit** (25-30%)
3. Show improvement over episodes
4. Demonstrate learned strategies beyond greedy pursuit

## Integration with RL Training

```python
from evaluation import Evaluator

class MyRLTrainer:
    def __init__(self):
        # Quick validation during training
        self.val_evaluator = Evaluator(
            num_episodes=5,
            scenarios=['easy', 'medium']
        )
        
        # Thorough testing
        self.test_evaluator = Evaluator(
            num_episodes=50,
            scenarios=['easy', 'medium', 'hard', 'dense', 'sparse']
        )
    
    def validate(self, policy):
        # Quick validation during training
        return self.val_evaluator.evaluate(policy)
    
    def test(self, policy):
        # Final testing with detailed metrics
        return self.test_evaluator.evaluate(policy, detailed=True)
```

## Core Components

### Simple Interface (Recommended)

```python
from evaluation import Evaluator, evaluate, compare

# One-line evaluation
catch_rate = evaluate(policy)

# Customized evaluation
evaluator = Evaluator(num_episodes=20, scenarios=['dense'])
results = evaluator.evaluate(policy, detailed=True)

# Compare policies
compare({'policy1': p1, 'policy2': p2})
```

### Advanced Interface (For Statistical Analysis)

```python
from evaluation import PracticalEvaluator

# For detailed statistical analysis
evaluator = PracticalEvaluator()
results = evaluator.evaluate_policy(policy, episodes_per_scenario=100)
```

## Files in This Directory

- `evaluator.py` - Simple, unified evaluation interface (start here!)
- `evaluate_policies.py` - Command-line evaluation script
- `practical_evaluator.py` - Advanced evaluation with statistical analysis
- `statistical_analyzer.py` - Statistical utilities for detailed analysis
- `USAGE_GUIDE.md` - Detailed usage guide for all features

## Troubleshooting

### "Transformer checkpoint not found"
- Ensure `checkpoints/best_model.pt` exists
- Or specify custom path: `--checkpoint path/to/model.pt`

### "Module not found" errors
- Run from project root: `python evaluation/evaluate_policies.py`
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### Low transformer performance
- Check if checkpoint loaded correctly (should show architecture details)
- Transformer should perform similarly to Closest Pursuit
- If much worse, there may be a loading issue