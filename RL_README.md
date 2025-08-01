# Reinforcement Learning for Boid Catching

A clean, minimal PPO implementation that successfully trains agents to catch boids.

## Quick Start

### Training
```bash
python3 rl_train_simple.py --train
```

### Evaluation
```bash
python3 rl_train_simple.py --eval
```

### Both (Recommended)
```bash
python3 rl_train_simple.py --train --eval
```

## Performance
- **Baseline (ClosestPursuit)**: ~10%
- **Trained PPO**: ~22% (2x improvement)
- **Training time**: ~20 seconds for 20K timesteps

## Structure
```
rl_train_simple.py      # All training/evaluation logic
RL_LEARNINGS.md         # Key discoveries from debugging
models/ppo_simple.zip   # Trained model

rl/                     # Minimal package
├── __init__.py
└── environment/
    ├── __init__.py
    └── boid_env.py     # Gym environment wrapper
```

## Key Features
- Simple 6-feature extractor (closest boid info)
- Small MLP network [64, 32]
- 500 max steps per episode
- Fixed seed for reproducibility
- Standard PPO hyperparameters

## Customization
```bash
# More training
python3 rl_train_simple.py --train --timesteps 40000

# Different seed
python3 rl_train_simple.py --train --seed 42

# More evaluation episodes
python3 rl_train_simple.py --eval --episodes 20
```

See `RL_LEARNINGS.md` for detailed insights from the debugging process.