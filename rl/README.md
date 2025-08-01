# RL Training System

A comprehensive reinforcement learning training system for boid simulation using transformer models and PPO.

## Overview

This system builds upon the existing simulation, reward, and configuration infrastructure to provide:

- **Gym Environment**: `BoidEnvironment` - Wraps the simulation system for RL training
- **Transformer Models**: PyTorch implementation compatible with pre-trained models
- **PPO Training**: Integration with stable-baselines3 for reinforcement learning
- **Model Loading**: Load supervised learning models and continue with RL training
- **Comprehensive Testing**: Full test suite to ensure reliability

## Quick Start

### 1. Install Dependencies

```bash
pip install torch gymnasium stable-baselines3[extra] tensorboard
```

### 2. Run Tests

```bash
# Run all tests
python -m rl.tests.run_tests

# Run quick tests only
python -m rl.tests.run_tests --quick
```

### 3. Quick Training Example

```bash
# Quick test training (1 minute)
python -m rl.training.ppo_trainer --config quick_test

# Small scale training (10 minutes)  
python -m rl.training.ppo_trainer --config small_scale

# Full scale training (1+ hours)
python -m rl.training.ppo_trainer --config full_scale
```

### 4. Load Pre-trained Model

```bash
# Train using best_model.pt as starting point
python -m rl.training.ppo_trainer --config full_scale --model_checkpoint best_model.pt
```

## System Architecture

```
rl/
├── environment/           # Gym environment wrapper
│   └── boid_env.py       # BoidEnvironment class
├── models/               # Model loading and policies
│   ├── transformer_model.py    # PyTorch transformer
│   └── policy_wrapper.py       # SB3 policy integration
├── training/             # Training infrastructure
│   ├── ppo_trainer.py    # Main training script
│   └── config.py         # Training configurations
├── utils/                # Utilities and helpers
│   └── helpers.py        # Various utility functions
└── tests/                # Comprehensive test suite
    ├── test_environment.py     # Environment tests
    ├── test_model_loading.py   # Model loading tests  
    ├── test_integration.py     # Integration tests
    └── run_tests.py            # Test runner
```

## Key Components

### BoidEnvironment

Gym-compatible environment that wraps the existing simulation system:

```python
from rl.environment import BoidEnvironment

env = BoidEnvironment(
    num_boids=20,
    canvas_width=800,
    canvas_height=600,
    max_steps=1000,
    seed=42
)

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### TransformerModel

PyTorch transformer model compatible with pre-trained checkpoints:

```python
from rl.models import TransformerModel, TransformerModelLoader

# Create new model
model = TransformerModel(
    d_model=64,
    n_heads=8, 
    n_layers=4,
    ffn_hidden=256
)

# Load pre-trained model
loader = TransformerModelLoader()
model = loader.load_model("best_model.pt")
```

### PPO Training

Stable-baselines3 PPO training with custom transformer policy:

```python
from rl.training import PPOTrainer, TrainingConfig

# Create configuration
config = TrainingConfig(
    num_boids=20,
    total_timesteps=1_000_000,
    model_checkpoint="best_model.pt"
)

# Run training
trainer = PPOTrainer(config)
trainer.train()
```

## Configuration Options

### Training Configurations

- `quick_test`: Fast testing (50K timesteps, 2 envs)
- `small_scale`: Small training (500K timesteps, 4 envs)  
- `full_scale`: Full training (2M timesteps, 8 envs)
- `large_scale`: Large training (5M timesteps, 16 envs)

### Model Architectures

Models are configurable with:
- `d_model`: Model dimension (32, 64, 128, ...)
- `n_heads`: Number of attention heads (4, 8, 16, ...)
- `n_layers`: Number of transformer layers (2, 4, 8, ...)
- `ffn_hidden`: Feed-forward hidden dimension

## Testing

The system includes comprehensive tests:

```bash
# Test environment functionality
python -m rl.tests.test_environment

# Test model loading
python -m rl.tests.test_model_loading

# Test full integration
python -m rl.tests.test_integration

# Run all tests
python -m rl.tests.run_tests
```

## Model Loading

The system can load pre-trained supervised learning models:

1. **Checkpoint Format**: Supports `.pt` files with model state dict
2. **Architecture Matching**: Automatically detects and matches model architecture
3. **Graceful Fallback**: Creates new model if checkpoint not found
4. **Validation**: Extensive validation of loaded parameters

## Monitoring

Training progress is monitored via:

- **Tensorboard**: Real-time training metrics
- **Console Logging**: Episode rewards, boids caught, success rate
- **Checkpoints**: Regular model saving for recovery
- **Evaluation**: Periodic evaluation on separate episodes

## Advanced Usage

### Custom Environment

```python
from rl.environment import BoidEnvironment

env = BoidEnvironment(
    num_boids=30,           # More boids
    canvas_width=1200,      # Larger canvas
    canvas_height=800,
    max_steps=2000,         # Longer episodes
    seed=42
)
```

### Custom Model Architecture

```python
from rl.models import TransformerModel

model = TransformerModel(
    d_model=128,        # Larger model
    n_heads=16,         # More attention heads
    n_layers=8,         # Deeper network
    ffn_hidden=512,     # Larger feed-forward
    max_boids=50,       # Support more boids
    dropout=0.1
)
```

### Custom Training Configuration

```python
from rl.training import TrainingConfig

config = TrainingConfig(
    # Environment
    num_boids=25,
    canvas_width=1000,
    canvas_height=700,
    
    # Training  
    total_timesteps=3_000_000,
    learning_rate=1e-4,
    n_envs=12,
    
    # Model
    model_config={
        'd_model': 96,
        'n_heads': 12,
        'n_layers': 6,
        'ffn_hidden': 384
    }
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **CUDA Issues**: Use `device='cpu'` for CPU-only training
3. **Memory Issues**: Reduce `n_envs` or model size
4. **Slow Training**: Use smaller model or fewer boids for testing

### Debug Mode

Run with verbose logging:

```python
config.verbose = 2  # Maximum verbosity
trainer = PPOTrainer(config)
trainer.train()
```

### Test Specific Components

```bash
# Test only environment
python -c "from rl.tests.test_environment import *; test_environment_creation()"

# Test only model loading  
python -c "from rl.tests.test_model_loading import *; test_transformer_model_creation()"
```

## Performance Tips

1. **Use GPU**: Set `device='cuda'` for faster training
2. **Parallel Environments**: Increase `n_envs` for better sample efficiency
3. **Model Size**: Start with smaller models for faster iteration
4. **Batch Size**: Adjust `batch_size` based on available memory
5. **Checkpointing**: Use `save_freq` to save progress regularly

## Integration with Existing Code

This RL system fully integrates with existing components:

- **Simulation**: Uses `StateManager`, `InputProcessor`, `ActionProcessor`
- **Rewards**: Uses `RewardProcessor` for reward calculation
- **Configuration**: Uses centralized `CONSTANTS` from config
- **Random Generation**: Uses `RandomStateGenerator` for episode resets

No changes to existing code are required - the RL system is purely additive.