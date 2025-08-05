# PPO Reinforcement Learning System

This directory contains a complete PPO (Proximal Policy Optimization) reinforcement learning system that seamlessly integrates with the existing simulation infrastructure.

## 🎯 Overview

The PPO system fine-tunes the supervised learning transformer baseline using reinforcement learning. It reuses all existing components:

- **StateManager**: Environment stepping and state management  
- **RewardProcessor**: Reward calculation for RL training
- **PolicyEvaluator**: Performance evaluation and comparison
- **Simulation Runtime**: Core boid simulation physics

## 🏗️ Architecture

### Core Components

1. **PPOTransformerModel** (`rl_training/ppo_transformer_model.py`)
   - Extends SL transformer with value head for critic
   - Loads pre-trained weights from `best_model.pt`
   - Maintains same policy interface for compatibility

2. **PPOExperienceBuffer** (`rl_training/ppo_experience_buffer.py`)
   - Collects rollout trajectories from simulation
   - Computes advantages using GAE (Generalized Advantage Estimation)
   - Provides batched data for PPO training

3. **PPOTrainer** (`rl_training/ppo_trainer.py`)
   - Main training orchestrator with PPO loss computation
   - Integrates with existing evaluation system
   - Handles checkpointing and logging

4. **PPORolloutCollector** (`rl_training/ppo_experience_buffer.py`)
   - Bridges PPO with simulation infrastructure
   - Uses StateManager and RewardProcessor for data collection

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have a trained supervised learning model:
```bash
# Run the transformer training notebook first
jupyter notebook transformer_training.ipynb
```

This creates `checkpoints/best_model.pt` which serves as the PPO baseline.

### 2. Test Integration

Verify everything works together:
```bash
python test_ppo_integration.py
```

This runs comprehensive tests of all PPO components.

### 3. Start PPO Training

Run PPO training with default parameters:
```bash
python train_ppo.py
```

Or customize training:
```bash
python train_ppo.py \
  --iterations 100 \
  --rollout-steps 2048 \
  --learning-rate 3e-4 \
  --eval-interval 10 \
  --boids 20
```

### 4. Monitor Progress

Training outputs:
- **Checkpoints**: `checkpoints/ppo_iteration_*.pt`
- **Best Model**: `checkpoints/best_ppo_model.pt`  
- **Training Stats**: `checkpoints/ppo_training_stats.json`
- **Evaluation Results**: Printed during training

## 📊 Training Process

Each PPO iteration consists of:

1. **Rollout Collection**: Collect 2048 steps of experience using current policy
2. **Advantage Computation**: Calculate GAE advantages and returns
3. **Policy Update**: 4 epochs of PPO loss optimization with mini-batches
4. **Evaluation**: Periodic evaluation using existing PolicyEvaluator
5. **Checkpointing**: Save model and training statistics

## ⚙️ Configuration

Key hyperparameters:

```python
PPOTrainer(
    learning_rate=3e-4,      # Learning rate
    clip_epsilon=0.2,        # PPO clipping parameter  
    rollout_steps=2048,      # Steps per rollout
    ppo_epochs=4,            # Optimization epochs per rollout
    mini_batch_size=64,      # Mini-batch size
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE lambda parameter
    entropy_coef=0.01,       # Entropy bonus coefficient
    value_loss_coef=0.5      # Value loss coefficient
)
```

## 🔄 Integration with Existing Systems

### StateManager Integration
```python
# PPO uses existing StateManager interface
state_manager = StateManager()
state_manager.init(initial_state, ppo_policy)
result = state_manager.step()  # Same as before
```

### RewardProcessor Integration  
```python
# PPO uses existing reward calculation
reward_input = {
    'state': structured_input,
    'action': action_list,
    'caughtBoids': caught_boid_ids
}
reward_data = reward_processor.calculate_step_reward(reward_input)
```

### PolicyEvaluator Integration
```python
# PPO policy works with existing evaluation
evaluator = PolicyEvaluator()
result = evaluator.evaluate_policy(ppo_policy, "PPO_Model")
```

## 📈 Expected Results

Typical training progression:

- **Iteration 0**: Baseline SL performance (~0.6 catch rate)
- **Iterations 1-20**: Rapid improvement from RL fine-tuning  
- **Iterations 20-50**: Continued refinement and stability
- **Final**: Enhanced performance over SL baseline

## 🛠️ Customization

### Custom Reward Functions

Modify `rewards/reward_processor.py` to experiment with different reward designs:

```python
def calculate_step_reward(self, step_input):
    # Add custom reward components
    custom_reward = your_custom_logic(step_input)
    return {'total': custom_reward, ...}
```

### Custom Architectures

Extend `PPOTransformerModel` for architectural experiments:

```python
class CustomPPOModel(PPOTransformerModel):
    def __init__(self, ...):
        super().__init__(...)
        # Add custom layers
        self.custom_head = nn.Linear(d_model, custom_output_dim)
```

### Custom Training Logic

Extend `PPOTrainer` for advanced training procedures:

```python
class CustomPPOTrainer(PPOTrainer):
    def update_policy(self, buffer):
        # Add custom training logic
        return super().update_policy(buffer)
```

## 🧪 Testing and Debugging

### Component Tests
```bash
python -m rl_training.ppo_transformer_model  # Test model
python -m rl_training.ppo_experience_buffer  # Test buffer
python -m rl_training.ppo_trainer            # Test trainer
```

### Integration Test
```bash
python test_ppo_integration.py
```

### Manual Testing
```python
from rl_training import create_ppo_policy_from_sl

# Load and test policy
policy = create_ppo_policy_from_sl("checkpoints/best_model.pt")
action = policy.get_action(test_input)
print(f"Action: {action}")
```

## 📁 File Structure

```
rl_training/
├── __init__.py                    # Package exports
├── ppo_transformer_model.py       # PPO model with value head
├── ppo_experience_buffer.py       # Experience collection and GAE
└── ppo_trainer.py                 # Main training orchestrator

train_ppo.py                       # Training script
test_ppo_integration.py           # Integration tests
PPO_README.md                     # This documentation
```

## 🔍 Troubleshooting

### Common Issues

1. **"Checkpoint not found"**
   - Ensure you've trained an SL model first using `transformer_training.ipynb`

2. **"CUDA out of memory"**
   - Reduce `rollout_steps`, `mini_batch_size`, or use `--device cpu`

3. **"Training not improving"**
   - Check reward function, reduce learning rate, or adjust PPO hyperparameters

4. **"Evaluation hanging"**
   - Reduce evaluation episodes or episode length in PolicyEvaluator

### Performance Tips

- Use GPU for training: `--device cuda`
- Increase `rollout_steps` for more stable gradients
- Use multiple mini-batches per epoch for better sample efficiency
- Monitor entropy to ensure adequate exploration

## 🎉 Success Metrics

Your PPO training is successful when:

- ✅ Integration tests pass
- ✅ Training loss decreases over iterations  
- ✅ Evaluation catch rate improves over SL baseline
- ✅ Policy maintains stable performance
- ✅ Training completes without errors

Happy training! 🚀