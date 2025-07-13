# PyTorch Supervised Learning Environment

A modern PyTorch implementation for training the transformer predator using supervised learning.

## Overview

This environment provides GPU-accelerated supervised learning using a teacher policy to train the transformer predator.

## Key Features

- **Transformer Architecture** - Exact match to JavaScript implementation
- **GPU Acceleration** with automatic device detection
- **Multiprocessing Data Generation** for faster training
- **TensorBoard Logging** for monitoring
- **Model Checkpointing** with best model saving
- **Early Stopping** to prevent overfitting

## Quick Start

### 1. Install Dependencies
```bash
pip install torch tensorboard numpy tqdm
```

### 2. Start Training
```bash
# Basic training
python -m pytorch_training.train_supervised

# Custom parameters
python -m pytorch_training.train_supervised \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --train_episodes 2000
```

### 3. Monitor Training
```bash
tensorboard --logdir runs
```

## Training Configuration

### Key Parameters
```bash
--epochs 100              # Number of training epochs
--batch_size 32           # Batch size
--learning_rate 1e-3      # Learning rate
--train_episodes 1000     # Training episodes per epoch
--val_episodes 200        # Validation episodes
--num_workers 8           # Parallel workers for data generation
--device auto             # Device (cuda/cpu/auto)
```

## Architecture

### Transformer Model
```python
TransformerPredator(
    d_model=48,           # Model dimension
    n_heads=4,            # Attention heads  
    n_layers=3,           # Transformer layers
    ffn_hidden=96,        # FFN hidden size
    dropout=0.1           # Dropout rate
)
```

### Token Sequence
**[CLS] + [CTX] + Predator + Boids**
- **[CLS]**: Global aggregation token
- **[CTX]**: Canvas context (width, height)
- **Predator**: Velocity information
- **Boids**: Relative position and velocity

## Usage Examples

### Basic Training
```python
from pytorch_training import create_trainer

# Create trainer
trainer = create_trainer(
    train_episodes=1000,
    val_episodes=200,
    learning_rate=1e-3,
    batch_size=32
)

# Train model
trainer.train(num_epochs=100, early_stopping_patience=20)
```

### Custom Model
```python
from pytorch_training import TransformerPredator, SupervisedTrainer

# Create custom model
model = TransformerPredator(
    d_model=64,
    n_heads=8,
    n_layers=4
)

# Create trainer
trainer = SupervisedTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    learning_rate=5e-4,
    batch_size=64
)

trainer.train(num_epochs=100)
```

### Resume Training
```bash
python -m pytorch_training.train_supervised \
    --resume checkpoints/checkpoint_epoch_50.pt \
    --epochs 200
```

## Performance

### Training Speed
- **CPU**: ~500 samples/second
- **GPU**: ~5,000 samples/second (10x speedup)
- **Multiprocessing**: 4-8x faster data generation

### Memory Usage
- **Model**: ~71K parameters (~280KB)
- **Training batch**: ~1-2MB per batch (batch_size=32)

### Convergence
- **Typical convergence**: 50-100 epochs
- **Best validation loss**: ~0.001-0.005 MSE
- **Training time**: 10-30 minutes (GPU)

## File Structure

```
pytorch_training/
├── transformer_model.py        # PyTorch transformer implementation
├── teacher_policy.py           # Teacher policy for supervision
├── simulation_dataset.py       # Dataset and data loading
├── supervised_trainer.py       # Training loop and utilities
└── train_supervised.py         # Main training script
```

## Model Management

### Save Model
```python
# Automatic saving during training
trainer.train(num_epochs=100)  # Saves best_model.pt

# Manual saving
trainer.save_checkpoint('my_model.pt')
```

### Load Model
```python
# Load checkpoint
trainer.load_checkpoint('checkpoints/best_model.pt')

# Use for prediction
prediction = trainer.evaluate_sample(structured_inputs)
```

## Troubleshooting

### Out of Memory
```bash
--batch_size 16
--train_episodes 500
```

### Slow Convergence
```bash
--learning_rate 5e-3
--train_episodes 2000
```

### Overfitting
```bash
--early_stopping 15
--val_episodes 500
```

This environment provides a robust foundation for training transformer-based predator AI with modern deep learning techniques while maintaining compatibility with the JavaScript inference system. 