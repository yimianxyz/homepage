# AI Predator-Prey Ecosystem

A transformer encoder-based predator hunting autonomous boids in a simulated ecosystem.

## Features

- **Transformer Encoder** (d_model=48, n_heads=4, n_layers=3) for predator decision-making
- **Multi-Head Self-Attention** for modeling complex boid interactions
- **Dynamic Token Sequences** with variable-length processing
- **Device-Independent Normalization** for consistent behavior
- **Separated Data Generation & Training** for clean workflow
- **PyTorch to JavaScript Export** for seamless browser deployment
- **Jupyter Notebook Training** for Google Colab compatibility

## Quick Start

### Option 1: Jupyter Notebook (Recommended for Colab)

**Perfect for Google Colab with free GPU access!**

1. **Open `train_model.ipynb`** in Google Colab or Jupyter
2. **Enable GPU** runtime in Colab (Runtime → Change runtime type → GPU)
3. **Run all cells** for complete pipeline:
   - Data generation (50 episodes training, 10 validation)
   - Model training (10 epochs with checkpointing)
   - Export to JavaScript
   - Validation

The notebook includes:
- 🔥 **GPU acceleration** (if available)
- 📊 **Training visualization** with loss plots
- 💾 **Automatic checkpointing** every epoch
- 🚀 **Direct export** to JavaScript format
- ✅ **Built-in validation** tests

### Option 2: Command Line

### 1. Generate Training Data
```bash
# Generate training and validation data
python3 generate_train_val_data.py

# Or generate custom amounts
python3 generate_train_val_data.py --train-episodes 2000 --val-episodes 400
```

### 2. Train Model
```bash
# Train with generated data
python3 -m pytorch_training.train_supervised \
    --train-data data/train_data.pkl \
    --val-data data/val_data.pkl

# Custom training parameters
python3 -m pytorch_training.train_supervised \
    --train-data data/train_data.pkl \
    --val-data data/val_data.pkl \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-3
```

### 3. Export Model to JavaScript
```bash
# Export best model for browser deployment
python3 export_to_js.py \
    --checkpoint checkpoints/best_model.pt \
    --output src/config/model.js

# Export with detailed info
python3 export_to_js.py \
    --checkpoint checkpoints/best_model.pt \
    --output src/config/model.js \
    --info
```

### 4. Validate Simulation
```bash
python3 validate_simulation.py
```

## Architecture

### Core Simulation (Python)
```
python_simulation/
├── simulation.py         # Main simulation controller
├── boid.py              # Boid flocking behavior
├── predator.py          # Predator base class
├── vector.py            # Vector math utilities
├── constants.py         # Simulation parameters
├── input_processor.py   # Neural network input processing
└── action_processor.py  # Neural network output processing
```

### Training (PyTorch)
```
pytorch_training/
├── transformer_model.py    # Transformer implementation
├── supervised_trainer.py   # Training framework
├── simulation_dataset.py   # Data loading
├── teacher_policy.py       # Teacher policy for supervision
└── train_supervised.py     # Training script
```

### Data Generation & Training
```
generate_data.py             # Generate data from single parameters
generate_train_val_data.py   # Generate both train & val data
export_to_js.py             # Export PyTorch models to JavaScript
train_model.ipynb           # Complete Jupyter notebook pipeline
validate_simulation.py       # Validate JS compatibility
demo_usage.py               # Usage examples
```

## Key Components

### Transformer Architecture
- **d_model=48, n_heads=4, n_layers=3**
- **Token sequence**: [CLS] + [CTX] + Predator + Boids
- **~72,000 parameters** total
- **Dynamic sequences** (no padding waste)
 
### Data Generation (Separated)
- **Teacher policy supervision** for training data
- **Pickle format** for efficient storage
- **Configurable episodes** and simulation parameters
- **Separate train/validation** with different seeds

### Training (Simplified)
- **Load pre-generated data** for faster training
- **GPU acceleration** with automatic device detection
- **Tensorboard logging** for monitoring
- **Early stopping** and checkpointing

### Model Export
- **PyTorch to JavaScript** conversion
- **Compatible with browser transformer** implementation
- **Preserves all trained parameters**
- **Direct deployment** to production environment

## Complete Workflow

### Notebook Workflow (Recommended)
1. **Open `train_model.ipynb`** in Google Colab
2. **Run all cells** for complete pipeline
3. **Download `model_export.js`** for browser deployment

### Command Line Workflow
1. **Generate Data**: `python3 generate_train_val_data.py`
2. **Train Model**: `python3 -m pytorch_training.train_supervised --train-data data/train_data.pkl --val-data data/val_data.pkl`
3. **Export to JS**: `python3 export_to_js.py --checkpoint checkpoints/best_model.pt --output src/config/model.js`
4. **Deploy**: Load `src/config/model.js` in your browser application
5. **Validate**: `python3 validate_simulation.py`

## Model Management

### Training Outputs
- **`checkpoints/best_model.pt`** - Best validation loss model
- **`checkpoints/checkpoint_epoch_N.pt`** - Regular epoch checkpoints
- **`runs/`** - TensorBoard logs

### Export Options
```bash
# Export to default JS location
python3 export_to_js.py --checkpoint checkpoints/best_model.pt

# Export to custom location
python3 export_to_js.py \
    --checkpoint checkpoints/best_model.pt \
    --output my_model.js

# Show detailed model information
python3 export_to_js.py \
    --checkpoint checkpoints/best_model.pt \
    --info
```

## Installation

```bash
pip install -r requirements.txt
```

## Validation

The Python simulation is validated to be 100% exact to the JavaScript version:

```bash
python3 validate_simulation.py
```

Expected output: "All tests passed! Python simulation matches JavaScript behavior." 