# AI Predator-Prey Ecosystem

A transformer encoder-based predator hunting autonomous boids in a simulated ecosystem, using Vision Transformer architecture for dynamic entity interaction modeling.

## Features

- **Transformer Encoder** (d_model=48, n_heads=4, n_layers=3) for predator decision-making
- **Variable-Length Token Sequences** with dynamic entity processing (no fixed padding)
- **Multi-Head Self-Attention** for modeling complex boid interactions
- **GEGLU Feed-Forward Networks** for enhanced non-linear processing
- **Device-Independent Normalization** ensures consistent behavior across all screen sizes
- **Frame-Based Timing** for consistent behavior across devices
- **Two Training Modes**: Supervised learning and reinforcement learning
- **Model Persistence**: Save/load transformer parameters for continued training
- **Responsive Design** works on all screen sizes

## Usage

### Homepage
Open `index.html` to see the trained predator simulation running in the background.

### Training Interfaces
- **Supervised Learning**: Open `training-sl.html` to train using teacher policy
- **Reinforcement Learning**: Open `training-rl.html` to train using environment rewards

### Model Management
- **Reset Network**: Initialize transformer with random parameters
- **Load Network**: Load pre-trained transformer from `src/config/model.js`
- **Export Network**: Generate complete model.js content for saving trained parameters

## Technical Details

### Transformer Architecture
- **Model Dimensions**: d_model=48, n_heads=4, n_layers=3, ffn_hidden=96
- **Token Sequence**: [CLS] + [CTX] + Predator + Boids (variable length)
- **Attention Mechanism**: Multi-head self-attention with scaled dot-product
- **Feed-Forward**: GEGLU networks (Gate-Enhanced Gated Linear Units)
- **Output**: [CLS] token → 2D steering forces via linear projection
- **Parameters**: ~17,000 total parameters
- **Memory Efficiency**: 100% utilization vs 14% with fixed arrays

### Token Types & Embeddings
- **[CLS] Token**: Global aggregation token for final steering decision
- **[CTX] Token**: Canvas context `[width/D, height/D]` for spatial awareness
- **Predator Token**: Velocity info `[vx/V, vy/V, 0, 0]` with padding
- **Boid Tokens**: Relative position and velocity `[dx/D, dy/D, dvx/V, dvy/V]`
- **Type Embeddings**: Distinct embeddings for each entity type (cls, ctx, predator, boid)

### Input Processing
- **Structured Format**: `{context: {canvasWidth, canvasHeight}, predator: {velX, velY}, boids: [{relX, relY, velX, velY}, ...]}`
- **Dynamic Sequences**: 3 + N_boids tokens (no fixed padding)
- **Normalization**: Distances by max_distance (D), velocities by max_velocity (V)
- **Relative Encoding**: All boid positions relative to predator
- **Attention Efficiency**: O(S²) complexity where S = 3 + N_boids

### Training
- **Supervised**: Transformer learns to imitate simple pursuit behavior using teacher policy
- **Reinforcement**: Transformer learns from success/failure rewards (TODO: implement backpropagation)
- **Episodes**: Continue until predator catches enough boids (≤20 remaining)
- **Rewards**: `max(1000 - completion_frames, 10)` for faster completion
- **Model Persistence**: Complete parameter export/import for continuing training

## Architecture

```
├── index.html              # Homepage with background simulation
├── training-sl.html        # Supervised learning interface  
├── training-rl.html        # Reinforcement learning interface
├── transformer_test.html   # Architecture testing and analysis
└── src/
    ├── ai/                 # Transformer encoder components
    │   ├── transformer_encoder.js    # Main transformer implementation
    │   ├── input_processor.js        # Structured input processing
    │   └── action_processor.js       # Output action conversion
    ├── simulation/         # Boids and predator entities
    │   ├── boid.js              # Autonomous boid behavior
    │   ├── predator.js          # Base predator class
    │   ├── neural_predator.js   # Transformer-controlled predator
    │   └── simulation.js        # Main simulation controller
    ├── training/           # Learning algorithms
    │   ├── supervised_learning.js      # Teacher policy training
    │   ├── reinforcement_learning.js   # Reward-based training
    │   └── teacher_policy.js          # Simple pursuit behavior
    ├── ui/                 # Interface controllers
    │   ├── base_trainer.js          # Common training functionality
    │   ├── supervised_trainer.js    # Supervised learning UI
    │   └── reinforcement_trainer.js  # Reinforcement learning UI
    ├── utils/              # Vector math utilities
    └── config/             # Constants and model weights
        ├── constants.js    # Simulation parameters
        └── model.js        # Transformer parameters (TRANSFORMER_PARAMS)
```

## Implementation

### Transformer Processing Flow
1. **Input Structuring**: Convert boids/predator to structured format
2. **Token Building**: Create [CLS] + [CTX] + Predator + Boids sequence
3. **Embedding**: Apply input projections + type embeddings
4. **Transformer Blocks**: 3 layers of LayerNorm → MHSA → GEGLU FFN
5. **Output Projection**: Extract [CLS] token → 2D steering forces

### Key Features
- **Dynamic Attention**: Models variable numbers of boids without padding waste
- **Entity Interaction**: Self-attention captures complex predator-boid relationships
- **Efficient Processing**: <10ms inference time for typical scenarios
- **Scalable Architecture**: Vision Transformer design adapts to varying entity counts
- **Device Consistency**: Fixed normalization ensures identical behavior across all devices and screen sizes
- **Performance**: Optimized attention computation with fused QKV operations

## Performance Characteristics

- **Inference Time**: <10ms for typical scenarios (5-50 boids)
- **Memory Usage**: 100% efficient (no padding waste)
- **Parameter Count**: ~17,000 parameters
- **Attention Complexity**: O(S²) where S = sequence length
- **Token Efficiency**: 3 + N_boids tokens vs 51 fixed slots (14% → 100% utilization)

## Model Files

The transformer parameters are stored in `src/config/model.js` as `window.TRANSFORMER_PARAMS`. If this file is empty or contains invalid parameters, the system will initialize with random weights and display a warning. Use the "Export Network" button in training interfaces to generate complete model.js content after training. 