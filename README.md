# AI Predator-Prey Ecosystem

A neural network predator hunting autonomous boids in a simulated ecosystem.

## Features

- **204→64→32→2 Neural Network** for predator decision-making
- **Complete Information Design** with all 50 boids + predator encoded as input vectors
- **Device-Independent Normalization** ensures consistent behavior across all screen sizes
- **Frame-Based Timing** for consistent behavior across devices
- **Two Training Modes**: Supervised learning and reinforcement learning
- **Responsive Design** works on all screen sizes

## Usage

### Homepage
Open `index.html` to see the trained predator simulation running in the background.

### Training Interfaces
- **Supervised Learning**: Open `training-sl.html` to train using teacher policy
- **Reinforcement Learning**: Open `training-rl.html` to train using environment rewards

## Technical Details

### Neural Network
- **Input**: 204 neurons (51 entities × 4 features: 50 boids + 1 predator)
- **Hidden 1**: 64 neurons with tanh activation
- **Hidden 2**: 32 neurons with tanh activation
- **Output**: 2 neurons (steering forces X, Y)
- **Complete Information**: Processes all boids without distance limitations

### Input Encoding
- **Boid Vectors**: `[rel_x, rel_y, vel_x, vel_y]` normalized to [-1, 1]
- **Predator Vector**: `[canvas_width_norm, canvas_height_norm, vel_x, vel_y]` normalized to [-1, 1]
- **Fixed Size**: Always 51 entities (padded with zeros if fewer boids exist)
- **No Vision Limits**: All boids included regardless of distance
- **Unified Velocity Norm**: All velocities normalized by max(boid_speed, predator_speed) for consistency
- **World Context**: Predator vector includes normalized canvas dimensions for spatial awareness

### Training
- **Supervised**: Neural network learns to imitate simple pursuit behavior
- **Reinforcement**: Neural network learns from success/failure rewards
- **Episodes**: Continue until predator catches enough boids (≤20 remaining)
- **Rewards**: `max(1000 - completion_frames, 10)` for faster completion

## Architecture

```
├── index.html              # Homepage with background simulation
├── training-sl.html        # Supervised learning interface  
├── training-rl.html        # Reinforcement learning interface
└── src/
    ├── ai/                 # Neural network components
    ├── simulation/         # Boids and predator entities
    ├── training/           # Learning algorithms
    ├── ui/                 # Interface controllers
    ├── utils/              # Vector math utilities
    └── config/             # Constants and model weights
```

## Implementation

- **Boid Flocking**: Reynolds rules (separation, alignment, cohesion) plus predator avoidance
- **Predator AI**: Complete information neural network with 51-entity encoding and 4-layer architecture
- **Device Consistency**: Fixed normalization ensures identical behavior across all devices and screen sizes
- **Performance**: Optimized loops, centralized constants, zero unused code
- **Consistency**: Frame-based timing ensures identical behavior across devices 