# AI Predator-Prey Ecosystem

A neural network predator hunting autonomous boids in a simulated ecosystem.

## Features

- **22→12→8→2 Neural Network** for predator decision-making
- **Vision-Based Design** with fixed 400×568 rectangular vision area
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
- **Input**: 22 neurons (5 nearest boids × 4 features + predator velocity)
- **Hidden 1**: 12 neurons with tanh activation
- **Hidden 2**: 8 neurons with tanh activation
- **Output**: 2 neurons (steering forces X, Y)
- **Vision**: Only processes boids within 400×568 rectangular area

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
- **Predator AI**: Vision-based neural network with rectangular vision area and 4-layer architecture
- **Performance**: Optimized loops, centralized constants, zero unused code
- **Consistency**: Frame-based timing ensures identical behavior across devices 