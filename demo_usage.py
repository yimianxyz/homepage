"""
Demo Usage - Essential examples of the Python simulation for training

This demonstrates the core workflow for training with the Python simulation.
"""

import random
import numpy as np
from python_simulation import Simulation, InputProcessor, ActionProcessor, CONSTANTS

def demo_basic_usage():
    """Demo: Basic simulation and neural network integration"""
    print("🎮 Python Simulation Demo")
    print("=" * 40)
    
    # Create simulation environment
    sim = Simulation(canvas_width=800, canvas_height=600)
    sim.initialize()
    
    # Create processors
    input_processor = InputProcessor()
    action_processor = ActionProcessor()
    
    print(f"✓ Simulation initialized with {len(sim.boids)} boids")
    print(f"✓ Canvas size: {sim.canvas_width}x{sim.canvas_height}")
    
    # Get current state
    state = sim.get_state()
    
    # Process inputs for neural network
    structured_inputs = input_processor.process_inputs(
        state['boids'],
        state['predator']['position'],
        state['predator']['velocity'],
        state['canvas_width'],
        state['canvas_height']
    )
    
    print(f"✓ Processed {len(structured_inputs['boids'])} boids")
    print(f"✓ Context: {structured_inputs['context']['canvasWidth']:.3f}x{structured_inputs['context']['canvasHeight']:.3f}")
    
    # Simulate neural network outputs
    neural_outputs = [0.3, -0.2]  # Example outputs in [-1, 1] range
    
    # Convert to game actions
    actions = action_processor.process_action(neural_outputs)
    
    print(f"✓ Neural outputs: {neural_outputs}")
    print(f"✓ Game actions: [{actions[0]:.4f}, {actions[1]:.4f}]")
    
    # Apply actions and step simulation
    sim.set_predator_acceleration(actions[0], actions[1])
    sim.step()
    
    print("✓ Simulation stepped successfully")

def demo_training_loop():
    """Demo: Simple training loop"""
    print("\n🎯 Training Loop Demo")
    print("=" * 40)
    
    # Set seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    sim = Simulation(800, 600)
    input_processor = InputProcessor()
    action_processor = ActionProcessor()
    
    episodes = 3
    
    for episode in range(episodes):
        print(f"\n📊 Episode {episode + 1}")
        
        # Reset simulation
        sim.reset()
        step_count = 0
        max_steps = 50
        
        while not sim.is_episode_complete() and step_count < max_steps:
            # Get current state
            state = sim.get_state()
            
            # Process inputs
            structured_inputs = input_processor.process_inputs(
                state['boids'],
                state['predator']['position'],
                state['predator']['velocity'],
                state['canvas_width'],
                state['canvas_height']
            )
            
            # Simulate neural network (random for demo)
            neural_outputs = [
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
            ]
            
            # Convert to actions
            actions = action_processor.process_action(neural_outputs)
            
            # Apply and step
            sim.set_predator_acceleration(actions[0], actions[1])
            sim.step()
            
            step_count += 1
            
        # Episode results
        final_boids = len(sim.boids)
        success = sim.is_episode_complete()
        reward = max(1000 - step_count, 10) if success else 0
        
        print(f"   Steps: {step_count}, Boids: {final_boids}, Success: {success}, Reward: {reward}")

def demo_pytorch_integration():
    """Demo: PyTorch integration example"""
    print("\n🤖 PyTorch Integration Template")
    print("=" * 40)
    
    print("Example PyTorch training code:")
    print("""
import torch
import torch.nn as nn
from python_simulation import Simulation, InputProcessor, ActionProcessor

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Your transformer implementation here
        self.linear = nn.Linear(4, 2)  # Simple placeholder
    
    def forward(self, structured_inputs):
        # Process structured inputs through transformer
        # Return [batch_size, 2] steering forces
        return torch.tanh(self.linear(torch.randn(1, 4)))

# Training loop
model = SimpleTransformer()
    sim = Simulation(800, 600)
    input_processor = InputProcessor()
    action_processor = ActionProcessor()
    
for episode in range(100):
        sim.reset()
        
        while not sim.is_episode_complete():
            state = sim.get_state()
            
            structured_inputs = input_processor.process_inputs(
                state['boids'],
                state['predator']['position'], 
                state['predator']['velocity'],
                state['canvas_width'],
                state['canvas_height']
            )
            
            neural_outputs = model(structured_inputs)
        actions = action_processor.process_action(neural_outputs.detach().numpy())
            
            sim.set_predator_acceleration(actions[0], actions[1])
            sim.step()
    """)

def demo_validation():
    """Demo: Validation testing"""
    print("\n🧪 Validation Demo")
    print("=" * 40)
    
    print("Run validation to ensure JS compatibility:")
    print("python validate_simulation.py")
    print("\nExpected output:")
    print("✓ All tests passed! Python simulation matches JavaScript behavior.")

def main():
    """Run all demos"""
    demo_basic_usage()
    demo_training_loop()
    demo_pytorch_integration()
    demo_validation()
    
    print("\n🎉 Demo complete!")
    print("\nKey Features:")
    print("- 100% exact JavaScript simulation match")
    print("- GPU acceleration with PyTorch")
    print("- Structured input processing")
    print("- Seamless browser deployment")
    
    print("\nNext steps:")
    print("1. python validate_simulation.py")
    print("2. python -m pytorch_training.train_supervised")

if __name__ == "__main__":
    main() 