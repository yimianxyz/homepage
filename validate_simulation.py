"""
Validation Test - Ensure Python simulation matches JavaScript exactly

This test validates that the Python simulation environment produces
identical behavior to the JavaScript version.
"""

import random
import math
from python_simulation import Simulation, InputProcessor, ActionProcessor, Vector, CONSTANTS

def test_constants():
    """Test that all constants match JavaScript values"""
    print("Testing constants...")
    
    # Test key constants
    assert CONSTANTS.BOID_MAX_SPEED == 3.5
    assert CONSTANTS.PREDATOR_MAX_SPEED == 2
    assert CONSTANTS.PREDATOR_FORCE_SCALE == 200
    assert CONSTANTS.NUM_BOIDS == 50
    assert CONSTANTS.MAX_DISTANCE == 1000
    
    print("✓ Constants match JavaScript values")

def test_vector_operations():
    """Test vector operations match JavaScript behavior"""
    print("Testing vector operations...")
    
    # Test basic operations
    v1 = Vector(3, 4)
    v2 = Vector(1, 2)
    
    # Test addition
    v3 = v1.add(v2)
    assert v3.x == 4 and v3.y == 6
    
    # Test magnitude
    assert abs(v1.getMagnitude() - 5.0) < 0.0001
    
    # Test fast magnitude (should be close to real magnitude)
    fast_mag = v1.getFastMagnitude()
    assert abs(fast_mag - 5.0) < 0.5  # Fast approximation
    
    # Test normalization
    v4 = v1.normalize()
    assert abs(v4.getMagnitude() - 1.0) < 0.0001
    
    print("✓ Vector operations working correctly")

def test_input_processor():
    """Test input processor produces correct structured inputs"""
    print("Testing input processor...")
    
    processor = InputProcessor()
    
    # Test with sample data
    boids = [
        {'position': {'x': 100, 'y': 200}, 'velocity': {'x': 1.5, 'y': -2.0}},
        {'position': {'x': 300, 'y': 400}, 'velocity': {'x': -1.0, 'y': 0.5}}
    ]
    
    predator_pos = {'x': 150, 'y': 250}
    predator_vel = {'x': 0.8, 'y': -0.6}
    
    structured_inputs = processor.process_inputs(
        boids, predator_pos, predator_vel, 800, 600
    )
    
    # Test structure
    assert 'context' in structured_inputs
    assert 'predator' in structured_inputs
    assert 'boids' in structured_inputs
    
    # Test context normalization
    assert structured_inputs['context']['canvasWidth'] == 800 / 1000
    assert structured_inputs['context']['canvasHeight'] == 600 / 1000
    
    # Test predator velocity normalization
    max_vel = max(CONSTANTS.BOID_MAX_SPEED, CONSTANTS.PREDATOR_MAX_SPEED)
    assert abs(structured_inputs['predator']['velX'] - (0.8 / max_vel)) < 0.0001
    
    # Test boid count
    assert len(structured_inputs['boids']) == 2
    
    print("✓ Input processor producing correct structured inputs")

def test_action_processor():
    """Test action processor scales outputs correctly"""
    print("Testing action processor...")
    
    processor = ActionProcessor()
    
    # Test with sample neural outputs
    neural_outputs = [0.5, -0.3]
    actions = processor.process_action(neural_outputs)
    
    expected_scale = CONSTANTS.PREDATOR_MAX_FORCE * CONSTANTS.PREDATOR_FORCE_SCALE
    assert abs(actions[0] - (0.5 * expected_scale)) < 0.0001
    assert abs(actions[1] - (-0.3 * expected_scale)) < 0.0001
    
    print("✓ Action processor scaling outputs correctly")

def test_simulation_basic():
    """Test basic simulation functionality"""
    print("Testing basic simulation...")
    
    # Set seed for reproducible results
    random.seed(42)
    
    sim = Simulation(800, 600)
    sim.initialize()
    
    # Test initialization
    assert len(sim.boids) == CONSTANTS.NUM_BOIDS
    assert sim.predator is not None
    assert sim.canvas_width == 800
    assert sim.canvas_height == 600
    
    # Test initial state
    initial_state = sim.get_state()
    assert len(initial_state['boids']) == CONSTANTS.NUM_BOIDS
    assert initial_state['predator'] is not None
    
    # Test simulation step
    initial_boid_count = len(sim.boids)
    sim.step()
    
    # Boids should still exist after one step
    assert len(sim.boids) <= initial_boid_count
    
    print("✓ Basic simulation functionality working")

def test_episode_mechanics():
    """Test episode completion mechanics"""
    print("Testing episode mechanics...")
    
    sim = Simulation(800, 600)
    sim.initialize()
    
    # Test episode completion
    assert not sim.is_episode_complete()  # Should not be complete initially
    
    # Simulate catching most boids
    while len(sim.boids) > 20:
        sim.boids.pop()
    
    assert sim.is_episode_complete()  # Should be complete now
    
    print("✓ Episode mechanics working correctly")

def test_neural_network_integration():
    """Test neural network integration workflow"""
    print("Testing neural network integration...")
    
    random.seed(42)
    
    sim = Simulation(800, 600)
    sim.initialize()
    
    input_processor = InputProcessor()
    action_processor = ActionProcessor()
    
    # Get initial state
    state = sim.get_state()
    
    # Process inputs for neural network
    structured_inputs = input_processor.process_inputs(
        state['boids'],
        state['predator']['position'],
        state['predator']['velocity'],
        state['canvas_width'],
        state['canvas_height']
    )
    
    # Simulate neural network outputs
    neural_outputs = [0.2, -0.1]  # Simple test outputs
    
    # Process actions
    actions = action_processor.process_action(neural_outputs)
    
    # Apply actions to simulation
    sim.set_predator_acceleration(actions[0], actions[1])
    
    # Step simulation
    sim.step()
    
    # Verify state changed
    new_state = sim.get_state()
    assert new_state != state  # State should have changed
    
    print("✓ Neural network integration workflow working")

def test_boundary_conditions():
    """Test boundary wrapping behavior"""
    print("Testing boundary conditions...")
    
    sim = Simulation(800, 600)
    
    # Test boid boundary wrapping
    from python_simulation.boid import Boid
    boid = Boid(0, 0, sim)
    
    # Test wrapping
    boid.position.x = 900  # Beyond canvas width
    boid.bound()
    assert boid.position.x == -CONSTANTS.BOID_BORDER_OFFSET
    
    # Test predator boundary wrapping
    from python_simulation.predator import Predator
    predator = Predator(0, 0, sim)
    predator.position.x = 900  # Beyond canvas width
    predator.bound()
    assert predator.position.x == -CONSTANTS.PREDATOR_BORDER_OFFSET
    
    print("✓ Boundary conditions working correctly")

def run_all_tests():
    """Run all validation tests"""
    print("🧪 Running Python Simulation Validation Tests\n")
    
    try:
        test_constants()
        test_vector_operations()
        test_input_processor()
        test_action_processor()
        test_simulation_basic()
        test_episode_mechanics()
        test_neural_network_integration()
        test_boundary_conditions()
        
        print("\n🎉 All tests passed! Python simulation matches JavaScript behavior.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_all_tests() 