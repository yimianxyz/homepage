"""
State Manager - Simple state management for simulation

This provides a simple interface that bridges between policy and runtime,
handling input/output conversion and state management. 
This MUST match exactly with the JavaScript implementation.
"""

from typing import Dict, List, Any, Optional
import copy

from ..processors import InputProcessor, ActionProcessor
from ..runtime.simulation_runtime import simulation_step

class StateManager:
    """Simple state manager that bridges policy and runtime"""
    
    def __init__(self):
        # Initialize processors
        self.input_processor = InputProcessor()
        self.action_processor = ActionProcessor()
        
        # State variables
        self.current_state: Optional[Dict[str, Any]] = None
        self.policy = None
    
    def init(self, 
             initial_state: Dict[str, Any],
             policy) -> None:
        """
        Initialize state manager with initial state and policy
        
        Args:
            initial_state: Initial state dict with structure:
                {
                    'boids_states': [
                        {
                            'position': {'x': float, 'y': float},
                            'velocity': {'x': float, 'y': float}
                        },
                        ...
                    ],
                    'predator_state': {
                        'position': {'x': float, 'y': float},
                        'velocity': {'x': float, 'y': float}
                    },
                    'canvas_width': float,
                    'canvas_height': float
                }
            policy: Policy object with get_action method
        """
        # Validate input structure
        required_keys = ['boids_states', 'predator_state', 'canvas_width', 'canvas_height']
        for key in required_keys:
            if key not in initial_state:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate boids_states structure
        if not isinstance(initial_state['boids_states'], list):
            raise ValueError("boids_states must be a list")
        
        for i, boid_state in enumerate(initial_state['boids_states']):
            if not isinstance(boid_state, dict):
                raise ValueError(f"boids_states[{i}] must be a dict")
            if 'position' not in boid_state or 'velocity' not in boid_state:
                raise ValueError(f"boids_states[{i}] must have 'position' and 'velocity' keys")
        
        # Validate predator_state structure
        if not isinstance(initial_state['predator_state'], dict):
            raise ValueError("predator_state must be a dict")
        if 'position' not in initial_state['predator_state'] or 'velocity' not in initial_state['predator_state']:
            raise ValueError("predator_state must have 'position' and 'velocity' keys")
        
        # Deep copy the initial state to avoid mutation
        self.current_state = copy.deepcopy(initial_state)
        
        # Store policy
        self.policy = policy
    
    def step(self) -> Dict[str, Any]:
        """
        Run one simulation step using policy and return updated state
        
        Returns:
            Updated state dict with same structure as input
        """
        if self.current_state is None or self.policy is None:
            raise ValueError("State manager not initialized. Call init() first.")
        
        # Convert current state to structured format for policy
        structured_inputs = self._convert_state_to_structured_inputs(self.current_state)
        
        # Get policy action
        policy_outputs = self.policy.get_action(structured_inputs)
        
        # Convert policy outputs to game actions
        actions = self.action_processor.process_action(policy_outputs)
        predator_action = {
            'force_x': actions[0],
            'force_y': actions[1]
        }
        
        # Run simulation step
        step_result = simulation_step(
            self.current_state['boids_states'],
            self.current_state['predator_state'],
            predator_action,
            self.current_state['canvas_width'],
            self.current_state['canvas_height']
        )
        
        # Remove caught boids from the state
        caught_boids = step_result['caught_boids']
        new_boids_states = step_result['boids_states']
        
        # Remove caught boids in reverse order to maintain indices
        for i in reversed(caught_boids):
            new_boids_states.pop(i)
        
        # Update current state
        self.current_state = {
            'boids_states': new_boids_states,
            'predator_state': step_result['predator_state'],
            'canvas_width': self.current_state['canvas_width'],
            'canvas_height': self.current_state['canvas_height']
        }
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state without running a step
        
        Returns:
            Current state dict (deep copy to prevent mutation)
        """
        if self.current_state is None:
            raise ValueError("State manager not initialized. Call init() first.")
        
        return copy.deepcopy(self.current_state)
    
    def _convert_state_to_structured_inputs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw state to structured inputs for policy
        
        Args:
            state: Raw state dict
            
        Returns:
            Structured inputs dict for policy
        """
        # Convert boids states to the format expected by InputProcessor
        boids_for_processor = []
        for boid_state in state['boids_states']:
            boids_for_processor.append({
                'position': boid_state['position'],
                'velocity': boid_state['velocity']
            })
        
        # Use InputProcessor to convert to structured format
        structured_inputs = self.input_processor.process_inputs(
            boids_for_processor,
            state['predator_state']['position'],
            state['predator_state']['velocity'],
            state['canvas_width'],
            state['canvas_height']
        )
        
        return structured_inputs 