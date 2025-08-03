"""
TorchRL Environment Wrapper for Boids Simulation

Clean architecture environment that:
- Uses simulation_step() directly (no StateManager)
- No policy management or temporary policies  
- Clean separation of concerns
- Eliminates the TemporaryRLPolicy anti-pattern
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Composite, Unbounded
from tensordict import TensorDict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.random_state_generator import RandomStateGenerator
from simulation.runtime.simulation_runtime import simulation_step
from simulation.processors import InputProcessor
from rewards.reward_processor import RewardProcessor
from config.constants import CONSTANTS


class BoidsEnvironment(EnvBase):
    """
    Clean TorchRL environment for boids simulation.
    
    This environment:
    - Uses simulation_step() directly for physics (no StateManager)
    - NO policy management (policies are external)
    - NO TemporaryRLPolicy or policy swapping
    - Clean, simple, and maintainable architecture
    - Actions are 2D continuous forces for the predator
    - Rewards calculated using existing reward processor
    - Episodes terminate when all boids caught or max steps reached
    """
    
    def __init__(
        self,
        canvas_width: int = 800,
        canvas_height: int = 600,
        num_boids: int = 20,
        max_steps: int = 1000,
        max_boids_for_model: int = 50,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(device=device)
        
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.num_boids = num_boids
        self.max_steps = max_steps
        self.max_boids_for_model = max_boids_for_model
        
        # Store device for consistent usage
        self._device = device if device is not None else torch.device('cpu')
        
        # Initialize components (NO StateManager!)
        self.state_generator = RandomStateGenerator(seed=seed)
        self.input_processor = InputProcessor()
        self.reward_processor = RewardProcessor()
        
        # Episode tracking
        self.current_step = 0
        self.initial_boid_count = num_boids
        self.current_boids_states = []
        self.current_predator_state = {}
        
        # Define action and observation specs
        self._make_specs()
        
        # Set seed if provided
        if seed is not None:
            self.set_seed(seed)
    
    def _make_specs(self) -> None:
        """Define the specs for observations and actions"""
        
        # Action spec: 2D continuous action in [-1, 1]
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=torch.float32,
            device=self._device,
        )
        
        # Observation spec: Normalized observations bounded in reasonable ranges
        # Format: [canvas_w, canvas_h, pred_vx, pred_vy, boid1_data, boid2_data, ...]
        # Each boid: [rel_x, rel_y, vel_x, vel_y]
        # All values are normalized, most should be in [-1, 1] range
        obs_size = 4 + (self.max_boids_for_model * 4)  # context(2) + predator(2) + boids
        
        self.observation_spec = Composite(
            observation=Bounded(
                low=-2.0,  # Allow some range beyond [-1,1] for safety
                high=2.0,
                shape=(obs_size,),
                dtype=torch.float32,
                device=self._device,
            ),
            shape=(),
        )
        
        # Reward and done specs
        self.reward_spec = Unbounded(
            shape=(1,),
            dtype=torch.float32,
            device=self._device,
        )
        
        self.done_spec = Composite(
            done=Unbounded(
                shape=(1,),
                dtype=torch.bool,
                device=self._device,
            ),
            terminated=Unbounded(
                shape=(1,),
                dtype=torch.bool,
                device=self._device,
            ),
            truncated=Unbounded(
                shape=(1,),
                dtype=torch.bool,
                device=self._device,
            ),
            shape=(),
        )
    
    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        """Reset the environment to a new episode"""
        
        # Generate random initial state
        initial_state = self.state_generator.generate_scattered_state(
            self.num_boids, self.canvas_width, self.canvas_height
        )
        
        # Store current state directly (NO StateManager!)
        self.current_boids_states = initial_state['boids_states'].copy()
        self.current_predator_state = initial_state['predator_state'].copy()
        
        # Reset episode tracking
        self.current_step = 0
        self.initial_boid_count = len(self.current_boids_states)
        
        # Convert to observation tensor
        obs_tensor = self._state_to_tensor()
        
        return TensorDict(
            {
                "observation": obs_tensor,
                "done": torch.tensor([False], dtype=torch.bool, device=self._device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self._device),
                "truncated": torch.tensor([False], dtype=torch.bool, device=self._device),
            },
            batch_size=(),
            device=self._device,
        )
    
    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Execute one step in the environment - handles both single and batch processing"""
        
        # Extract action from tensordict - handle both single env and batch
        action = tensordict["action"]
        
        # For single environment (non-parallel), action should be 1D
        if action.dim() == 1:
            # Single action case
            batch_size = 1
            action = action.unsqueeze(0)  # Add batch dimension for consistency
        else:
            # Batch case (parallel environments) - not supported yet, but prepare for it
            batch_size = action.shape[0]
            if batch_size > 1:
                raise NotImplementedError("Batch processing not yet supported in this environment")
        
        # Process single action (squeeze back to 1D for compatibility)
        action = action.squeeze(0)
        
        # Convert RL action to predator action format (keep on device, avoid .item() for efficiency)
        predator_action = {
            'force_x': (action[0] * CONSTANTS.PREDATOR_MAX_FORCE).item(),
            'force_y': (action[1] * CONSTANTS.PREDATOR_MAX_FORCE).item()
        }
        
        # Store pre-step state for reward calculation
        pre_step_structured = self._get_structured_state()
        
        # Run simulation step DIRECTLY (no StateManager, no TemporaryRLPolicy!)
        result = simulation_step(
            boids_states=self.current_boids_states,
            predator_state=self.current_predator_state,
            predator_action=predator_action,
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height
        )
        
        # Update state from simulation result
        self.current_boids_states = result['boids_states']
        self.current_predator_state = result['predator_state']
        caught_boids_indices = result['caught_boids']
        
        # Extract caught boid IDs BEFORE modifying the list
        caught_boid_ids = []
        for i in caught_boids_indices:
            if i < len(self.current_boids_states):
                caught_boid_ids.append(self.current_boids_states[i]['id'])
        
        # Now safely remove caught boids (in reverse order to maintain indices)
        for i in reversed(sorted(caught_boids_indices)):
            if i < len(self.current_boids_states):
                self.current_boids_states.pop(i)
        
        # Calculate reward - avoid unnecessary tensor conversions during forward pass
        reward_input = {
            'state': pre_step_structured,
            'action': [action[0].item(), action[1].item()],  # Convert to list more efficiently
            'caughtBoids': caught_boid_ids
        }
        reward_result = self.reward_processor.calculate_step_reward(reward_input)
        reward = torch.tensor([reward_result['total']], dtype=torch.float32, device=self._device)
        
        # Update step count
        self.current_step += 1
        
        # Check termination conditions
        terminated = len(self.current_boids_states) == 0  # All boids caught
        truncated = self.current_step >= self.max_steps  # Max steps reached
        done = terminated or truncated
        
        # Convert state to observation tensor
        obs_tensor = self._state_to_tensor()
        
        # Return new TensorDict with proper structure
        return TensorDict(
            {
                "observation": obs_tensor,
                "reward": reward,
                "done": torch.tensor([done], dtype=torch.bool, device=self._device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self._device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self._device),
            },
            batch_size=(),
            device=self._device,
        )
    
    def _state_to_tensor(self) -> torch.Tensor:
        """Convert current simulation state to flat tensor observation using InputProcessor"""
        
        # Use the existing InputProcessor for consistent normalization
        structured_inputs = self._get_structured_state()
        
        # Flatten structured inputs to tensor format expected by RL policy
        obs = []
        
        # Context information (canvas dimensions - already normalized)
        obs.extend([
            structured_inputs['context']['canvasWidth'],
            structured_inputs['context']['canvasHeight']
        ])
        
        # Predator information (velocities - already normalized)
        obs.extend([
            structured_inputs['predator']['velX'],
            structured_inputs['predator']['velY']
        ])
        
        # Boid information (already normalized by InputProcessor)
        num_boids_to_add = min(len(structured_inputs['boids']), self.max_boids_for_model)
        
        for i in range(num_boids_to_add):
            boid = structured_inputs['boids'][i]
            obs.extend([
                boid['relX'],  # Already normalized relative position
                boid['relY'],  # Already normalized relative position  
                boid['velX'],  # Already normalized velocity
                boid['velY']   # Already normalized velocity
            ])
        
        # Pad with zeros if fewer boids than max
        while len(obs) < 4 + (self.max_boids_for_model * 4):
            obs.extend([0.0, 0.0, 0.0, 0.0])
        
        return torch.tensor(obs, dtype=torch.float32, device=self._device)
    
    def _get_structured_state(self) -> Dict[str, Any]:
        """Convert current state to structured format for reward calculation"""
        
        # Convert boids to format expected by InputProcessor
        boids_for_processor = []
        for boid in self.current_boids_states:
            boids_for_processor.append({
                'id': boid['id'],
                'position': boid['position'],
                'velocity': boid['velocity']
            })
        
        return self.input_processor.process_inputs(
            boids_for_processor,
            self.current_predator_state['position'],
            self.current_predator_state['velocity'],
            self.canvas_width,
            self.canvas_height
        )
    
    def set_seed(self, seed: Optional[int] = None) -> None:
        """Set the seed for the environment"""
        if seed is not None:
            self.state_generator = RandomStateGenerator(seed=seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def _set_seed(self, seed: Optional[int] = None) -> None:
        """TorchRL required method - delegate to set_seed"""
        self.set_seed(seed)