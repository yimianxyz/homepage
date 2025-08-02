"""
PPO Policy Wrapper - Adapter for existing evaluation system

This wrapper adapts PPO-trained models to work with the existing PolicyEvaluator
system, ensuring consistent evaluation across all policy types.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Union
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PPOPolicyWrapper:
    """
    Wrapper that adapts PPO model to the standard policy interface
    
    This allows PPO-trained models to be evaluated using the existing
    PolicyEvaluator system without any modifications to the evaluation code.
    """
    
    def __init__(self, ppo_model, device: torch.device = None):
        """
        Initialize PPO policy wrapper
        
        Args:
            ppo_model: Trained PPO model from Stable Baselines3
            device: PyTorch device for inference
        """
        if ppo_model is None:
            raise ValueError("PPO model cannot be None")
        
        self.ppo_model = ppo_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set model to evaluation mode
        if hasattr(self.ppo_model.policy, 'eval'):
            self.ppo_model.policy.eval()
        
        print(f"✓ PPO Policy Wrapper initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model type: {type(self.ppo_model).__name__}")
    
    def get_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get action using PPO model (compatible with existing evaluation system)
        
        Args:
            structured_inputs: Standard structured input format:
                - context: {canvasWidth: float, canvasHeight: float}
                - predator: {velX: float, velY: float}
                - boids: [{relX: float, relY: float, velX: float, velY: float}, ...]
                
        Returns:
            Action list [x, y] in [-1, 1] range
        """
        # Validate structured inputs
        if not isinstance(structured_inputs, dict):
            raise ValueError(f"Expected dict structured_inputs, got {type(structured_inputs)}")
        
        required_keys = ['context', 'predator', 'boids']
        for key in required_keys:
            if key not in structured_inputs:
                raise ValueError(f"Missing required key '{key}' in structured_inputs")
        
        # Convert structured input to flat observation for PPO model
        flat_obs = self._structured_to_flat(structured_inputs)
        
        # Get action from PPO model
        action, _ = self.ppo_model.predict(flat_obs, deterministic=True)
        
        # Ensure action is in [-1, 1] range and convert to list
        action = np.clip(action, -1.0, 1.0)
        return action.tolist()
    
    def get_normalized_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get normalized action (deprecated - use get_action instead)
        
        Args:
            structured_inputs: Standard structured input format
            
        Returns:
            Normalized action [x, y] in [-1, 1] range
        """
        return self.get_action(structured_inputs)
    
    def _structured_to_flat(self, structured_inputs: Dict[str, Any], max_boids: int = 50) -> np.ndarray:
        """
        Convert structured observation to flat array for PPO model
        
        Args:
            structured_inputs: Structured observation
            max_boids: Maximum number of boids for padding
            
        Returns:
            Flat observation array
        """
        # Extract components
        context = structured_inputs['context']
        predator = structured_inputs['predator']
        boids = structured_inputs['boids']
        
        # Validate component structure
        if 'canvasWidth' not in context or 'canvasHeight' not in context:
            raise ValueError("Invalid context structure - missing canvas dimensions")
        if 'velX' not in predator or 'velY' not in predator:
            raise ValueError("Invalid predator structure - missing velocity")
        
        # Start with context and predator info
        flat_obs = [
            float(context['canvasWidth']),
            float(context['canvasHeight']),
            float(predator['velX']),
            float(predator['velY']),
            float(len(boids))  # Number of boids
        ]
        
        # Add boid features (pad to max_boids)
        for i in range(max_boids):
            if i < len(boids):
                boid = boids[i]
                # Validate boid structure
                required_boid_keys = ['relX', 'relY', 'velX', 'velY']
                for key in required_boid_keys:
                    if key not in boid:
                        raise ValueError(f"Missing required boid key '{key}' in boid {i}")
                        
                flat_obs.extend([
                    float(boid['relX']),
                    float(boid['relY']),
                    float(boid['velX']),
                    float(boid['velY'])
                ])
            else:
                # Padding for missing boids
                flat_obs.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(flat_obs, dtype=np.float32)

class PPOTransformerPolicyWrapper:
    """
    Wrapper specifically for transformer-based PPO policies
    
    This wrapper can directly use the structured observation format
    without conversion, providing more efficient inference.
    """
    
    def __init__(self, ppo_model, device: torch.device = None):
        """
        Initialize transformer PPO policy wrapper
        
        Args:
            ppo_model: PPO model with transformer policy
            device: PyTorch device for inference
        """
        if ppo_model is None:
            raise ValueError("PPO model cannot be None")
        
        self.ppo_model = ppo_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract transformer policy for direct access
        self.transformer_policy = self.ppo_model.policy
        
        # Validate that this is actually a transformer policy
        if not hasattr(self.transformer_policy, 'features_extractor'):
            raise ValueError("PPO model must have transformer features extractor")
        
        # Set to evaluation mode
        if hasattr(self.transformer_policy, 'eval'):
            self.transformer_policy.eval()
        
        print(f"✓ PPO Transformer Policy Wrapper initialized:")
        print(f"  Device: {self.device}")
        print(f"  Direct transformer access: Available")
    
    def get_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get action using transformer policy directly
        
        Args:
            structured_inputs: Standard structured input format
                
        Returns:
            Action list [x, y] in [-1, 1] range
        """
        # Validate structured inputs
        if not isinstance(structured_inputs, dict):
            raise ValueError(f"Expected dict structured_inputs, got {type(structured_inputs)}")
        
        required_keys = ['context', 'predator', 'boids']
        for key in required_keys:
            if key not in structured_inputs:
                raise ValueError(f"Missing required key '{key}' in structured_inputs")
        
        # Use transformer features extractor directly
        with torch.no_grad():
            features = self.transformer_policy.features_extractor.transformer_encoder(structured_inputs)
            
            # Get action from policy network
            action_mean = self.transformer_policy.mlp_extractor['policy'](features)
            
            # Get final action through action net if it exists
            if hasattr(self.transformer_policy, 'action_net'):
                action = self.transformer_policy.action_net(action_mean)
            else:
                # Use the policy head directly
                action = action_mean
            
            # Apply tanh and clip to ensure [-1, 1] range
            action = torch.tanh(action)
            action = torch.clamp(action, -1.0, 1.0)
        
        return action.cpu().numpy().tolist()
    
    def get_normalized_action(self, structured_inputs: Dict[str, Any]) -> List[float]:
        """
        Get normalized action (deprecated - use get_action instead)
        
        Args:
            structured_inputs: Standard structured input format
            
        Returns:
            Normalized action [x, y] in [-1, 1] range
        """
        return self.get_action(structured_inputs)

def create_policy_wrapper(ppo_model, wrapper_type: str = "auto") -> Union[PPOPolicyWrapper, PPOTransformerPolicyWrapper]:
    """
    Factory function to create appropriate policy wrapper
    
    Args:
        ppo_model: Trained PPO model
        wrapper_type: Type of wrapper ("auto", "standard", "transformer")
        
    Returns:
        Policy wrapper compatible with existing evaluation system
    """
    if ppo_model is None:
        raise ValueError("PPO model cannot be None")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if wrapper_type == "auto":
        # Auto-detect wrapper type based on model features
        if (hasattr(ppo_model.policy, 'features_extractor') and 
            hasattr(ppo_model.policy.features_extractor, 'transformer_encoder')):
            wrapper_type = "transformer"
        else:
            wrapper_type = "standard"
    
    if wrapper_type == "transformer":
        wrapper = PPOTransformerPolicyWrapper(ppo_model, device)
    elif wrapper_type == "standard":
        wrapper = PPOPolicyWrapper(ppo_model, device)
    else:
        raise ValueError(f"Unknown wrapper_type: {wrapper_type}")
    
    print(f"✓ Created {wrapper_type} policy wrapper")
    return wrapper

def evaluate_ppo_model(ppo_model, model_name: str = "PPO Model", wrapper_type: str = "auto"):
    """
    Evaluate PPO model using existing evaluation system
    
    Args:
        ppo_model: Trained PPO model
        model_name: Name for evaluation report
        wrapper_type: Type of wrapper to use
        
    Returns:
        Evaluation results
    """
    if ppo_model is None:
        raise ValueError("PPO model cannot be None")
    
    from evaluation.policy_evaluator import PolicyEvaluator
    
    # Create policy wrapper
    policy_wrapper = create_policy_wrapper(ppo_model, wrapper_type)
    
    # Create evaluator and run evaluation
    evaluator = PolicyEvaluator()
    results = evaluator.evaluate_policy(policy_wrapper, model_name)
    
    return results

if __name__ == "__main__":
    print("PPO Policy Wrapper - Production Version")
    print("All mocks and fallbacks removed")
    print("Requires actual trained PPO model for testing")