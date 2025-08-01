"""
Simple, unified evaluator interface for policy evaluation.

This module provides a clean API for evaluating policies that can be used
by RL training, testing, or any other system that needs policy evaluation.

Example usage:
    from evaluation import Evaluator
    
    # Quick evaluation
    evaluator = Evaluator()
    score = evaluator.evaluate(policy)  # Returns catch rate
    
    # Detailed evaluation
    results = evaluator.evaluate(policy, detailed=True)
    print(f"Catch rate: {results['catch_rate']:.1%}")
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.state_manager.state_manager import StateManager
from simulation.random_state_generator.random_state_generator import generate_random_state


class Evaluator:
    """
    Simple evaluator for policy performance.
    
    Provides a minimal interface focused on ease of use:
    - evaluate(policy) -> float: Returns catch rate
    - evaluate(policy, detailed=True) -> dict: Returns detailed metrics
    """
    
    def __init__(
        self,
        num_episodes: int = 10,
        max_steps: int = 1000,
        scenarios: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize evaluator with default settings.
        
        Args:
            num_episodes: Episodes to run per scenario (default: 10)
            max_steps: Max steps per episode (default: 1000)
            scenarios: List of scenario names or None for default set
            seed: Random seed for reproducibility
        """
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.seed = seed
        
        # Define evaluation scenarios
        self._scenarios = {
            'easy': {'num_boids': 5, 'width': 400, 'height': 300},
            'medium': {'num_boids': 10, 'width': 600, 'height': 400},
            'hard': {'num_boids': 20, 'width': 800, 'height': 600},
            'dense': {'num_boids': 15, 'width': 400, 'height': 300},
            'sparse': {'num_boids': 10, 'width': 1000, 'height': 800}
        }
        
        # Select scenarios to use
        if scenarios is None:
            self.scenarios = ['easy', 'medium', 'dense']  # Default balanced set
        else:
            self.scenarios = scenarios
    
    def evaluate(
        self,
        policy: Any,
        detailed: bool = False,
        verbose: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Evaluate a policy.
        
        Args:
            policy: Policy object with get_action(structured_inputs) method
            detailed: If True, return detailed metrics dict. If False, return catch rate
            verbose: Print progress during evaluation
            
        Returns:
            If detailed=False: float catch rate (0.0 to 1.0)
            If detailed=True: dict with comprehensive metrics
            
        Example:
            # Simple usage
            catch_rate = evaluator.evaluate(policy)
            
            # Detailed usage
            results = evaluator.evaluate(policy, detailed=True)
            print(f"Catch rate: {results['catch_rate']:.1%}")
            print(f"Success rate: {results['success_rate']:.1%}")
        """
        all_metrics = []
        
        # Set base seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Evaluate on each scenario
        for scenario_name in self.scenarios:
            if scenario_name not in self._scenarios:
                continue
                
            scenario = self._scenarios[scenario_name]
            
            if verbose:
                print(f"Evaluating {scenario_name} scenario...")
            
            # Run episodes for this scenario
            scenario_metrics = self._evaluate_scenario(
                policy,
                scenario,
                scenario_name,
                verbose
            )
            
            all_metrics.extend(scenario_metrics)
        
        # Calculate aggregate metrics
        if detailed:
            return self._calculate_detailed_metrics(all_metrics)
        else:
            # Return simple catch rate
            total_caught = sum(m['boids_caught'] for m in all_metrics)
            total_boids = sum(m['total_boids'] for m in all_metrics)
            return total_caught / total_boids if total_boids > 0 else 0.0
    
    def evaluate_multiple(
        self,
        policies: Dict[str, Any],
        verbose: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple policies and return comparative results.
        
        Args:
            policies: Dict mapping policy names to policy objects
            verbose: Print progress
            
        Returns:
            Dict mapping policy names to their detailed results
            
        Example:
            results = evaluator.evaluate_multiple({
                'random': RandomPolicy(),
                'greedy': ClosestPursuitPolicy(),
                'learned': MyRLPolicy()
            })
            
            for name, metrics in results.items():
                print(f"{name}: {metrics['catch_rate']:.1%}")
        """
        results = {}
        
        for name, policy in policies.items():
            if verbose:
                print(f"\nEvaluating {name}...")
            
            results[name] = self.evaluate(policy, detailed=True, verbose=verbose)
        
        return results
    
    def quick_test(self, policy: Any) -> bool:
        """
        Quick test to verify policy works correctly.
        
        Args:
            policy: Policy to test
            
        Returns:
            True if policy works, False otherwise
        """
        try:
            # Test with sample input
            test_input = {
                'context': {'canvasWidth': 0.8, 'canvasHeight': 0.6},
                'predator': {'velX': 0.1, 'velY': -0.2},
                'boids': [
                    {'relX': 0.1, 'relY': 0.3, 'velX': 0.5, 'velY': -0.1},
                    {'relX': -0.2, 'relY': 0.1, 'velX': -0.3, 'velY': 0.4}
                ]
            }
            
            action = policy.get_action(test_input)
            
            # Verify output format
            if not isinstance(action, (list, tuple)) or len(action) != 2:
                return False
            
            # Verify output range
            if not all(-1 <= a <= 1 for a in action):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _evaluate_scenario(
        self,
        policy: Any,
        scenario: Dict[str, Any],
        scenario_name: str,
        verbose: bool
    ) -> List[Dict[str, Any]]:
        """Evaluate policy on a single scenario."""
        metrics = []
        
        for episode in range(self.num_episodes):
            # Generate initial state with unique seed
            episode_seed = hash((scenario_name, episode, self.seed)) % 2**32
            
            initial_state = generate_random_state(
                num_boids=scenario['num_boids'],
                canvas_width=scenario['width'],
                canvas_height=scenario['height'],
                seed=episode_seed
            )
            
            # Run episode
            episode_metrics = self._run_episode(
                policy,
                initial_state,
                scenario['num_boids']
            )
            
            episode_metrics['scenario'] = scenario_name
            metrics.append(episode_metrics)
            
            if verbose and (episode + 1) % 5 == 0:
                print(f"  Completed {episode + 1}/{self.num_episodes} episodes")
        
        return metrics
    
    def _run_episode(
        self,
        policy: Any,
        initial_state: Dict[str, Any],
        total_boids: int
    ) -> Dict[str, Any]:
        """Run a single episode and return metrics."""
        # Initialize simulation
        state_manager = StateManager()
        state_manager.init(initial_state, policy)
        
        # Track metrics
        steps = 0
        boids_caught = 0
        
        # Run episode
        while steps < self.max_steps and len(state_manager.current_state['boids_states']) > 0:
            # Take step
            result = state_manager.step()
            
            # Count catches
            if 'caught_boids' in result and result['caught_boids']:
                boids_caught += len(result['caught_boids'])
            
            steps += 1
        
        # Calculate metrics
        success = len(state_manager.current_state['boids_states']) == 0
        catch_rate = boids_caught / total_boids
        efficiency = boids_caught / steps if steps > 0 else 0
        
        return {
            'boids_caught': boids_caught,
            'total_boids': total_boids,
            'steps': steps,
            'success': success,
            'catch_rate': catch_rate,
            'efficiency': efficiency
        }
    
    def _calculate_detailed_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics from episode results."""
        # Aggregate metrics
        total_episodes = len(all_metrics)
        total_caught = sum(m['boids_caught'] for m in all_metrics)
        total_boids = sum(m['total_boids'] for m in all_metrics)
        total_steps = sum(m['steps'] for m in all_metrics)
        successful_episodes = sum(m['success'] for m in all_metrics)
        
        # Calculate rates
        catch_rate = total_caught / total_boids if total_boids > 0 else 0.0
        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0
        avg_efficiency = total_caught / total_steps if total_steps > 0 else 0.0
        
        # Per-scenario breakdown
        scenario_metrics = {}
        for scenario in set(m['scenario'] for m in all_metrics):
            scenario_data = [m for m in all_metrics if m['scenario'] == scenario]
            scenario_metrics[scenario] = {
                'catch_rate': np.mean([m['catch_rate'] for m in scenario_data]),
                'success_rate': np.mean([m['success'] for m in scenario_data]),
                'episodes': len(scenario_data)
            }
        
        return {
            # Primary metrics
            'catch_rate': catch_rate,
            'success_rate': success_rate,
            'efficiency': avg_efficiency,
            
            # Statistics
            'catch_rate_std': np.std([m['catch_rate'] for m in all_metrics]),
            'total_episodes': total_episodes,
            'total_boids': total_boids,
            'total_caught': total_caught,
            
            # Scenario breakdown
            'scenarios': scenario_metrics,
            
            # Raw data for further analysis
            'raw_metrics': all_metrics
        }


# Convenience functions for even simpler usage
def evaluate(policy: Any, episodes: int = 10) -> float:
    """
    Quick evaluation function that returns catch rate.
    
    Args:
        policy: Policy to evaluate
        episodes: Episodes per scenario (default: 10)
        
    Returns:
        Catch rate as float (0.0 to 1.0)
        
    Example:
        from evaluation import evaluate
        catch_rate = evaluate(my_policy)
        print(f"Policy achieves {catch_rate:.1%} catch rate")
    """
    evaluator = Evaluator(num_episodes=episodes)
    return evaluator.evaluate(policy)


def compare(policies: Dict[str, Any], episodes: int = 10) -> None:
    """
    Compare multiple policies and print results.
    
    Args:
        policies: Dict mapping names to policy objects
        episodes: Episodes per scenario
        
    Example:
        from evaluation import compare
        compare({
            'baseline': RandomPolicy(),
            'learned': MyRLPolicy()
        })
    """
    evaluator = Evaluator(num_episodes=episodes)
    results = evaluator.evaluate_multiple(policies)
    
    # Print comparison table
    print(f"\n{'Policy':<20} {'Catch Rate':<15} {'Success Rate':<15}")
    print("-" * 50)
    
    for name, metrics in results.items():
        catch_rate = f"{metrics['catch_rate']*100:.1f}%"
        success_rate = f"{metrics['success_rate']*100:.1f}%"
        print(f"{name:<20} {catch_rate:<15} {success_rate:<15}")


# For backward compatibility
def create_evaluator(**kwargs) -> Evaluator:
    """Create an evaluator with custom settings."""
    return Evaluator(**kwargs)