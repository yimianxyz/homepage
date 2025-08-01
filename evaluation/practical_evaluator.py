"""Practical evaluation system based on actual simulation dynamics."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.state_manager.state_manager import StateManager
from simulation.random_state_generator.random_state_generator import generate_random_state
from config.constants import SimulationConstants


class PracticalEvaluator:
    """Evaluation system designed for the realities of the boid simulation."""
    
    def __init__(self):
        self.constants = SimulationConstants()
        
    def evaluate_policy(
        self, 
        policy: Any,
        test_scenarios: List[Dict[str, Any]] = None,
        episodes_per_scenario: int = 10,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Evaluate policy across multiple practical scenarios.
        
        Args:
            policy: Policy to evaluate
            test_scenarios: List of test scenarios (uses defaults if None)
            episodes_per_scenario: Number of episodes per scenario
            verbose: Print progress
            
        Returns:
            Comprehensive evaluation results
        """
        if test_scenarios is None:
            test_scenarios = self.get_default_scenarios()
        
        results = {
            'overall_metrics': {},
            'scenario_results': {},
            'performance_profile': {}
        }
        
        all_metrics = []
        
        for scenario in test_scenarios:
            if verbose:
                print(f"\nEvaluating scenario: {scenario['name']}")
            
            scenario_metrics = self._evaluate_scenario(
                policy, 
                scenario, 
                episodes_per_scenario
            )
            
            results['scenario_results'][scenario['name']] = scenario_metrics
            all_metrics.extend(scenario_metrics['episodes'])
        
        # Calculate overall metrics
        results['overall_metrics'] = self._calculate_overall_metrics(all_metrics)
        
        # Create performance profile
        results['performance_profile'] = self._create_performance_profile(results['scenario_results'])
        
        return results
    
    def get_default_scenarios(self) -> List[Dict[str, Any]]:
        """Get default test scenarios based on simulation insights."""
        return [
            {
                'name': 'easy_small',
                'num_boids': 5,
                'canvas_width': 400,
                'canvas_height': 300,
                'max_steps': 500,
                'description': 'Easy scenario - few boids, small arena'
            },
            {
                'name': 'medium_standard',
                'num_boids': 10,
                'canvas_width': 600,
                'canvas_height': 400,
                'max_steps': 1000,
                'description': 'Medium difficulty - standard setup'
            },
            {
                'name': 'hard_large',
                'num_boids': 20,
                'canvas_width': 800,
                'canvas_height': 600,
                'max_steps': 1500,
                'description': 'Hard scenario - many boids, large arena'
            },
            {
                'name': 'dense_challenge',
                'num_boids': 15,
                'canvas_width': 400,
                'canvas_height': 300,
                'max_steps': 800,
                'description': 'Dense scenario - high boid density'
            },
            {
                'name': 'sparse_challenge',
                'num_boids': 10,
                'canvas_width': 1000,
                'canvas_height': 800,
                'max_steps': 2000,
                'description': 'Sparse scenario - low boid density'
            }
        ]
    
    def _evaluate_scenario(
        self, 
        policy: Any, 
        scenario: Dict[str, Any], 
        num_episodes: int
    ) -> Dict[str, Any]:
        """Evaluate policy on a specific scenario."""
        episodes = []
        
        for episode_idx in range(num_episodes):
            # Different seed for each episode
            seed = 1000 * hash(scenario['name']) + episode_idx
            
            metrics = self._run_episode(
                policy,
                scenario['num_boids'],
                scenario['canvas_width'],
                scenario['canvas_height'],
                scenario['max_steps'],
                seed
            )
            
            episodes.append(metrics)
        
        # Calculate scenario statistics
        catch_rates = [e['catch_rate'] for e in episodes]
        efficiencies = [e['efficiency'] for e in episodes]
        early_catches = [e['early_catch_rate'] for e in episodes]
        
        return {
            'episodes': episodes,
            'avg_catch_rate': np.mean(catch_rates),
            'std_catch_rate': np.std(catch_rates),
            'avg_efficiency': np.mean(efficiencies),
            'avg_early_catch_rate': np.mean(early_catches),
            'success_rate': sum(e['success'] for e in episodes) / num_episodes,
            'scenario_info': scenario
        }
    
    def _run_episode(
        self,
        policy: Any,
        num_boids: int,
        canvas_width: float,
        canvas_height: float,
        max_steps: int,
        seed: int
    ) -> Dict[str, Any]:
        """Run single episode and collect detailed metrics."""
        
        # Generate initial state
        initial_state = generate_random_state(
            num_boids=num_boids,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            seed=seed
        )
        
        # Initialize simulation
        state_manager = StateManager()
        state_manager.init(initial_state, policy)
        
        # Tracking variables
        steps = 0
        boids_caught = 0
        catch_times = []
        distances_to_nearest = []
        
        # Run episode
        while steps < max_steps and len(state_manager.current_state['boids_states']) > 0:
            # Record distance to nearest boid before step
            dist = self._get_nearest_boid_distance(state_manager.current_state)
            if dist is not None:
                distances_to_nearest.append(dist)
            
            # Step simulation
            result = state_manager.step()
            
            # Track catches
            if 'caught_boids' in result and result['caught_boids']:
                for _ in result['caught_boids']:
                    boids_caught += 1
                    catch_times.append(steps)
            
            steps += 1
        
        # Calculate metrics
        success = len(state_manager.current_state['boids_states']) == 0
        catch_rate = boids_caught / num_boids
        efficiency = boids_caught / steps if steps > 0 else 0
        
        # Early game performance (first 25% of episode)
        early_steps = max_steps // 4
        early_catches = sum(1 for t in catch_times if t < early_steps)
        early_catch_rate = early_catches / num_boids
        
        # Average pursuit distance
        avg_pursuit_distance = np.mean(distances_to_nearest) if distances_to_nearest else 0
        
        return {
            'steps': steps,
            'boids_caught': boids_caught,
            'success': success,
            'catch_rate': catch_rate,
            'efficiency': efficiency,
            'early_catch_rate': early_catch_rate,
            'catch_times': catch_times,
            'avg_pursuit_distance': avg_pursuit_distance,
            'final_boids': len(state_manager.current_state['boids_states'])
        }
    
    def _get_nearest_boid_distance(self, state: Dict[str, Any]) -> Optional[float]:
        """Calculate distance to nearest boid."""
        if not state['boids_states']:
            return None
            
        px = state['predator_state']['position']['x']
        py = state['predator_state']['position']['y']
        
        min_dist = float('inf')
        for boid in state['boids_states']:
            bx = boid['position']['x']
            by = boid['position']['y']
            dist = np.sqrt((px - bx)**2 + (py - by)**2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _calculate_overall_metrics(self, all_episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall metrics across all episodes."""
        return {
            'total_episodes': len(all_episodes),
            'overall_catch_rate': np.mean([e['catch_rate'] for e in all_episodes]),
            'overall_efficiency': np.mean([e['efficiency'] for e in all_episodes]),
            'overall_success_rate': sum(e['success'] for e in all_episodes) / len(all_episodes),
            'avg_pursuit_distance': np.mean([e['avg_pursuit_distance'] for e in all_episodes]),
            'catch_rate_std': np.std([e['catch_rate'] for e in all_episodes]),
            'percentiles': {
                '25th': np.percentile([e['catch_rate'] for e in all_episodes], 25),
                '50th': np.percentile([e['catch_rate'] for e in all_episodes], 50),
                '75th': np.percentile([e['catch_rate'] for e in all_episodes], 75),
                '90th': np.percentile([e['catch_rate'] for e in all_episodes], 90)
            }
        }
    
    def _create_performance_profile(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a performance profile showing strengths/weaknesses."""
        profile = {}
        
        # Extract key metrics for each scenario type
        for scenario_name, results in scenario_results.items():
            if 'easy' in scenario_name:
                profile['easy_performance'] = results['avg_catch_rate']
            elif 'hard' in scenario_name:
                profile['hard_performance'] = results['avg_catch_rate']
            elif 'dense' in scenario_name:
                profile['dense_performance'] = results['avg_catch_rate']
            elif 'sparse' in scenario_name:
                profile['sparse_performance'] = results['avg_catch_rate']
        
        # Calculate adaptability (variance across scenarios)
        catch_rates = [r['avg_catch_rate'] for r in scenario_results.values()]
        profile['adaptability'] = 1.0 - np.std(catch_rates) / (np.mean(catch_rates) + 0.001)
        
        # Early game performance
        early_rates = [r['avg_early_catch_rate'] for r in scenario_results.values()]
        profile['early_game_strength'] = np.mean(early_rates)
        
        return profile
    
    def compare_policies(
        self,
        policies: Dict[str, Any],
        test_scenarios: List[Dict[str, Any]] = None,
        episodes_per_scenario: int = 10,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Compare multiple policies fairly."""
        
        if test_scenarios is None:
            test_scenarios = self.get_default_scenarios()
        
        # Evaluate each policy
        policy_results = {}
        for name, policy in policies.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating: {name}")
                print('='*60)
            
            policy_results[name] = self.evaluate_policy(
                policy, 
                test_scenarios, 
                episodes_per_scenario,
                verbose
            )
        
        # Create comparison
        comparison = {
            'policy_results': policy_results,
            'rankings': self._rank_policies(policy_results),
            'head_to_head': self._head_to_head_comparison(policy_results)
        }
        
        return comparison
    
    def _rank_policies(self, policy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank policies by different criteria."""
        rankings = {}
        
        # Overall catch rate ranking
        catch_rates = {
            name: results['overall_metrics']['overall_catch_rate'] 
            for name, results in policy_results.items()
        }
        rankings['by_catch_rate'] = sorted(catch_rates.items(), key=lambda x: x[1], reverse=True)
        
        # Efficiency ranking
        efficiencies = {
            name: results['overall_metrics']['overall_efficiency']
            for name, results in policy_results.items()
        }
        rankings['by_efficiency'] = sorted(efficiencies.items(), key=lambda x: x[1], reverse=True)
        
        # Adaptability ranking
        adaptabilities = {
            name: results['performance_profile'].get('adaptability', 0)
            for name, results in policy_results.items()
        }
        rankings['by_adaptability'] = sorted(adaptabilities.items(), key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _head_to_head_comparison(self, policy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create pairwise comparisons between policies."""
        comparisons = {}
        
        policy_names = list(policy_results.keys())
        for i, name1 in enumerate(policy_names):
            for name2 in policy_names[i+1:]:
                key = f"{name1}_vs_{name2}"
                
                # Compare overall metrics
                metrics1 = policy_results[name1]['overall_metrics']
                metrics2 = policy_results[name2]['overall_metrics']
                
                comparisons[key] = {
                    'catch_rate_diff': metrics1['overall_catch_rate'] - metrics2['overall_catch_rate'],
                    'efficiency_diff': metrics1['overall_efficiency'] - metrics2['overall_efficiency'],
                    'winner': name1 if metrics1['overall_catch_rate'] > metrics2['overall_catch_rate'] else name2
                }
        
        return comparisons
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*60)
        print("POLICY EVALUATION SUMMARY")
        print("="*60)
        
        metrics = results['overall_metrics']
        print(f"\nOverall Performance ({metrics['total_episodes']} episodes):")
        print(f"  Catch Rate: {metrics['overall_catch_rate']:.1%} ± {metrics['catch_rate_std']:.1%}")
        print(f"  Efficiency: {metrics['overall_efficiency']:.4f} boids/step")
        print(f"  Success Rate: {metrics['overall_success_rate']:.1%}")
        print(f"  Avg Pursuit Distance: {metrics['avg_pursuit_distance']:.1f}")
        
        print(f"\nCatch Rate Percentiles:")
        for p, v in metrics['percentiles'].items():
            print(f"  {p}: {v:.1%}")
        
        print(f"\nPerformance Profile:")
        profile = results['performance_profile']
        for key, value in profile.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
        
        print("\nScenario Breakdown:")
        for name, scenario_result in results['scenario_results'].items():
            info = scenario_result['scenario_info']
            print(f"\n  {name} ({info['num_boids']} boids, {info['canvas_width']}x{info['canvas_height']}):")
            print(f"    Catch Rate: {scenario_result['avg_catch_rate']:.1%}")
            print(f"    Success Rate: {scenario_result['success_rate']:.1%}")
            print(f"    Early Game: {scenario_result['avg_early_catch_rate']:.1%} caught in first 25%")
        
        print("="*60)