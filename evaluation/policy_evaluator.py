"""
Strategic Policy Evaluator - Advanced evaluation system for emergent flock strategies

This module provides strategic evaluation that captures:
- Multi-phase performance (early/mid/late game)
- Formation-specific strategies (scattered vs clustered boids)
- Strategic depth metrics (consistency, adaptability)
- Emergent behavior analysis over time

Optimized for speed while capturing long-term strategic behaviors.
"""

import time
import statistics
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.state_manager import StateManager
from simulation.random_state_generator import RandomStateGenerator


@dataclass
class StrategicResult:
    """Results for strategic evaluation with temporal and formation breakdown"""
    policy_name: str
    
    # Overall metrics (compatible with existing code)
    overall_catch_rate: float
    overall_std_catch_rate: float
    evaluation_time_seconds: float
    total_episodes: int
    successful_episodes: int
    
    # Strategic depth metrics
    early_phase_rate: float      # 0-100 steps performance
    mid_phase_rate: float        # 100-300 steps performance  
    late_phase_rate: float       # 300+ steps performance
    
    scattered_rate: float        # Scattered boids (herding challenge)
    clustered_rate: float        # Clustered boids (flock-breaking challenge)
    
    strategy_consistency: float  # Performance variance across phases (0-1)
    adaptability_score: float    # Performance across formations (0-1)
    
    # Strategic insights
    strategy_type: str          # "Early-game", "Late-game", "Consistent"
    formation_style: str        # "Adaptive", "Herding", "Flock-breaker"
    
    # Raw data
    all_catch_rates: List[float]
    episode_details: List[Dict]


# Alias for backward compatibility with existing code
EvaluationResult = StrategicResult


def run_strategic_episode(policy, initial_state: Dict, max_steps: int = 2500,
                         efficiency_target: float = 0.95) -> Dict[str, Any]:
    """
    Run a single strategic episode with phase tracking
    
    Args:
        policy: Policy to evaluate
        initial_state: Initial simulation state  
        max_steps: Maximum steps (2500 to match RL training horizon)
        efficiency_target: Stop early if this catch rate achieved
        
    Returns:
        Detailed episode results with phase breakdown
    """
    state_manager = StateManager()
    state_manager.init(initial_state, policy)
    
    initial_boids = len(initial_state['boids_states'])
    total_catches = 0
    step = 0
    
    # Track catches by phase
    phase_catches = {'early': 0, 'mid': 0, 'late': 0}
    
    # Run episode with phase tracking
    for step in range(max_steps):
        result = state_manager.step()
        
        # Count catches this step
        step_catches = 0
        if 'caught_boids' in result:
            step_catches = len(result['caught_boids'])
            total_catches += step_catches
        
        # Track by phase (adjusted for 2500-step episodes)
        if step < 500:  # Early: 0-500 steps
            phase_catches['early'] += step_catches
        elif step < 1500:  # Mid: 500-1500 steps
            phase_catches['mid'] += step_catches
        else:  # Late: 1500+ steps
            phase_catches['late'] += step_catches
        
        # Early termination if highly efficient
        current_rate = total_catches / initial_boids if initial_boids > 0 else 0.0
        if current_rate >= efficiency_target:
            break
        
        # Check if all boids caught
        if len(result['boids_states']) == 0:
            break
    
    # Calculate phase-specific rates
    early_rate = phase_catches['early'] / initial_boids if initial_boids > 0 else 0.0
    mid_rate = phase_catches['mid'] / initial_boids if initial_boids > 0 else 0.0
    late_rate = phase_catches['late'] / initial_boids if initial_boids > 0 else 0.0
    
    overall_rate = total_catches / initial_boids if initial_boids > 0 else 0.0
    
    return {
        'overall_rate': overall_rate,
        'total_catches': total_catches,
        'episode_length': step + 1,
        'early_rate': early_rate,
        'mid_rate': mid_rate, 
        'late_rate': late_rate,
        'terminated_early': step < max_steps - 1,
        'successful': True  # All episodes are successful
    }


class PolicyEvaluator:
    """
    Strategic Policy Evaluator - Optimized for emergent behavior analysis
    
    Key innovations:
    1. Multi-phase evaluation (early/mid/late game performance)
    2. Formation-diverse scenarios (scattered vs clustered boids)
    3. Strategic depth metrics (consistency, adaptability)
    4. Fast execution (~1 minute per policy)
    """
    
    def __init__(self):
        """Initialize strategic policy evaluator"""
        self.generator = RandomStateGenerator()
        
        # Strategic evaluation parameters (optimized for speed + depth)
        self.canvas_width = 400
        self.canvas_height = 300
        self.boid_count = 12  # Sweet spot for flock dynamics
        
        print(f"Strategic PolicyEvaluator initialized:")
        print(f"  Scenario: {self.canvas_width}×{self.canvas_height}, {self.boid_count} boids")
        print(f"  Focus: Emergent flock strategies and long-term performance")
        print(f"  Formations: Scattered (herding) vs Clustered (flock-breaking)")
        print(f"  Phases: Early/Mid/Late game analysis")
    
    def evaluate_policy(self, policy, policy_name: str = "Policy") -> StrategicResult:
        """
        Strategic evaluation with multi-phase and formation testing
        
        Evaluation design:
        - 5 strategic episodes (2500 steps each) - full horizon testing
        - Mix of scattered and clustered formations  
        - Total: ~12500 steps = consistent with RL training horizon
        
        Args:
            policy: Policy with get_action method
            policy_name: Name for reporting
            
        Returns:
            StrategicResult with comprehensive analysis
        """
        print(f"\n🧠 Strategic evaluation: {policy_name}")
        
        start_time = time.time()
        
        all_results = []
        scattered_results = []
        clustered_results = []
        
        # Strategic episode mix for comprehensive testing
        episodes = [
            # Strategic depth episodes (2500 steps to match RL training)
            {'type': 'scattered', 'steps': 2500, 'seed': 300},
            {'type': 'clustered', 'steps': 2500, 'seed': 301}, 
            {'type': 'scattered', 'steps': 2500, 'seed': 302},
            {'type': 'clustered', 'steps': 2500, 'seed': 303},
            {'type': 'scattered', 'steps': 2500, 'seed': 304},
        ]
        
        for i, episode_config in enumerate(episodes):
            formation_type = episode_config['type']
            max_steps = episode_config['steps']
            seed = episode_config['seed']
            
            # Generate appropriate initial state
            if formation_type == 'scattered':
                initial_state = self.generator.generate_scattered_state(
                    self.boid_count, self.canvas_width, self.canvas_height
                )
            else:  # clustered
                initial_state = self.generator.generate_clustered_state(
                    self.boid_count, self.canvas_width, self.canvas_height
                )
            
            # Set seed for reproducibility
            self.generator.seed = seed
            
            # Run strategic episode
            result = run_strategic_episode(policy, initial_state, max_steps)
            result['formation_type'] = formation_type
            result['episode_config'] = episode_config
            
            all_results.append(result)
            
            # Categorize by formation
            if formation_type == 'scattered':
                scattered_results.append(result)
            else:
                clustered_results.append(result)
        
        # Aggregate statistics
        all_rates = [r['overall_rate'] for r in all_results]
        overall_mean = statistics.mean(all_rates)
        overall_std = statistics.stdev(all_rates) if len(all_rates) > 1 else 0.0
        
        # Phase analysis
        early_rates = [r['early_rate'] for r in all_results]
        mid_rates = [r['mid_rate'] for r in all_results] 
        late_rates = [r['late_rate'] for r in all_results]
        
        early_mean = statistics.mean(early_rates)
        mid_mean = statistics.mean(mid_rates)
        late_mean = statistics.mean(late_rates)
        
        # Formation analysis
        scattered_mean = statistics.mean([r['overall_rate'] for r in scattered_results]) if scattered_results else 0.0
        clustered_mean = statistics.mean([r['overall_rate'] for r in clustered_results]) if clustered_results else 0.0
        
        # Strategic depth metrics
        phase_means = [early_mean, mid_mean, late_mean]
        strategy_consistency = 1.0 - (statistics.stdev(phase_means) if len(phase_means) > 1 else 0.0)
        strategy_consistency = max(0.0, min(1.0, strategy_consistency))  # Clamp to [0,1]
        
        adaptability_score = 1.0 - abs(scattered_mean - clustered_mean)
        adaptability_score = max(0.0, min(1.0, adaptability_score))  # Clamp to [0,1]
        
        # Strategic profile analysis
        if late_mean > early_mean * 1.2:
            strategy_type = "Late-game specialist"
        elif early_mean > late_mean * 1.2:
            strategy_type = "Early-game specialist"
        else:
            strategy_type = "Consistent performer"
        
        formation_diff = abs(scattered_mean - clustered_mean)
        if formation_diff < 0.05:
            formation_style = "Adaptive"
        elif scattered_mean > clustered_mean:
            formation_style = "Herding specialist"
        else:
            formation_style = "Flock-breaker"
        
        eval_time = time.time() - start_time
        
        # Count successful episodes (all are successful in our system)
        successful_episodes = len(all_results)
        total_episodes = len(all_results)
        
        # Create strategic result
        strategic_result = StrategicResult(
            policy_name=policy_name,
            overall_catch_rate=overall_mean,
            overall_std_catch_rate=overall_std,
            evaluation_time_seconds=eval_time,
            total_episodes=total_episodes,
            successful_episodes=successful_episodes,
            early_phase_rate=early_mean,
            mid_phase_rate=mid_mean,
            late_phase_rate=late_mean,
            scattered_rate=scattered_mean,
            clustered_rate=clustered_mean,
            strategy_consistency=strategy_consistency,
            adaptability_score=adaptability_score,
            strategy_type=strategy_type,
            formation_style=formation_style,
            all_catch_rates=all_rates,
            episode_details=all_results
        )
        
        # Print strategic summary
        print(f"   ✅ Strategic evaluation complete:")
        print(f"      Overall: {overall_mean:.3f} ± {overall_std:.3f}")
        print(f"      Strategy: {strategy_type} | Formation: {formation_style}")
        print(f"      Phases: Early {early_mean:.3f} | Mid {mid_mean:.3f} | Late {late_mean:.3f}")
        print(f"      Consistency: {strategy_consistency:.3f} | Adaptability: {adaptability_score:.3f}")
        print(f"      Time: {eval_time:.1f}s")
        
        return strategic_result
    
    def compare_policies(self, policies: List[Tuple[Any, str]]) -> Dict[str, Any]:
        """
        Strategic comparison of multiple policies with detailed analysis
        
        Args:
            policies: List of (policy_object, policy_name) tuples
            
        Returns:
            Dictionary with comprehensive comparison results
        """
        print(f"\n🧠 STRATEGIC POLICY COMPARISON")
        print(f"=" * 70)
        
        results = {}
        start_time = time.time()
        
        # Evaluate each policy strategically
        for policy, name in policies:
            result = self.evaluate_policy(policy, name)
            results[name] = result
        
        total_time = time.time() - start_time
        
        # Strategic comparison table
        print(f"\n📊 STRATEGIC RESULTS:")
        print(f"=" * 95)
        print(f"{'Policy':<15} {'Overall':<10} {'Early':<8} {'Mid':<8} {'Late':<8} {'Scatter':<8} {'Cluster':<8} {'Time':<8}")
        print(f"-" * 95)
        
        # Sort by overall performance
        sorted_results = sorted(results.items(), key=lambda x: x[1].overall_catch_rate, reverse=True)
        
        best_overall = -1
        for name, result in sorted_results:
            overall = result.overall_catch_rate
            early = result.early_phase_rate
            mid = result.mid_phase_rate
            late = result.late_phase_rate
            scattered = result.scattered_rate
            clustered = result.clustered_rate
            eval_time = result.evaluation_time_seconds
            
            marker = " 🏆" if overall > best_overall else ""
            if overall > best_overall:
                best_overall = overall
            
            print(f"{name:<15} {overall:<10.3f} {early:<8.3f} {mid:<8.3f} {late:<8.3f} "
                  f"{scattered:<8.3f} {clustered:<8.3f} {eval_time:<8.1f}s{marker}")
        
        # Strategic insights
        print(f"\n🧠 STRATEGIC INSIGHTS:")
        for i, (policy_name, result) in enumerate(sorted_results[:3], 1):
            print(f"   {i}. {policy_name}: {result.strategy_type} | {result.formation_style}")
            print(f"      Consistency: {result.strategy_consistency:.3f} | Adaptability: {result.adaptability_score:.3f}")
        
        print(f"\n⏱️  Total strategic evaluation: {total_time:.1f}s")
        print(f"=" * 95)
        
        return {
            'results': results,
            'sorted_results': sorted_results,
            'total_time': total_time,
            'best_policy': sorted_results[0][0] if sorted_results else None
        }


# Keep old class names for any existing imports
class ScenarioResult:
    """Legacy placeholder - not used in strategic evaluation"""
    pass