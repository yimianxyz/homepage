"""
Low-Variance Policy Evaluator - Statistically robust evaluation system

This module provides evaluation with:
- Configurable episode count for time/variance tradeoff
- 95% confidence interval reporting
- Fixed seed protocol for reproducibility
- Balanced formation testing
- Statistical comparison capabilities

Default: 15 episodes for <9% minimum detectable improvement in ~90s
"""

import time
import statistics
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import stats

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.state_manager import StateManager
from simulation.random_state_generator import RandomStateGenerator


@dataclass
class StrategicResult:
    """Results with statistical confidence measures"""
    policy_name: str
    
    # Core metrics with confidence
    overall_catch_rate: float
    overall_std_catch_rate: float
    std_error: float
    confidence_95_lower: float
    confidence_95_upper: float
    confidence_interval_width: float
    
    # Evaluation metadata
    evaluation_time_seconds: float
    total_episodes: int
    successful_episodes: int
    
    # Phase analysis
    early_phase_rate: float      # 0-500 steps performance
    mid_phase_rate: float        # 500-1500 steps performance  
    late_phase_rate: float       # 1500+ steps performance
    
    # Formation analysis
    scattered_rate: float        # Scattered boids (herding challenge)
    clustered_rate: float        # Clustered boids (flock-breaking challenge)
    
    # Strategic metrics
    strategy_consistency: float  # Performance variance across phases (0-1)
    adaptability_score: float    # Performance across formations (0-1)
    
    # Strategic insights
    primary_strategy: str        # Phase-based strategy profile
    formation_preference: str    # Formation-based preference
    strategic_insights: List[str]
    
    # Raw data for further analysis
    all_performances: List[float]
    detailed_results: List[Dict]


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
    Low-Variance Policy Evaluator - Statistically robust evaluation
    
    Key features:
    1. Configurable episode count (default: 15 for optimal tradeoff)
    2. 95% confidence interval reporting
    3. Fixed seed protocol for reproducibility
    4. Balanced formation testing
    5. Statistical comparison capabilities
    
    Time/precision tradeoffs:
    - 5 episodes: ~30s, detects >15% changes (legacy mode)
    - 10 episodes: ~60s, detects >11% changes (quick mode)
    - 15 episodes: ~90s, detects >9% changes (recommended)
    - 30 episodes: ~180s, detects >6% changes (high precision)
    """
    
    def __init__(self, num_episodes: int = 15, base_seed: int = 1000):
        """
        Initialize low-variance policy evaluator
        
        Args:
            num_episodes: Number of episodes to run (default: 15)
            base_seed: Starting seed for reproducibility (default: 1000)
        """
        self.generator = RandomStateGenerator()
        self.num_episodes = num_episodes
        self.base_seed = base_seed
        
        # Evaluation parameters
        self.canvas_width = 400
        self.canvas_height = 300
        self.boid_count = 12
        self.episode_length = 2500
        
        # Calculate expected precision
        population_std_estimate = 0.089  # From empirical analysis
        expected_std_error = population_std_estimate / np.sqrt(self.num_episodes)
        expected_min_detectable = 1.96 * expected_std_error * 2
        
        print(f"Low-Variance PolicyEvaluator initialized:")
        print(f"  Episodes: {self.num_episodes} (balanced formations)")
        print(f"  Expected precision: ±{expected_std_error:.3f} (can detect >{expected_min_detectable*100:.1f}% changes)")
        print(f"  Base seed: {self.base_seed}")
        print(f"  Estimated time: ~{self.num_episodes * 6}s")
    
    def evaluate_policy(self, policy, policy_name: str = "Policy") -> StrategicResult:
        """
        Evaluate policy with statistical confidence measures
        
        Uses fixed seed protocol and balanced formations for reproducibility.
        Reports mean performance with 95% confidence intervals.
        
        Args:
            policy: Policy with get_action method
            policy_name: Name for reporting
            
        Returns:
            StrategicResult with confidence intervals and detailed analysis
        """
        print(f"\n📊 Evaluating {policy_name} ({self.num_episodes} episodes)...")
        
        start_time = time.time()
        
        all_results = []
        scattered_results = []
        clustered_results = []
        performances = []
        
        # Generate balanced episode set
        for i in range(self.num_episodes):
            # Alternate formations for balance
            formation_type = 'scattered' if i % 2 == 0 else 'clustered'
            seed = self.base_seed + i
            
            # Set seed for reproducibility
            self.generator.seed = seed
            
            # Generate appropriate initial state
            if formation_type == 'scattered':
                initial_state = self.generator.generate_scattered_state(
                    self.boid_count, self.canvas_width, self.canvas_height
                )
            else:  # clustered
                initial_state = self.generator.generate_clustered_state(
                    self.boid_count, self.canvas_width, self.canvas_height
                )
            
            # Run strategic episode
            result = run_strategic_episode(policy, initial_state, self.episode_length)
            result['formation_type'] = formation_type
            result['seed'] = seed
            
            all_results.append(result)
            performances.append(result['overall_rate'])
            
            # Categorize by formation
            if formation_type == 'scattered':
                scattered_results.append(result)
            else:
                clustered_results.append(result)
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"   Progress: {i+1}/{self.num_episodes} episodes")
        
        # Calculate statistics with confidence intervals
        overall_mean = np.mean(performances)
        overall_std = np.std(performances, ddof=1)  # Sample standard deviation
        std_error = overall_std / np.sqrt(self.num_episodes)
        
        # 95% confidence interval
        ci_margin = 1.96 * std_error
        ci_lower = overall_mean - ci_margin
        ci_upper = overall_mean + ci_margin
        ci_width = ci_margin * 2
        
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
        strategy_consistency = max(0.0, min(1.0, strategy_consistency))
        
        adaptability_score = 1.0 - abs(scattered_mean - clustered_mean)
        adaptability_score = max(0.0, min(1.0, adaptability_score))
        
        # Determine primary strategy
        if late_mean > early_mean * 1.2 and late_mean > mid_mean:
            primary_strategy = "Late-game specialist"
        elif early_mean > late_mean * 1.2 and early_mean > mid_mean:
            primary_strategy = "Early-game specialist"
        else:
            primary_strategy = "Consistent performer"
        
        # Determine formation preference  
        if scattered_mean > clustered_mean * 1.1:
            formation_preference = "Herding specialist"
        elif clustered_mean > scattered_mean * 1.1:
            formation_preference = "Flock-breaker"
        else:
            formation_preference = "Adaptive"
        
        # Strategic insights
        max_phase = max(early_mean, mid_mean, late_mean)
        weakest_phase = "early" if early_mean == min(phase_means) else "mid" if mid_mean == min(phase_means) else "late"
        
        strategic_insights = [
            f"Primary strategy: {primary_strategy}",
            f"Formation preference: {formation_preference}",
            f"Strongest phase: {'early' if early_mean == max_phase else 'mid' if mid_mean == max_phase else 'late'}",
            f"Improvement opportunity: {weakest_phase} phase"
        ]
        
        eval_time = time.time() - start_time
        
        # Count successful episodes (all are successful in our system)
        successful_episodes = len(all_results)
        total_episodes = len(all_results)
        
        # Create result with confidence intervals
        strategic_result = StrategicResult(
            policy_name=policy_name,
            overall_catch_rate=overall_mean,
            overall_std_catch_rate=overall_std,
            std_error=std_error,
            confidence_95_lower=ci_lower,
            confidence_95_upper=ci_upper,
            confidence_interval_width=ci_width,
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
            primary_strategy=primary_strategy,
            formation_preference=formation_preference,
            strategic_insights=strategic_insights,
            all_performances=performances,
            detailed_results=all_results
        )
        
        # Print summary with confidence intervals
        print(f"\n   ✅ Evaluation complete:")
        print(f"      Performance: {overall_mean:.4f} ± {ci_margin:.4f} (95% CI)")
        print(f"      Confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"      Std error: {std_error:.4f} ({std_error/overall_mean*100:.1f}% relative)")
        print(f"      Strategy: {primary_strategy} | Formation: {formation_preference}")
        print(f"      Phases: Early {early_mean:.3f} | Mid {mid_mean:.3f} | Late {late_mean:.3f}")
        print(f"      Time: {eval_time:.1f}s")
        
        return strategic_result
    
    def compare_policies(self, policies: List[Tuple[Any, str]]) -> Dict[str, Any]:
        """
        Compare policies with statistical significance testing
        
        Args:
            policies: List of (policy_object, policy_name) tuples
            
        Returns:
            Dictionary with comparison results including p-values
        """
        print(f"\n🔬 STATISTICAL POLICY COMPARISON")
        print(f"=" * 70)
        
        results = {}
        start_time = time.time()
        
        # Evaluate each policy
        for policy, name in policies:
            result = self.evaluate_policy(policy, name)
            results[name] = result
        
        total_time = time.time() - start_time
        
        # Results table with confidence intervals
        print(f"\n📊 RESULTS WITH 95% CONFIDENCE INTERVALS:")
        print(f"=" * 100)
        print(f"{'Policy':<20} {'Performance':<20} {'CI Lower':<10} {'CI Upper':<10} {'Phases (E/M/L)':<20}")
        print(f"-" * 100)
        
        # Sort by overall performance
        sorted_results = sorted(results.items(), key=lambda x: x[1].overall_catch_rate, reverse=True)
        
        for name, result in sorted_results:
            perf_str = f"{result.overall_catch_rate:.4f} ± {result.confidence_interval_width/2:.4f}"
            phases_str = f"{result.early_phase_rate:.2f}/{result.mid_phase_rate:.2f}/{result.late_phase_rate:.2f}"
            
            print(f"{name:<20} {perf_str:<20} {result.confidence_95_lower:<10.4f} "
                  f"{result.confidence_95_upper:<10.4f} {phases_str:<20}")
        
        # Statistical comparisons for pairs
        if len(policies) == 2:
            print(f"\n📈 STATISTICAL COMPARISON:")
            name1, name2 = list(results.keys())
            result1, result2 = results[name1], results[name2]
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(
                result2.all_performances,
                result1.all_performances,
                equal_var=False  # Welch's t-test
            )
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(result1.all_performances, ddof=1) + 
                                 np.var(result2.all_performances, ddof=1)) / 2)
            cohens_d = (result2.overall_catch_rate - result1.overall_catch_rate) / pooled_std
            
            # Improvement
            improvement = (result2.overall_catch_rate - result1.overall_catch_rate) / result1.overall_catch_rate * 100
            
            print(f"   Baseline ({name1}): {result1.overall_catch_rate:.4f}")
            print(f"   Compared ({name2}): {result2.overall_catch_rate:.4f}")
            print(f"   Improvement: {improvement:+.1f}%")
            print(f"   p-value: {p_value:.4f} {'✅ (significant at α=0.05)' if p_value < 0.05 else '❌ (not significant)'}")
            print(f"   Effect size (Cohen's d): {cohens_d:.2f} ", end="")
            if abs(cohens_d) < 0.2:
                print("(negligible)")
            elif abs(cohens_d) < 0.5:
                print("(small)")
            elif abs(cohens_d) < 0.8:
                print("(medium)")
            else:
                print("(large)")
        
        print(f"\n⏱️  Total evaluation time: {total_time:.1f}s")
        print(f"=" * 100)
        
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