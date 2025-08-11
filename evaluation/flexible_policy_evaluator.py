"""
Flexible Policy Evaluator - Supports custom episode lengths and boid counts

This fixes the critical issue where evaluation was hardcoded to 2500 steps,
causing mismatch with training episode lengths.
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
from evaluation.policy_evaluator import StrategicResult, run_strategic_episode


class FlexiblePolicyEvaluator:
    """
    Enhanced evaluator that supports custom episode lengths and boid counts
    
    Key improvement: Episode length matches training configuration
    """
    
    def __init__(self):
        """Initialize flexible policy evaluator"""
        self.generator = RandomStateGenerator()
        self.state_manager = StateManager()
        
        # Default parameters (can be overridden)
        self.default_canvas_width = 400
        self.default_canvas_height = 300
        self.default_boid_count = 12
        self.default_episode_length = 2500
        
        print(f"Flexible PolicyEvaluator initialized")
        print(f"  Supports custom episode lengths and boid counts")
        print(f"  Default: {self.default_episode_length} steps, {self.default_boid_count} boids")
    
    def evaluate_policy(self, policy, policy_name: str = "Policy", 
                       episode_length: int = None, 
                       boid_count: int = None,
                       num_episodes: int = 5) -> StrategicResult:
        """
        Evaluate policy with custom episode length and boid count
        
        Args:
            policy: Policy with get_action method
            policy_name: Name for reporting
            episode_length: Custom episode length (default: 2500)
            boid_count: Custom boid count (default: 12)
            num_episodes: Number of evaluation episodes (default: 5)
            
        Returns:
            StrategicResult with comprehensive analysis
        """
        # Use custom or default parameters
        episode_length = episode_length or self.default_episode_length
        boid_count = boid_count or self.default_boid_count
        
        print(f"\n🧠 Flexible evaluation: {policy_name}")
        print(f"   Episode length: {episode_length} steps")
        print(f"   Boid count: {boid_count}")
        print(f"   Episodes: {num_episodes}")
        
        start_time = time.time()
        
        all_results = []
        scattered_results = []
        clustered_results = []
        
        # Create episode configurations
        episodes = []
        for i in range(num_episodes):
            formation_type = 'scattered' if i % 2 == 0 else 'clustered'
            episodes.append({
                'type': formation_type,
                'steps': episode_length,  # Use custom episode length
                'seed': 300 + i,
                'boid_count': boid_count  # Use custom boid count
            })
        
        for i, episode_config in enumerate(episodes):
            formation_type = episode_config['type']
            max_steps = episode_config['steps']
            seed = episode_config['seed']
            ep_boid_count = episode_config['boid_count']
            
            # Generate appropriate initial state with custom boid count
            if formation_type == 'scattered':
                initial_state = self.generator.generate_scattered_state(
                    ep_boid_count, self.default_canvas_width, self.default_canvas_height
                )
            else:  # clustered
                initial_state = self.generator.generate_clustered_state(
                    ep_boid_count, self.default_canvas_width, self.default_canvas_height
                )
            
            # Set seed for reproducibility
            self.generator.seed = seed
            
            # Run strategic episode with custom length
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
        
        evaluation_time = time.time() - start_time
        
        # Strategic insights
        max_phase = max(early_mean, mid_mean, late_mean)
        weakest_phase = "early" if early_mean == min(phase_means) else "mid" if mid_mean == min(phase_means) else "late"
        
        strategic_insights = [
            f"Primary strategy: {primary_strategy}",
            f"Formation preference: {formation_preference}",
            f"Strongest phase: {'early' if early_mean == max_phase else 'mid' if mid_mean == max_phase else 'late'}",
            f"Improvement opportunity: {weakest_phase} phase"
        ]
        
        # Print summary
        print(f"   ✅ Strategic evaluation complete:")
        print(f"      Overall: {overall_mean:.3f} ± {overall_std:.3f}")
        print(f"      Strategy: {primary_strategy} | Formation: {formation_preference}")
        print(f"      Phases: Early {early_mean:.3f} | Mid {mid_mean:.3f} | Late {late_mean:.3f}")
        print(f"      Consistency: {strategy_consistency:.3f} | Adaptability: {adaptability_score:.3f}")
        print(f"      Episode length: {episode_length} steps | Boid count: {boid_count}")
        print(f"      Time: {evaluation_time:.1f}s")
        
        return StrategicResult(
            policy_name=policy_name,
            overall_catch_rate=overall_mean,
            overall_std_catch_rate=overall_std,
            evaluation_time_seconds=evaluation_time,
            total_episodes=num_episodes,
            successful_episodes=sum(1 for r in all_results if r['overall_rate'] > 0.5),
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
            detailed_results=all_results
        )