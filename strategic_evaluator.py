#!/usr/bin/env python3
"""
Strategic Policy Evaluator - Capture emergent flock strategies and long-term performance

This evaluator tests multiple temporal phases and formation types to capture
the full strategic depth of policies, especially transformer-based ones that
may develop sophisticated multi-step strategies.

Key innovations:
1. Multi-phase evaluation (early/mid/late game performance)
2. Formation-diverse scenarios (scattered vs clustered boids)
3. Adaptive termination (reward efficiency)
4. Strategic depth metrics (not just total catches)
"""

import time
import statistics
import sys
import os
import torch
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.state_manager import StateManager
from simulation.random_state_generator import RandomStateGenerator


@dataclass
class StrategicResult:
    """Results for strategic evaluation with temporal and formation breakdown"""
    policy_name: str
    
    # Overall metrics
    overall_catch_rate: float
    overall_std: float
    total_time: float
    
    # Phase-specific performance (early/mid/late game)
    early_phase_rate: float    # 0-100 steps
    mid_phase_rate: float      # 100-300 steps  
    late_phase_rate: float     # 300+ steps
    
    # Formation-specific performance
    scattered_rate: float      # Scattered boids (herding challenge)
    clustered_rate: float      # Clustered boids (flock-breaking challenge)
    
    # Strategic depth metrics
    efficiency_curve: List[float]    # Catch rate over time
    strategy_consistency: float      # Performance variance across phases
    adaptability_score: float       # Performance across formations
    
    # Raw data
    all_catch_rates: List[float]
    episode_details: List[Dict]


def run_strategic_episode(policy, initial_state: Dict, max_steps: int = 400,
                         efficiency_target: float = 0.8) -> Dict[str, Any]:
    """
    Run a single strategic episode with phase tracking
    
    Args:
        policy: Policy to evaluate
        initial_state: Initial simulation state  
        max_steps: Maximum steps (400 for strategic depth)
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
    catch_timeline = []  # For efficiency curve
    
    # Run episode with phase tracking
    for step in range(max_steps):
        result = state_manager.step()
        
        # Count catches this step
        step_catches = 0
        if 'caught_boids' in result:
            step_catches = len(result['caught_boids'])
            total_catches += step_catches
        
        # Track by phase
        if step < 100:
            phase_catches['early'] += step_catches
        elif step < 300:
            phase_catches['mid'] += step_catches
        else:
            phase_catches['late'] += step_catches
        
        # Record catch rate at this point (for efficiency curve)
        current_rate = total_catches / initial_boids if initial_boids > 0 else 0.0
        catch_timeline.append(current_rate)
        
        # Early termination if highly efficient
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
        'efficiency_curve': catch_timeline,
        'terminated_early': step < max_steps - 1
    }


def strategic_evaluate_policy(policy, policy_name: str = "Policy") -> StrategicResult:
    """
    Strategic evaluation with multi-phase and formation testing
    
    Evaluation design:
    - 3 quick episodes (150 steps) - screening
    - 4 strategic episodes (400 steps) - depth testing  
    - Mix of scattered and clustered formations
    - Total: ~2000 steps = 40-60s evaluation time
    """
    print(f"🧠 Strategic evaluation: {policy_name}")
    print(f"   Testing emergent flock strategies...")
    
    start_time = time.time()
    generator = RandomStateGenerator()
    
    all_results = []
    scattered_results = []
    clustered_results = []
    
    # Canvas optimized for strategic behavior
    canvas_width, canvas_height = 400, 300
    boid_count = 12  # Sweet spot for flock dynamics
    
    print(f"   Scenario: {canvas_width}×{canvas_height}, {boid_count} boids")
    
    # Episode mix for strategic depth
    episodes = [
        # Quick screening episodes (150 steps)
        {'type': 'scattered', 'steps': 150, 'seed': 300},
        {'type': 'clustered', 'steps': 150, 'seed': 301}, 
        {'type': 'scattered', 'steps': 150, 'seed': 302},
        
        # Strategic depth episodes (400 steps)
        {'type': 'clustered', 'steps': 400, 'seed': 303},
        {'type': 'scattered', 'steps': 400, 'seed': 304},
        {'type': 'clustered', 'steps': 400, 'seed': 305},
        {'type': 'scattered', 'steps': 400, 'seed': 306},
    ]
    
    print(f"   Episodes: {len(episodes)} (3 quick + 4 strategic)")
    
    for i, episode_config in enumerate(episodes):
        formation_type = episode_config['type']
        max_steps = episode_config['steps']
        seed = episode_config['seed']
        
        # Generate appropriate initial state
        if formation_type == 'scattered':
            initial_state = generator.generate_scattered_state(boid_count, canvas_width, canvas_height)
        else:  # clustered
            initial_state = generator.generate_clustered_state(boid_count, canvas_width, canvas_height)
        
        # Set seed for reproducibility
        generator.seed = seed
        
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
        
        # Progress indicator
        print(f"     Episode {i+1}/{len(episodes)}: {formation_type} -> {result['overall_rate']:.3f} "
              f"({result['episode_length']} steps)")
    
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
    # Strategy consistency: lower variance = more consistent
    phase_means = [early_mean, mid_mean, late_mean]
    strategy_consistency = 1.0 - (statistics.stdev(phase_means) if len(phase_means) > 1 else 0.0)
    
    # Adaptability: smaller difference between formations = more adaptable
    adaptability_score = 1.0 - abs(scattered_mean - clustered_mean)
    
    # Efficiency curve (average across episodes)
    max_curve_length = max(len(r['efficiency_curve']) for r in all_results)
    avg_efficiency_curve = []
    for step in range(max_curve_length):
        step_values = []
        for result in all_results:
            if step < len(result['efficiency_curve']):
                step_values.append(result['efficiency_curve'][step])
        if step_values:
            avg_efficiency_curve.append(statistics.mean(step_values))
    
    eval_time = time.time() - start_time
    
    # Create strategic result
    strategic_result = StrategicResult(
        policy_name=policy_name,
        overall_catch_rate=overall_mean,
        overall_std=overall_std,
        total_time=eval_time,
        early_phase_rate=early_mean,
        mid_phase_rate=mid_mean,
        late_phase_rate=late_mean,
        scattered_rate=scattered_mean,
        clustered_rate=clustered_mean,
        efficiency_curve=avg_efficiency_curve,
        strategy_consistency=strategy_consistency,
        adaptability_score=adaptability_score,
        all_catch_rates=all_rates,
        episode_details=all_results
    )
    
    # Print strategic summary
    print(f"   ✅ Strategic evaluation complete:")
    print(f"      Overall: {overall_mean:.3f} ± {overall_std:.3f}")
    print(f"      Phases: Early {early_mean:.3f} | Mid {mid_mean:.3f} | Late {late_mean:.3f}")
    print(f"      Formations: Scattered {scattered_mean:.3f} | Clustered {clustered_mean:.3f}")
    print(f"      Strategy consistency: {strategy_consistency:.3f}")
    print(f"      Adaptability: {adaptability_score:.3f}")
    print(f"      Time: {eval_time:.1f}s")
    
    return strategic_result


def compare_strategic_policies(policies: List[Tuple[Any, str]]) -> Dict[str, Any]:
    """
    Strategic comparison of multiple policies with detailed analysis
    """
    print(f"\n🧠 STRATEGIC POLICY COMPARISON")
    print(f"=" * 70)
    print(f"  Focus: Emergent flock strategies and long-term performance")
    print(f"  Formations: Scattered (herding test) vs Clustered (flock-breaking)")
    print(f"  Phases: Early/Mid/Late game performance analysis")
    print(f"=" * 70)
    
    results = {}
    start_time = time.time()
    
    # Evaluate each policy strategically
    for policy, name in policies:
        result = strategic_evaluate_policy(policy, name)
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
        eval_time = result.total_time
        
        marker = " 🏆" if overall > best_overall else ""
        if overall > best_overall:
            best_overall = overall
        
        print(f"{name:<15} {overall:<10.3f} {early:<8.3f} {mid:<8.3f} {late:<8.3f} "
              f"{scattered:<8.3f} {clustered:<8.3f} {eval_time:<8.1f}s{marker}")
    
    # Strategic insights
    print(f"\n🧠 STRATEGIC INSIGHTS:")
    
    if len(results) >= 2:
        top_policies = [name for name, _ in sorted_results[:2]]
        
        for i, policy_name in enumerate(top_policies[:2]):
            result = results[policy_name]
            
            # Analyze strategic patterns
            phases = [result.early_phase_rate, result.mid_phase_rate, result.late_phase_rate]
            
            # Strategic profile
            if result.late_phase_rate > result.early_phase_rate * 1.2:
                strategy_type = "🐢 Late-game specialist (develops strategy over time)"
            elif result.early_phase_rate > result.late_phase_rate * 1.2:
                strategy_type = "🐰 Early-game specialist (quick initial gains)"
            else:
                strategy_type = "📊 Consistent performer (steady throughout)"
            
            # Formation specialization
            formation_diff = abs(result.scattered_rate - result.clustered_rate)
            if formation_diff < 0.05:
                formation_style = "🎯 Adaptive (handles all formations)"
            elif result.scattered_rate > result.clustered_rate:
                formation_style = "🧲 Herding specialist (excels with scattered boids)"
            else:
                formation_style = "💥 Flock-breaker (excels with clustered boids)"
            
            print(f"   {i+1}. {policy_name}:")
            print(f"      {strategy_type}")
            print(f"      {formation_style}")
            print(f"      Consistency: {result.strategy_consistency:.3f} | Adaptability: {result.adaptability_score:.3f}")
    
    # Performance improvements analysis
    if len(results) >= 2:
        print(f"\n📈 STRATEGIC IMPROVEMENTS:")
        top_name = sorted_results[0][0]
        second_name = sorted_results[1][0]
        
        top_result = results[top_name]
        second_result = results[second_name]
        
        overall_improvement = top_result.overall_catch_rate - second_result.overall_catch_rate
        improvement_pct = (overall_improvement / max(second_result.overall_catch_rate, 0.001)) * 100
        
        print(f"   {top_name} vs {second_name}:")
        print(f"   Overall improvement: {overall_improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        # Phase-specific improvements
        early_imp = top_result.early_phase_rate - second_result.early_phase_rate
        mid_imp = top_result.mid_phase_rate - second_result.mid_phase_rate
        late_imp = top_result.late_phase_rate - second_result.late_phase_rate
        
        print(f"   Phase improvements: Early {early_imp:+.3f} | Mid {mid_imp:+.3f} | Late {late_imp:+.3f}")
        
        # Strategic significance
        if overall_improvement > 0.05:
            print(f"   ✅ SIGNIFICANT strategic improvement!")
        elif overall_improvement > 0.02:
            print(f"   ⚠️  MODERATE strategic improvement")
        else:
            print(f"   ➖ MINIMAL strategic difference")
    
    print(f"\n⏱️  Total strategic evaluation: {total_time:.1f}s")
    print(f"=" * 95)
    
    return {
        'results': results,
        'sorted_results': sorted_results,
        'total_time': total_time,
        'best_policy': sorted_results[0][0] if sorted_results else None
    }


if __name__ == "__main__":
    print("🧠 Testing Strategic Evaluator...")
    
    try:
        from policy.human_prior.random_policy import RandomPolicy
        from policy.human_prior.closest_pursuit_policy import ClosestPursuitPolicy
        from policy.transformer.transformer_policy import TransformerPolicy
        
        policies = [
            (RandomPolicy(), "Random"),
            (ClosestPursuitPolicy(), "Pursuit"),
        ]
        
        # Add SL baseline if available
        if os.path.exists("checkpoints/best_model.pt"):
            policies.append((TransformerPolicy("checkpoints/best_model.pt"), "SL_Baseline"))
        
        # Run strategic comparison
        comparison = compare_strategic_policies(policies)
        
        print(f"\n✅ Strategic evaluation complete!")
        print(f"   Best strategic policy: {comparison['best_policy']}")
        print(f"   Total time: {comparison['total_time']:.1f}s")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()