#!/usr/bin/env python3
"""
Fast Policy Evaluator - Optimized for speed while maintaining statistical validity

Key optimizations:
- Shorter episodes (200 steps max instead of 1500)
- Simplified scenarios (single test case)
- Direct execution (no multiprocessing overhead)
- Focus on differentiation between policies
"""

import time
import statistics
import sys
import os
from typing import Dict, List, Tuple, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.state_manager import StateManager
from simulation.random_state_generator import generate_random_state


def fast_evaluate_policy(policy, policy_name: str = "Policy", 
                        episodes: int = 15, max_steps: int = 250) -> Dict[str, Any]:
    """
    Fast policy evaluation optimized for speed
    
    Args:
        policy: Policy with get_action method
        policy_name: Name for reporting
        episodes: Number of episodes (15 for good statistics)
        max_steps: Max steps per episode (250 for speed)
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"🎯 Fast evaluating {policy_name}...")
    
    start_time = time.time()
    
    # Fixed test scenario - balanced for differentiation
    canvas_width, canvas_height = 400, 300  # Smaller for speed
    boid_count = 10  # Fewer boids for speed
    
    catch_rates = []
    total_catches_list = []
    episode_lengths = []
    
    for episode in range(episodes):
        # Generate episode with unique seed
        initial_state = generate_random_state(
            boid_count, canvas_width, canvas_height, 
            seed=100 + episode  # Fixed seeds for reproducibility
        )
        initial_boids = len(initial_state['boids_states'])
        
        # Initialize simulation
        state_manager = StateManager()
        state_manager.init(initial_state, policy)
        
        total_catches = 0
        step = 0
        
        # Run episode
        for step in range(max_steps):
            result = state_manager.step()
            
            # Count catches
            if 'caught_boids' in result:
                total_catches += len(result['caught_boids'])
            
            # Early termination if all boids caught
            if len(result['boids_states']) == 0:
                break
        
        # Calculate metrics
        catch_rate = total_catches / initial_boids if initial_boids > 0 else 0.0
        catch_rates.append(catch_rate)
        total_catches_list.append(total_catches)
        episode_lengths.append(step + 1)
    
    # Compute statistics
    mean_catch_rate = statistics.mean(catch_rates)
    std_catch_rate = statistics.stdev(catch_rates) if len(catch_rates) > 1 else 0.0
    mean_catches = statistics.mean(total_catches_list)
    mean_length = statistics.mean(episode_lengths)
    
    eval_time = time.time() - start_time
    
    print(f"   Catch rate: {mean_catch_rate:.3f} ± {std_catch_rate:.3f}")
    print(f"   Avg catches: {mean_catches:.1f}, Avg length: {mean_length:.0f} steps")
    print(f"   Time: {eval_time:.1f}s")
    
    return {
        'policy_name': policy_name,
        'mean_catch_rate': mean_catch_rate,
        'std_catch_rate': std_catch_rate,
        'catch_rates': catch_rates,
        'mean_catches': mean_catches,
        'mean_length': mean_length,
        'eval_time': eval_time,
        'episodes': episodes
    }


def compare_policies_fast(policies: List[Tuple[Any, str]]) -> Dict[str, Any]:
    """
    Fast comparison of multiple policies
    
    Args:
        policies: List of (policy_object, policy_name) tuples
    
    Returns:
        Dictionary with comparison results
    """
    print(f"\n🚀 FAST POLICY COMPARISON")
    print(f"=" * 50)
    print(f"  Test scenario: 400×300 canvas, 10 boids")
    print(f"  Episodes per policy: 15")
    print(f"  Max steps per episode: 250")
    print(f"=" * 50)
    
    results = {}
    start_time = time.time()
    
    # Evaluate each policy
    for policy, name in policies:
        result = fast_evaluate_policy(policy, name)
        results[name] = result
    
    total_time = time.time() - start_time
    
    # Print comparison table
    print(f"\n📊 RESULTS COMPARISON:")
    print(f"=" * 65)
    print(f"{'Policy':<15} {'Catch Rate':<12} {'±Std':<8} {'Avg Catches':<12} {'Time':<8}")
    print("-" * 65)
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_catch_rate'], reverse=True)
    
    best_rate = -1
    for name, result in sorted_results:
        catch_rate = result['mean_catch_rate']
        std_rate = result['std_catch_rate']
        avg_catches = result['mean_catches']
        eval_time = result['eval_time']
        
        marker = " 🏆" if catch_rate > best_rate else ""
        if catch_rate > best_rate:
            best_rate = catch_rate
        
        print(f"{name:<15} {catch_rate:<12.3f} {std_rate:<8.3f} {avg_catches:<12.1f} {eval_time:<8.1f}s{marker}")
    
    # Statistical significance test
    print(f"\n📈 STATISTICAL ANALYSIS:")
    if len(results) >= 2:
        # Compare top 2 policies
        top_names = [name for name, _ in sorted_results[:2]]
        if len(top_names) >= 2:
            policy1_rates = results[top_names[0]]['catch_rates']
            policy2_rates = results[top_names[1]]['catch_rates']
            
            # Simple t-test approximation
            mean1 = statistics.mean(policy1_rates)
            mean2 = statistics.mean(policy2_rates)
            
            # Effect size (Cohen's d)
            pooled_std = ((statistics.stdev(policy1_rates)**2 + statistics.stdev(policy2_rates)**2) / 2)**0.5
            effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            improvement = mean1 - mean2
            improvement_pct = (improvement / max(mean2, 0.001)) * 100
            
            print(f"   {top_names[0]} vs {top_names[1]}:")
            print(f"   Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
            print(f"   Effect size: d={effect_size:.3f}")
            
            if effect_size > 0.8:
                print(f"   ✅ LARGE effect - highly significant difference")
            elif effect_size > 0.5:
                print(f"   ✅ MEDIUM effect - significant difference")
            elif effect_size > 0.2:
                print(f"   ⚠️  SMALL effect - may be significant")
            else:
                print(f"   ❌ NO effect - no meaningful difference")
    
    print(f"\n⏱️  Total evaluation time: {total_time:.1f}s")
    print(f"=" * 65)
    
    return {
        'results': results,
        'sorted_results': sorted_results,
        'total_time': total_time,
        'best_policy': sorted_results[0][0] if sorted_results else None
    }


if __name__ == "__main__":
    # Test the fast evaluator
    print("🧪 Testing Fast Evaluator...")
    
    # Load baseline policies for testing
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
        
        # Run comparison
        comparison = compare_policies_fast(policies)
        
        print(f"\n✅ Fast evaluation complete!")
        print(f"   Best policy: {comparison['best_policy']}")
        print(f"   Total time: {comparison['total_time']:.1f}s")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")