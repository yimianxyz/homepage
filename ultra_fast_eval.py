#!/usr/bin/env python3
"""
Ultra Fast Evaluation - Minimal evaluation for quick RL vs SL comparison

Optimizations:
- Very short episodes (100 steps)
- Fewer episodes (10) 
- Smaller scenario (8 boids)
- Focus on relative performance, not absolute
"""

import time
import statistics
import sys
import os
import torch
from typing import Dict, List, Tuple, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.state_manager import StateManager
from simulation.random_state_generator import generate_random_state


def ultra_fast_eval(policy, name: str, episodes: int = 10, max_steps: int = 100) -> Dict[str, Any]:
    """Ultra fast evaluation - focus on speed over precision"""
    print(f"⚡ {name}...")
    
    start_time = time.time()
    
    # Very simple scenario
    canvas_width, canvas_height = 300, 200
    boid_count = 8
    
    catch_rates = []
    
    for episode in range(episodes):
        initial_state = generate_random_state(
            boid_count, canvas_width, canvas_height, seed=200 + episode
        )
        initial_boids = len(initial_state['boids_states'])
        
        state_manager = StateManager()
        state_manager.init(initial_state, policy)
        
        total_catches = 0
        
        for step in range(max_steps):
            result = state_manager.step()
            
            if 'caught_boids' in result:
                total_catches += len(result['caught_boids'])
            
            if len(result['boids_states']) == 0:
                break
        
        catch_rate = total_catches / initial_boids if initial_boids > 0 else 0.0
        catch_rates.append(catch_rate)
    
    mean_rate = statistics.mean(catch_rates)
    std_rate = statistics.stdev(catch_rates) if len(catch_rates) > 1 else 0.0
    eval_time = time.time() - start_time
    
    print(f"   {mean_rate:.3f} ± {std_rate:.3f} ({eval_time:.1f}s)")
    
    return {
        'name': name,
        'mean_rate': mean_rate,
        'std_rate': std_rate,
        'catch_rates': catch_rates,
        'time': eval_time
    }


def load_ppo_policy(checkpoint_path: str):
    """Load PPO policy from checkpoint"""
    from rl_training.ppo_transformer_model import PPOTransformerModel, PPOTransformerPolicy
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    arch = checkpoint['architecture']
    
    model = PPOTransformerModel(
        d_model=arch['d_model'],
        n_heads=arch['n_heads'], 
        n_layers=arch['n_layers'],
        ffn_hidden=arch['ffn_hidden'],
        max_boids=arch['max_boids']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    policy = PPOTransformerPolicy(model)
    policy.eval()
    
    return policy


def main():
    print("⚡ ULTRA FAST POLICY COMPARISON")
    print("=" * 45)
    print("Scenario: 300×200, 8 boids, 10 episodes, 100 steps max")
    print("=" * 45)
    
    results = []
    
    # Test baseline policies
    try:
        from policy.human_prior.random_policy import RandomPolicy
        from policy.human_prior.closest_pursuit_policy import ClosestPursuitPolicy
        from policy.transformer.transformer_policy import TransformerPolicy
        
        # Quick policies first
        results.append(ultra_fast_eval(RandomPolicy(), "Random"))
        results.append(ultra_fast_eval(ClosestPursuitPolicy(), "Pursuit"))
        
        # SL baseline
        if os.path.exists("checkpoints/best_model.pt"):
            results.append(ultra_fast_eval(TransformerPolicy("checkpoints/best_model.pt"), "SL"))
        
        # Find and test latest PPO checkpoint
        import glob
        ppo_checkpoints = glob.glob("checkpoints/ppo_iteration_*.pt")
        if ppo_checkpoints:
            # Get latest iteration
            latest_iteration = -1
            latest_checkpoint = None
            
            for checkpoint in ppo_checkpoints:
                try:
                    basename = os.path.basename(checkpoint)
                    iteration_str = basename.replace("ppo_iteration_", "").replace(".pt", "")
                    iteration = int(iteration_str)
                    if iteration > latest_iteration:
                        latest_iteration = iteration
                        latest_checkpoint = checkpoint
                except ValueError:
                    continue
            
            if latest_checkpoint:
                print(f"Loading PPO checkpoint: iteration {latest_iteration}")
                ppo_policy = load_ppo_policy(latest_checkpoint)
                results.append(ultra_fast_eval(ppo_policy, f"PPO_iter{latest_iteration}"))
        
        # Print comparison
        print("\n📊 RESULTS:")
        print("-" * 45)
        
        # Sort by performance
        results.sort(key=lambda x: x['mean_rate'], reverse=True)
        
        for i, result in enumerate(results, 1):
            name = result['name']
            rate = result['mean_rate']
            std = result['std_rate']
            time_taken = result['time']
            marker = " 🏆" if i == 1 else ""
            
            print(f"{i}. {name:<12}: {rate:.3f} ± {std:.3f} ({time_taken:.1f}s){marker}")
        
        # Show improvements
        if len(results) >= 2:
            print("\n📈 KEY COMPARISONS:")
            
            # Find SL and PPO results
            sl_result = next((r for r in results if 'SL' in r['name']), None)
            ppo_result = next((r for r in results if 'PPO' in r['name']), None)
            
            if sl_result and ppo_result:
                improvement = ppo_result['mean_rate'] - sl_result['mean_rate']
                improvement_pct = (improvement / max(sl_result['mean_rate'], 0.001)) * 100
                
                print(f"PPO vs SL: {improvement:+.3f} ({improvement_pct:+.1f}%)")
                
                if improvement > 0.02:  # 2% absolute improvement
                    print("✅ PPO shows clear improvement!")
                elif improvement > 0.01:
                    print("⚠️  PPO shows modest improvement")
                elif improvement > -0.01:
                    print("➖ PPO roughly equals SL")
                else:
                    print("❌ PPO below SL (needs more training)")
        
        total_time = sum(r['time'] for r in results)
        print(f"\n⏱️  Total time: {total_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()