#!/usr/bin/env python3
"""
Checkpoint Evaluator - Evaluate any saved PPO checkpoint

This script can load and evaluate any PPO checkpoint, even if training was interrupted.
Perfect for testing progress after timeouts or crashes.

Usage:
    python evaluate_checkpoint.py checkpoints/ppo_iteration_25.pt
    python evaluate_checkpoint.py checkpoints/best_ppo_model.pt
    python evaluate_checkpoint.py --latest
    python evaluate_checkpoint.py --compare-all
"""

import argparse
import os
import sys
import torch
import glob
import time
from pathlib import Path
import json
from typing import Dict, List, Tuple

# Add current directory to path for imports  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training.ppo_transformer_model import PPOTransformerModel, PPOTransformerPolicy
from policy.transformer.transformer_policy import TransformerPolicy
from policy.human_prior.random_policy import RandomPolicy
from policy.human_prior.closest_pursuit_policy import ClosestPursuitPolicy
from evaluation.policy_evaluator import PolicyEvaluator
from simulation.state_manager import StateManager
from simulation.random_state_generator import generate_random_state


def load_ppo_checkpoint(checkpoint_path: str) -> PPOTransformerPolicy:
    """Load PPO policy from checkpoint"""
    print(f"Loading PPO checkpoint: {checkpoint_path}")
    
    # Load checkpoint (disable weights_only for compatibility)  
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get architecture info
    arch = checkpoint['architecture']
    
    # Create model
    model = PPOTransformerModel(
        d_model=arch['d_model'],
        n_heads=arch['n_heads'], 
        n_layers=arch['n_layers'],
        ffn_hidden=arch['ffn_hidden'],
        max_boids=arch['max_boids']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create policy wrapper
    policy = PPOTransformerPolicy(model)
    policy.eval()
    
    print(f"✅ Loaded PPO checkpoint:")
    print(f"   Iteration: {checkpoint['iteration']}")
    print(f"   Timesteps: {checkpoint['total_timesteps']:,}")
    print(f"   Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
    
    return policy, checkpoint


def quick_evaluate(policy, name: str, num_episodes: int = 10) -> Dict[str, float]:
    """Quick evaluation on standard scenarios"""
    print(f"\n🎯 Evaluating {name}...")
    
    evaluator = PolicyEvaluator()
    
    # Use faster evaluation settings
    evaluator.test_scenarios = [((400, 300), 8)]  # Smaller scenarios
    evaluator.episodes_per_scenario = num_episodes
    evaluator.max_steps_per_episode = 200  # Shorter episodes
    
    start_time = time.time()
    result = evaluator.evaluate_policy(policy, name)
    eval_time = time.time() - start_time
    
    print(f"   Catch rate: {result.overall_catch_rate:.3f} ± {result.overall_std_catch_rate:.3f}")
    print(f"   Episodes: {result.successful_episodes}/{result.total_episodes}")
    print(f"   Time: {eval_time:.1f}s")
    
    return {
        'name': name,
        'catch_rate': result.overall_catch_rate,
        'catch_rate_std': result.overall_std_catch_rate,
        'successful_episodes': result.successful_episodes,
        'total_episodes': result.total_episodes,
        'eval_time': eval_time
    }


def compare_with_baselines(ppo_policy, checkpoint_info: Dict) -> Dict[str, Dict]:
    """Compare PPO policy with baseline policies"""
    print(f"\n📊 POLICY COMPARISON")
    print(f"=" * 50)
    
    results = {}
    
    # Test baseline policies
    random_policy = RandomPolicy()
    pursuit_policy = ClosestPursuitPolicy()
    
    # Load SL baseline if available
    sl_policy = None
    if os.path.exists("checkpoints/best_model.pt"):
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
    
    # Evaluate all policies
    results['random'] = quick_evaluate(random_policy, "Random")
    results['pursuit'] = quick_evaluate(pursuit_policy, "Pursuit")
    
    if sl_policy:
        results['sl_baseline'] = quick_evaluate(sl_policy, "SL Baseline")
    
    results['ppo'] = quick_evaluate(ppo_policy, f"PPO (iter {checkpoint_info['iteration']})")
    
    # Print comparison table
    print(f"\n🏆 RESULTS SUMMARY:")
    print(f"=" * 50)
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['catch_rate'], reverse=True)
    
    for i, (name, result) in enumerate(sorted_results, 1):
        catch_rate = result['catch_rate']
        catch_std = result['catch_rate_std']
        print(f"   {i}. {name:15s}: {catch_rate:.3f} ± {catch_std:.3f}")
    
    # Calculate improvements
    print(f"\n📈 IMPROVEMENTS:")
    ppo_rate = results['ppo']['catch_rate']
    
    if 'sl_baseline' in results:
        sl_rate = results['sl_baseline']['catch_rate'] 
        improvement = ppo_rate - sl_rate
        improvement_pct = (improvement / max(sl_rate, 0.001)) * 100
        print(f"   PPO vs SL: {improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        if improvement > 0:
            print(f"   ✅ PPO IMPROVED over SL baseline!")
        elif improvement == 0:
            print(f"   ➖ PPO EQUALS SL performance")
        else:
            print(f"   ❌ PPO below SL (needs more training)")
    
    pursuit_rate = results['pursuit']['catch_rate']
    ppo_vs_pursuit = ppo_rate - pursuit_rate  
    ppo_vs_pursuit_pct = (ppo_vs_pursuit / max(pursuit_rate, 0.001)) * 100
    print(f"   PPO vs Pursuit: {ppo_vs_pursuit:+.3f} ({ppo_vs_pursuit_pct:+.1f}%)")
    
    return results


def find_latest_checkpoint() -> str:
    """Find the latest PPO checkpoint"""
    checkpoint_pattern = "checkpoints/ppo_iteration_*.pt"
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        raise FileNotFoundError("No PPO checkpoints found")
    
    # Extract iteration numbers and find latest
    latest_checkpoint = None
    latest_iteration = -1
    
    for checkpoint in checkpoints:
        try:
            # Extract iteration from filename like "ppo_iteration_25.pt"
            basename = os.path.basename(checkpoint)
            iteration_str = basename.replace("ppo_iteration_", "").replace(".pt", "")
            iteration = int(iteration_str)
            
            if iteration > latest_iteration:
                latest_iteration = iteration
                latest_checkpoint = checkpoint
        except ValueError:
            continue
    
    if latest_checkpoint is None:
        raise FileNotFoundError("No valid PPO iteration checkpoints found")
    
    print(f"🔍 Found latest checkpoint: {latest_checkpoint} (iteration {latest_iteration})")
    return latest_checkpoint


def compare_all_checkpoints():
    """Compare all available PPO checkpoints"""
    print(f"🔍 COMPARING ALL CHECKPOINTS")
    print(f"=" * 60)
    
    # Find all PPO checkpoints
    checkpoint_pattern = "checkpoints/ppo_iteration_*.pt"
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print("❌ No PPO checkpoints found")
        return
    
    # Sort by iteration number
    checkpoint_data = []
    for checkpoint in checkpoints:
        try:
            basename = os.path.basename(checkpoint)
            iteration_str = basename.replace("ppo_iteration_", "").replace(".pt", "")
            iteration = int(iteration_str)
            checkpoint_data.append((iteration, checkpoint))
        except ValueError:
            continue
    
    checkpoint_data.sort()
    
    print(f"Found {len(checkpoint_data)} checkpoints to compare")
    
    # Quick evaluation of each
    results = []
    for iteration, checkpoint_path in checkpoint_data:
        try:
            policy, checkpoint_info = load_ppo_checkpoint(checkpoint_path)
            result = quick_evaluate(policy, f"Iter {iteration}", num_episodes=5)
            result['iteration'] = iteration
            result['checkpoint_path'] = checkpoint_path
            results.append(result)
        except Exception as e:
            print(f"⚠️  Failed to evaluate {checkpoint_path}: {e}")
    
    # Print comparison table
    print(f"\n📊 CHECKPOINT COMPARISON:")
    print(f"=" * 60)
    print(f"{'Iter':>6} {'Catch Rate':>12} {'±Std':>8} {'Episodes':>10} {'Time':>8}")
    print(f"-" * 60)
    
    best_rate = -1
    best_iteration = -1
    
    for result in results:
        iteration = result['iteration']
        catch_rate = result['catch_rate']
        catch_std = result['catch_rate_std']
        episodes = f"{result['successful_episodes']}/{result['total_episodes']}"
        eval_time = result['eval_time']
        
        marker = ""
        if catch_rate > best_rate:
            best_rate = catch_rate
            best_iteration = iteration
            marker = " 🏆"
        
        print(f"{iteration:>6} {catch_rate:>12.3f} {catch_std:>8.3f} {episodes:>10} {eval_time:>8.1f}s{marker}")
    
    print(f"\n🏆 Best checkpoint: Iteration {best_iteration} ({best_rate:.3f} catch rate)")
    
    # Show learning curve
    if len(results) > 1:
        print(f"\n📈 Learning curve:")
        print(f"   Start: {results[0]['catch_rate']:.3f} (iter {results[0]['iteration']})")
        print(f"   End:   {results[-1]['catch_rate']:.3f} (iter {results[-1]['iteration']})")
        improvement = results[-1]['catch_rate'] - results[0]['catch_rate']
        print(f"   Change: {improvement:+.3f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO checkpoints')
    parser.add_argument('checkpoint', nargs='?', help='Path to checkpoint file')
    parser.add_argument('--latest', action='store_true', help='Evaluate latest checkpoint')
    parser.add_argument('--compare-all', action='store_true', help='Compare all checkpoints')
    parser.add_argument('--quick', action='store_true', help='Quick evaluation (fewer episodes)')
    
    args = parser.parse_args()
    
    try:
        if args.compare_all:
            compare_all_checkpoints()
            return
        
        # Determine checkpoint to evaluate
        if args.latest:
            checkpoint_path = find_latest_checkpoint()
        elif args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            # Default to latest
            try:
                checkpoint_path = find_latest_checkpoint()
                print("No checkpoint specified, using latest...")
            except FileNotFoundError:
                print("❌ No checkpoint specified and no checkpoints found")
                print("Usage: python evaluate_checkpoint.py <checkpoint_path>")
                return 1
        
        # Validate checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return 1
        
        # Load and evaluate
        print(f"🚀 CHECKPOINT EVALUATION")
        print(f"=" * 60)
        
        policy, checkpoint_info = load_ppo_checkpoint(checkpoint_path)
        
        # Compare with baselines
        results = compare_with_baselines(policy, checkpoint_info)
        
        # Save results
        results_path = f"checkpoints/evaluation_results_{checkpoint_info['iteration']}.json"
        with open(results_path, 'w') as f:
            # Convert to serializable format
            serializable_results = {}
            for name, result in results.items():
                serializable_results[name] = {k: float(v) if isinstance(v, (int, float)) else v 
                                            for k, v in result.items()}
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved: {results_path}")
        print(f"✅ Evaluation complete!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())