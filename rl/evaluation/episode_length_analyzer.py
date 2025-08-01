#!/usr/bin/env python3
"""
Episode Length Analyzer - Determine optimal episode length for RL training

This script analyzes episode dynamics with different models to determine
what episode length provides meaningful learning opportunities.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.environment import BoidEnvironment
from rl.models import TransformerModel, TransformerModelLoader
from rl.utils import set_seed


class EpisodeLengthAnalyzer:
    """Analyze optimal episode length for RL training"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_episode_dynamics(self, 
                                model,
                                model_name: str,
                                env_config: Dict[str, Any],
                                max_episode_length: int = 3000,
                                n_episodes: int = 10) -> Dict[str, Any]:
        """
        Analyze episode dynamics to determine optimal length
        
        Args:
            model: Model to test
            model_name: Name for identification
            env_config: Environment configuration
            max_episode_length: Maximum episode length to test
            n_episodes: Number of episodes to analyze
            
        Returns:
            Dictionary with episode analysis results
        """
        print(f"\n🔍 Analyzing episode dynamics for {model_name}")
        print(f"Max episode length: {max_episode_length}")
        print(f"Episodes to analyze: {n_episodes}")
        
        all_episode_data = []
        
        for episode in range(n_episodes):
            print(f"  Episode {episode + 1}/{n_episodes}")
            
            # Create environment
            env = BoidEnvironment(
                num_boids=env_config.get('num_boids', 10),
                canvas_width=env_config.get('canvas_width', 400),
                canvas_height=env_config.get('canvas_height', 300),
                max_steps=max_episode_length,
                seed=42 + episode
            )
            
            # Run episode
            episode_data = self._run_single_episode_analysis(model, env, max_episode_length)
            all_episode_data.append(episode_data)
            
            env.close()
        
        # Analyze results
        analysis = self._analyze_episode_patterns(all_episode_data, model_name)
        
        return analysis
    
    def _run_single_episode_analysis(self, 
                                   model, 
                                   env: BoidEnvironment, 
                                   max_steps: int) -> Dict[str, Any]:
        """Run single episode and collect detailed data"""
        
        obs, info = env.reset()
        total_boids = info['total_boids']
        
        # Episode tracking
        step_data = []
        total_reward = 0.0
        boids_caught = 0
        episode_length = 0
        
        # Set model to eval mode if possible
        if hasattr(model, 'eval'):
            model.eval()
        
        context_manager = torch.no_grad() if hasattr(model, 'eval') else self._dummy_context()
        
        with context_manager:
            for step in range(max_steps):
                # Get model action
                if hasattr(model, 'predict'):
                    # For stable-baselines3 models
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # For raw transformer models
                    structured_input = env.state_manager._convert_state_to_structured_inputs(
                        env.state_manager.get_state()
                    )
                    action = model(structured_input).cpu().numpy()
                
                # Take step
                obs, reward, terminated, truncated, step_info = env.step(action)
                
                # Record step data
                step_data.append({
                    'step': step,
                    'reward': reward,
                    'boids_remaining': step_info.get('boids_remaining', total_boids),
                    'boids_caught_this_step': step_info.get('boids_caught_this_step', 0),
                    'total_reward': total_reward + reward,
                    'action_magnitude': np.linalg.norm(action)
                })
                
                total_reward += reward
                boids_caught += step_info.get('boids_caught_this_step', 0)
                episode_length += 1
                
                if terminated or truncated:
                    break
        
        return {
            'episode_length': episode_length,
            'total_reward': total_reward,
            'boids_caught': boids_caught,
            'success_rate': boids_caught / total_boids,
            'step_data': step_data,
            'terminated_naturally': terminated,
            'hit_max_steps': episode_length >= max_steps
        }
    
    def _dummy_context(self):
        """Dummy context manager"""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()
    
    def _analyze_episode_patterns(self, 
                                all_episodes: List[Dict[str, Any]], 
                                model_name: str) -> Dict[str, Any]:
        """Analyze patterns across all episodes to determine optimal length"""
        
        print(f"\n📊 Analyzing episode patterns for {model_name}")
        
        # Extract key metrics
        episode_lengths = [ep['episode_length'] for ep in all_episodes]
        total_rewards = [ep['total_reward'] for ep in all_episodes]
        success_rates = [ep['success_rate'] for ep in all_episodes]
        boids_caught = [ep['boids_caught'] for ep in all_episodes]
        natural_terminations = [ep['terminated_naturally'] for ep in all_episodes]
        hit_max_steps = [ep['hit_max_steps'] for ep in all_episodes]
        
        # Calculate statistics
        stats = {
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'median_episode_length': np.median(episode_lengths),
            'min_episode_length': np.min(episode_lengths),
            'max_episode_length': np.max(episode_lengths),
            
            'mean_total_reward': np.mean(total_rewards),
            'std_total_reward': np.std(total_rewards),
            
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            
            'mean_boids_caught': np.mean(boids_caught),
            'natural_termination_rate': np.mean(natural_terminations),
            'timeout_rate': np.mean(hit_max_steps),
        }
        
        # Analyze reward dynamics over time
        reward_dynamics = self._analyze_reward_dynamics(all_episodes)
        
        # Determine optimal episode length
        optimal_length = self._determine_optimal_length(stats, reward_dynamics, all_episodes)
        
        # Print summary
        print(f"📈 Episode Statistics:")
        print(f"  Mean length: {stats['mean_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")
        print(f"  Median length: {stats['median_episode_length']:.1f}")
        print(f"  Range: [{stats['min_episode_length']:.0f}, {stats['max_episode_length']:.0f}]")
        print(f"  Natural termination rate: {stats['natural_termination_rate']:.1%}")
        print(f"  Timeout rate: {stats['timeout_rate']:.1%}")
        print(f"  Mean success rate: {stats['mean_success_rate']:.1%}")
        print(f"  Mean reward: {stats['mean_total_reward']:.3f}")
        
        return {
            'model_name': model_name,
            'statistics': stats,
            'reward_dynamics': reward_dynamics,
            'optimal_length': optimal_length,
            'all_episodes': all_episodes
        }
    
    def _analyze_reward_dynamics(self, all_episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how rewards evolve over episode time"""
        
        # Find maximum episode length for alignment
        max_length = max(ep['episode_length'] for ep in all_episodes)
        
        # Aggregate reward curves
        reward_curves = []
        for episode in all_episodes:
            step_data = episode['step_data']
            rewards = [step['reward'] for step in step_data]
            cumulative_rewards = [step['total_reward'] for step in step_data]
            
            # Pad to max length if needed
            while len(rewards) < max_length:
                rewards.append(0.0)
                cumulative_rewards.append(cumulative_rewards[-1] if cumulative_rewards else 0.0)
            
            reward_curves.append({
                'rewards': rewards[:max_length],
                'cumulative': cumulative_rewards[:max_length]
            })
        
        # Calculate mean reward progression
        mean_rewards = []
        mean_cumulative = []
        
        for step in range(max_length):
            step_rewards = [curve['rewards'][step] for curve in reward_curves if step < len(curve['rewards'])]
            step_cumulative = [curve['cumulative'][step] for curve in reward_curves if step < len(curve['cumulative'])]
            
            mean_rewards.append(np.mean(step_rewards) if step_rewards else 0.0)
            mean_cumulative.append(np.mean(step_cumulative) if step_cumulative else 0.0)
        
        # Find when rewards plateau
        plateau_start = self._find_reward_plateau(mean_cumulative)
        
        return {
            'max_length_analyzed': max_length,
            'mean_step_rewards': mean_rewards,
            'mean_cumulative_rewards': mean_cumulative,
            'plateau_start': plateau_start
        }
    
    def _find_reward_plateau(self, cumulative_rewards: List[float], window_size: int = 50) -> int:
        """Find when cumulative rewards start to plateau"""
        
        if len(cumulative_rewards) < window_size * 2:
            return len(cumulative_rewards) // 2
        
        # Calculate reward rate in sliding windows
        reward_rates = []
        for i in range(window_size, len(cumulative_rewards) - window_size):
            early_avg = np.mean(cumulative_rewards[i-window_size:i])
            late_avg = np.mean(cumulative_rewards[i:i+window_size])
            rate = (late_avg - early_avg) / window_size if early_avg > 0 else 0
            reward_rates.append(rate)
        
        # Find where rate drops below threshold
        threshold = max(reward_rates) * 0.1 if reward_rates else 0
        
        for i, rate in enumerate(reward_rates):
            if rate < threshold:
                return i + window_size
        
        return len(cumulative_rewards) // 2
    
    def _determine_optimal_length(self, 
                                stats: Dict[str, Any], 
                                dynamics: Dict[str, Any], 
                                all_episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine optimal episode length based on analysis"""
        
        # Method 1: Based on natural episode completion
        natural_length = stats['mean_episode_length'] * 2 if stats['natural_termination_rate'] > 0.5 else None
        
        # Method 2: Based on reward plateau
        plateau_length = dynamics['plateau_start'] * 1.5 if dynamics['plateau_start'] else None
        
        # Method 3: Based on percentile analysis
        episode_lengths = [ep['episode_length'] for ep in all_episodes]
        percentile_75 = np.percentile(episode_lengths, 75)
        percentile_90 = np.percentile(episode_lengths, 90)
        
        # Choose optimal length
        candidates = [
            natural_length,
            plateau_length,
            percentile_75,
            percentile_90
        ]
        
        # Filter out None values and choose reasonable length
        valid_candidates = [c for c in candidates if c is not None]
        
        if valid_candidates:
            # Choose the median of valid candidates, but ensure it's reasonable
            optimal = int(np.median(valid_candidates))
            optimal = max(1000, min(5000, optimal))  # Clamp between 1k-5k
        else:
            optimal = 2000  # Default fallback
        
        recommendation = {
            'recommended_length': optimal,
            'natural_based': natural_length,
            'plateau_based': plateau_length,
            'percentile_75': percentile_75,
            'percentile_90': percentile_90,
            'reasoning': self._get_length_reasoning(stats, dynamics)
        }
        
        print(f"\n🎯 Episode Length Recommendation:")
        print(f"  Recommended: {optimal} steps")
        print(f"  Natural completion based: {natural_length}")
        print(f"  Reward plateau based: {plateau_length}")
        print(f"  75th percentile: {percentile_75:.0f}")
        print(f"  90th percentile: {percentile_90:.0f}")
        
        return recommendation
    
    def _get_length_reasoning(self, stats: Dict[str, Any], dynamics: Dict[str, Any]) -> str:
        """Generate reasoning for the recommended length"""
        
        reasons = []
        
        if stats['natural_termination_rate'] > 0.7:
            reasons.append("High natural termination rate suggests episodes complete naturally")
        elif stats['timeout_rate'] > 0.8:
            reasons.append("High timeout rate suggests episodes need more time")
        
        if dynamics['plateau_start']:
            reasons.append(f"Rewards plateau around step {dynamics['plateau_start']}")
        
        if stats['mean_success_rate'] < 0.1:
            reasons.append("Low success rate suggests need for longer episodes")
        
        return "; ".join(reasons) if reasons else "Based on statistical analysis of episode patterns"


def test_episode_lengths():
    """Test episode length analysis with different models"""
    
    print("🔍 EPISODE LENGTH ANALYSIS")
    print("=" * 60)
    
    set_seed(42)
    
    # Environment configuration
    env_config = {
        'num_boids': 10,
        'canvas_width': 400,
        'canvas_height': 300
    }
    
    analyzer = EpisodeLengthAnalyzer()
    
    # Test 1: Random baseline
    print("\n🎲 Testing with random policy...")
    
    class RandomModel:
        def predict(self, obs, deterministic=True):
            action = np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
            return action, None
    
    random_model = RandomModel()
    random_analysis = analyzer.analyze_episode_dynamics(
        random_model, "Random_Policy", env_config, max_episode_length=3000, n_episodes=5
    )
    
    # Test 2: Try to load SL model if available
    sl_model_path = "/home/iotcat/homepage2/best_model.pt"
    
    if os.path.exists(sl_model_path):
        print(f"\n📚 Testing with SL model from {sl_model_path}...")
        try:
            from rl.models import TransformerModelLoader
            loader = TransformerModelLoader()
            sl_model = loader.load_model(sl_model_path)
            
            # Wrap for prediction interface
            class SLModelWrapper:
                def __init__(self, model):
                    self.model = model
                    
                def predict(self, obs, deterministic=True):
                    with torch.no_grad():
                        # Convert observation to structured input
                        # This is a simplified conversion - might need adjustment
                        structured_input = {
                            'context': {'canvasWidth': 0.5, 'canvasHeight': 0.4},
                            'predator': {'velX': 0.0, 'velY': 0.0},
                            'boids': []
                        }
                        
                        # Extract boid data from observation (simplified)
                        # Skip first 4 elements (context + predator), then process boids
                        boid_data = obs[4:].reshape(-1, 4)  # [relX, relY, velX, velY]
                        for i in range(min(10, len(boid_data))):  # Max 10 boids
                            if np.any(boid_data[i] != 0):  # Non-zero boid
                                structured_input['boids'].append({
                                    'relX': float(boid_data[i][0]),
                                    'relY': float(boid_data[i][1]),
                                    'velX': float(boid_data[i][2]),
                                    'velY': float(boid_data[i][3])
                                })
                        
                        action = self.model(structured_input).cpu().numpy()
                        return action, None
            
            sl_wrapped = SLModelWrapper(sl_model)
            sl_analysis = analyzer.analyze_episode_dynamics(
                sl_wrapped, "SL_Model", env_config, max_episode_length=3000, n_episodes=5
            )
            
        except Exception as e:
            print(f"⚠️  Could not load SL model: {e}")
            sl_analysis = None
    else:
        print(f"⚠️  SL model not found at {sl_model_path}")
        sl_analysis = None
    
    # Summary and recommendations
    print(f"\n📋 SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)
    
    print(f"\n🎲 Random Policy Analysis:")
    print(f"  Recommended episode length: {random_analysis['optimal_length']['recommended_length']}")
    print(f"  Reasoning: {random_analysis['optimal_length']['reasoning']}")
    
    if sl_analysis:
        print(f"\n📚 SL Model Analysis:")
        print(f"  Recommended episode length: {sl_analysis['optimal_length']['recommended_length']}")
        print(f"  Reasoning: {sl_analysis['optimal_length']['reasoning']}")
        
        # Compare the two
        random_rec = random_analysis['optimal_length']['recommended_length']
        sl_rec = sl_analysis['optimal_length']['recommended_length']
        
        print(f"\n🔄 Comparison:")
        print(f"  Random policy needs: {random_rec} steps")
        print(f"  SL model needs: {sl_rec} steps")
        
        final_recommendation = max(random_rec, sl_rec)
        print(f"  Final recommendation: {final_recommendation} steps")
        print(f"  (Choose higher value to accommodate both)")
        
    else:
        final_recommendation = random_analysis['optimal_length']['recommended_length']
        print(f"\n🎯 Final recommendation: {final_recommendation} steps")
    
    return final_recommendation


if __name__ == "__main__":
    recommended_length = test_episode_lengths()
    print(f"\n✅ Analysis complete. Recommended episode length: {recommended_length} steps")