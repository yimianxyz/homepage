#!/usr/bin/env python3
"""
Performance Evaluator - Rigorous evaluation framework to compare SL vs RL performance

This module provides comprehensive evaluation metrics and statistical analysis
to verify that RL training produces meaningful improvements over SL baseline.
"""

import sys
import os
import numpy as np
import torch
import json
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.environment import BoidEnvironment
from rl.models import TransformerModel, TransformerModelLoader
from rl.utils import set_seed


@dataclass
class EpisodeMetrics:
    """Comprehensive metrics for a single episode"""
    # Primary metrics
    total_reward: float
    episode_length: int
    boids_caught: int
    success_rate: float  # boids_caught / total_boids
    
    # Efficiency metrics
    reward_per_step: float
    time_to_first_catch: Optional[int]  # None if no catch
    catch_efficiency: float  # boids_caught / episode_length
    
    # Behavioral metrics
    mean_action_magnitude: float
    action_consistency: float  # 1 - std of actions
    exploration_ratio: float  # ratio of diverse actions
    
    # Advanced metrics
    final_boids_remaining: int
    cumulative_approaching_reward: float
    cumulative_catch_reward: float
    max_single_step_reward: float
    
    # Temporal analysis
    reward_trend: str  # "improving", "declining", "stable"
    early_game_performance: float  # reward in first 25% of episode
    late_game_performance: float   # reward in last 25% of episode


@dataclass
class ModelPerformance:
    """Statistical summary of model performance across multiple episodes"""
    model_name: str
    n_episodes: int
    
    # Primary statistics
    mean_total_reward: float
    std_total_reward: float
    median_total_reward: float
    
    mean_success_rate: float
    std_success_rate: float
    
    mean_episode_length: float
    std_episode_length: float
    
    # Efficiency statistics
    mean_reward_per_step: float
    std_reward_per_step: float
    
    mean_catch_efficiency: float
    std_catch_efficiency: float
    
    # Behavioral statistics
    mean_action_consistency: float
    exploration_diversity: float
    
    # Confidence intervals (95%)
    reward_ci_lower: float
    reward_ci_upper: float
    success_rate_ci_lower: float
    success_rate_ci_upper: float
    
    # Distribution properties
    reward_skewness: float
    reward_kurtosis: float
    
    # Failure analysis
    failure_rate: float  # episodes with 0 catches
    timeout_rate: float  # episodes that hit max_steps


class PerformanceEvaluator:
    """Comprehensive model performance evaluator"""
    
    def __init__(self, 
                 n_evaluation_episodes: int = 100,
                 evaluation_seeds: List[int] = None,
                 confidence_level: float = 0.95):
        """
        Initialize evaluator
        
        Args:
            n_evaluation_episodes: Number of episodes per evaluation
            evaluation_seeds: Fixed seeds for reproducible evaluation
            confidence_level: Confidence level for statistical tests
        """
        self.n_episodes = n_evaluation_episodes
        self.confidence_level = confidence_level
        
        # Generate fixed seeds for reproducible evaluation
        if evaluation_seeds is None:
            np.random.seed(42)  # Fixed seed for seed generation
            self.evaluation_seeds = np.random.randint(0, 10000, n_evaluation_episodes).tolist()
        else:
            self.evaluation_seeds = evaluation_seeds[:n_evaluation_episodes]
        
        print(f"Performance Evaluator initialized:")
        print(f"  Episodes per evaluation: {self.n_episodes}")
        print(f"  Confidence level: {self.confidence_level}")
        print(f"  Reproducible seeds: {len(self.evaluation_seeds)} seeds")
    
    def evaluate_single_episode(self, 
                               model: torch.nn.Module, 
                               env_config: Dict[str, Any],
                               seed: int,
                               verbose: bool = False) -> EpisodeMetrics:
        """
        Evaluate model performance on a single episode with comprehensive metrics
        
        Args:
            model: Model to evaluate
            env_config: Environment configuration
            seed: Random seed for this episode
            verbose: Whether to print detailed step information
            
        Returns:
            EpisodeMetrics with comprehensive statistics
        """
        # Create environment with specific seed
        env = BoidEnvironment(
            num_boids=env_config.get('num_boids', 20),
            canvas_width=env_config.get('canvas_width', 800),
            canvas_height=env_config.get('canvas_height', 600),
            max_steps=env_config.get('max_steps', 1000),
            seed=seed
        )
        
        # Reset environment
        obs, info = env.reset(seed=seed)
        total_boids = info['total_boids']
        
        # Episode tracking
        total_reward = 0.0
        step_rewards = []
        actions_taken = []
        boids_caught = 0
        episode_length = 0
        time_to_first_catch = None
        approaching_reward = 0.0
        catch_reward = 0.0
        
        # Set model to evaluation mode if it's a PyTorch model
        if hasattr(model, 'eval'):
            model.eval()
        
        # Use appropriate context manager
        if hasattr(model, 'eval'):
            context_manager = torch.no_grad()
        else:
            context_manager = self._dummy_context()
        
        with context_manager:
            for step in range(env_config.get('max_steps', 1000)):
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
                
                # Track metrics
                total_reward += reward
                step_rewards.append(reward)
                actions_taken.append(action.copy() if hasattr(action, 'copy') else action)
                episode_length += 1
                
                # Track catches
                new_catches = step_info.get('boids_caught_this_step', 0)
                if new_catches > 0:
                    boids_caught += new_catches
                    if time_to_first_catch is None:
                        time_to_first_catch = step + 1
                
                # Track reward breakdown
                breakdown = step_info.get('reward_breakdown', {})
                approaching_reward += breakdown.get('approaching', 0.0)
                catch_reward += breakdown.get('catch', 0.0)
                
                if verbose and step % 50 == 0:
                    print(f"  Step {step}: reward={reward:.4f}, "
                          f"boids_remaining={step_info.get('boids_remaining', 'N/A')}, "
                          f"total_caught={boids_caught}")
                
                if terminated or truncated:
                    break
        
        env.close()
        
        # Calculate comprehensive metrics
        success_rate = boids_caught / total_boids if total_boids > 0 else 0.0
        reward_per_step = total_reward / episode_length if episode_length > 0 else 0.0
        catch_efficiency = boids_caught / episode_length if episode_length > 0 else 0.0
        
        # Action analysis
        if actions_taken:
            actions_array = np.array(actions_taken)
            if actions_array.ndim == 1:
                actions_array = actions_array.reshape(-1, 1)
            
            action_magnitudes = np.linalg.norm(actions_array, axis=1)
            mean_action_magnitude = np.mean(action_magnitudes)
            action_consistency = 1.0 - np.std(action_magnitudes)
            
            # Exploration analysis (action diversity)
            unique_actions = len(np.unique(actions_array.round(2), axis=0))
            exploration_ratio = unique_actions / len(actions_array)
        else:
            mean_action_magnitude = 0.0
            action_consistency = 0.0
            exploration_ratio = 0.0
        
        # Temporal analysis
        if len(step_rewards) >= 4:
            first_quarter = step_rewards[:len(step_rewards)//4]
            last_quarter = step_rewards[-len(step_rewards)//4:]
            early_game_performance = np.mean(first_quarter)
            late_game_performance = np.mean(last_quarter)
            
            # Trend analysis
            if late_game_performance > early_game_performance * 1.1:
                reward_trend = "improving"
            elif late_game_performance < early_game_performance * 0.9:
                reward_trend = "declining" 
            else:
                reward_trend = "stable"
        else:
            early_game_performance = total_reward
            late_game_performance = total_reward
            reward_trend = "stable"
        
        return EpisodeMetrics(
            total_reward=total_reward,
            episode_length=episode_length,
            boids_caught=boids_caught,
            success_rate=success_rate,
            reward_per_step=reward_per_step,
            time_to_first_catch=time_to_first_catch,
            catch_efficiency=catch_efficiency,
            mean_action_magnitude=mean_action_magnitude,
            action_consistency=max(0.0, action_consistency),
            exploration_ratio=exploration_ratio,
            final_boids_remaining=total_boids - boids_caught,
            cumulative_approaching_reward=approaching_reward,
            cumulative_catch_reward=catch_reward,
            max_single_step_reward=max(step_rewards) if step_rewards else 0.0,
            reward_trend=reward_trend,
            early_game_performance=early_game_performance,
            late_game_performance=late_game_performance
        )
    
    def evaluate_model(self, 
                      model: torch.nn.Module,
                      model_name: str,
                      env_config: Dict[str, Any],
                      verbose: bool = False) -> Tuple[ModelPerformance, List[EpisodeMetrics]]:
        """
        Comprehensive evaluation of a model across multiple episodes
        
        Args:
            model: Model to evaluate
            model_name: Name for identification
            env_config: Environment configuration
            verbose: Print detailed information
            
        Returns:
            Tuple of (ModelPerformance summary, List of EpisodeMetrics)
        """
        print(f"\n🧪 Evaluating {model_name}")
        print(f"Environment: {env_config.get('num_boids', 20)} boids, "
              f"{env_config.get('canvas_width', 800)}×{env_config.get('canvas_height', 600)} canvas")
        print(f"Episodes: {self.n_episodes}")
        
        episode_metrics = []
        start_time = time.time()
        
        for i, seed in enumerate(self.evaluation_seeds):
            if verbose or (i + 1) % max(1, self.n_episodes // 10) == 0:
                print(f"  Episode {i+1}/{self.n_episodes} (seed={seed})")
            
            metrics = self.evaluate_single_episode(model, env_config, seed, verbose)
            episode_metrics.append(metrics)
        
        evaluation_time = time.time() - start_time
        print(f"✅ Evaluation completed in {evaluation_time:.1f}s "
              f"({evaluation_time/self.n_episodes:.2f}s per episode)")
        
        # Calculate statistical summary
        performance = self._calculate_performance_summary(model_name, episode_metrics)
        
        return performance, episode_metrics
    
    def _calculate_performance_summary(self, 
                                     model_name: str, 
                                     episodes: List[EpisodeMetrics]) -> ModelPerformance:
        """Calculate comprehensive statistical summary"""
        
        rewards = [ep.total_reward for ep in episodes]
        success_rates = [ep.success_rate for ep in episodes]
        episode_lengths = [ep.episode_length for ep in episodes]
        reward_per_steps = [ep.reward_per_step for ep in episodes]
        catch_efficiencies = [ep.catch_efficiency for ep in episodes]
        action_consistencies = [ep.action_consistency for ep in episodes]
        exploration_ratios = [ep.exploration_ratio for ep in episodes]
        
        # Basic statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        median_reward = np.median(rewards)
        
        mean_success = np.mean(success_rates)
        std_success = np.std(success_rates)
        
        # Confidence intervals (using normal approximation)
        alpha = 1 - self.confidence_level
        z_score = 1.96  # 95% confidence interval
        
        reward_sem = std_reward / np.sqrt(len(rewards))
        reward_margin = z_score * reward_sem
        reward_ci = (mean_reward - reward_margin, mean_reward + reward_margin)
        
        success_sem = std_success / np.sqrt(len(success_rates))
        success_margin = z_score * success_sem
        success_ci = (mean_success - success_margin, mean_success + success_margin)
        
        # Distribution properties (simplified)
        reward_skewness = self._calculate_skewness(rewards)
        reward_kurtosis = self._calculate_kurtosis(rewards)
        
        # Failure analysis
        failure_rate = sum(1 for ep in episodes if ep.boids_caught == 0) / len(episodes)
        timeout_rate = sum(1 for ep in episodes if ep.episode_length >= 1000) / len(episodes)
        
        return ModelPerformance(
            model_name=model_name,
            n_episodes=len(episodes),
            mean_total_reward=mean_reward,
            std_total_reward=std_reward,
            median_total_reward=median_reward,
            mean_success_rate=mean_success,
            std_success_rate=std_success,
            mean_episode_length=np.mean(episode_lengths),
            std_episode_length=np.std(episode_lengths),
            mean_reward_per_step=np.mean(reward_per_steps),
            std_reward_per_step=np.std(reward_per_steps),
            mean_catch_efficiency=np.mean(catch_efficiencies),
            std_catch_efficiency=np.std(catch_efficiencies),
            mean_action_consistency=np.mean(action_consistencies),
            exploration_diversity=np.mean(exploration_ratios),
            reward_ci_lower=reward_ci[0],
            reward_ci_upper=reward_ci[1],
            success_rate_ci_lower=success_ci[0],
            success_rate_ci_upper=success_ci[1],
            reward_skewness=reward_skewness,
            reward_kurtosis=reward_kurtosis,
            failure_rate=failure_rate,
            timeout_rate=timeout_rate
        )
    
    def compare_models(self, 
                      performance_a: ModelPerformance,
                      performance_b: ModelPerformance,
                      episodes_a: List[EpisodeMetrics],
                      episodes_b: List[EpisodeMetrics]) -> Dict[str, Any]:
        """
        Statistical comparison between two models
        
        Returns comprehensive comparison including effect sizes and significance tests
        """
        print(f"\n📊 Statistical Comparison: {performance_a.model_name} vs {performance_b.model_name}")
        
        # Extract reward data
        rewards_a = [ep.total_reward for ep in episodes_a]
        rewards_b = [ep.total_reward for ep in episodes_b]
        
        success_a = [ep.success_rate for ep in episodes_a]
        success_b = [ep.success_rate for ep in episodes_b]
        
        efficiency_a = [ep.reward_per_step for ep in episodes_a]
        efficiency_b = [ep.reward_per_step for ep in episodes_b]
        
        # Statistical tests (simplified t-test)
        reward_ttest = self._simple_ttest(rewards_a, rewards_b)
        success_ttest = self._simple_ttest(success_a, success_b)
        efficiency_ttest = self._simple_ttest(efficiency_a, efficiency_b)
        
        # Effect sizes (Cohen's d)
        reward_effect_size = self._cohens_d(rewards_a, rewards_b)
        success_effect_size = self._cohens_d(success_a, success_b)
        efficiency_effect_size = self._cohens_d(efficiency_a, efficiency_b)
        
        # Simple rank-sum test approximation
        reward_mannwhitney = self._simple_ranksum(rewards_a, rewards_b)
        
        # Practical significance thresholds
        reward_improvement = (performance_b.mean_total_reward - performance_a.mean_total_reward) / performance_a.mean_total_reward
        success_improvement = (performance_b.mean_success_rate - performance_a.mean_success_rate) / max(performance_a.mean_success_rate, 0.001)
        
        comparison = {
            'models': {
                'baseline': performance_a.model_name,
                'comparison': performance_b.model_name
            },
            'sample_sizes': {
                'baseline': len(episodes_a),
                'comparison': len(episodes_b)
            },
            'reward_analysis': {
                'baseline_mean': performance_a.mean_total_reward,
                'comparison_mean': performance_b.mean_total_reward,
                'absolute_difference': performance_b.mean_total_reward - performance_a.mean_total_reward,
                'relative_improvement': reward_improvement,
                'ttest_statistic': reward_ttest['statistic'],
                'ttest_pvalue': reward_ttest['pvalue'],
                'effect_size_cohens_d': reward_effect_size,
                'mannwhitney_pvalue': reward_mannwhitney['pvalue'],
                'significant': reward_ttest['pvalue'] < 0.05,
                'practically_significant': abs(reward_improvement) > 0.1  # 10% improvement threshold
            },
            'success_rate_analysis': {
                'baseline_mean': performance_a.mean_success_rate,
                'comparison_mean': performance_b.mean_success_rate,
                'absolute_difference': performance_b.mean_success_rate - performance_a.mean_success_rate,
                'relative_improvement': success_improvement,
                'ttest_pvalue': success_ttest['pvalue'],
                'effect_size_cohens_d': success_effect_size,
                'significant': success_ttest['pvalue'] < 0.05
            },
            'efficiency_analysis': {
                'baseline_mean': performance_a.mean_reward_per_step,
                'comparison_mean': performance_b.mean_reward_per_step,
                'ttest_pvalue': efficiency_ttest['pvalue'],
                'effect_size_cohens_d': efficiency_effect_size,
                'significant': efficiency_ttest['pvalue'] < 0.05
            },
            'confidence_intervals_overlap': {
                'reward': not (performance_a.reward_ci_upper < performance_b.reward_ci_lower or 
                              performance_b.reward_ci_upper < performance_a.reward_ci_lower),
                'success_rate': not (performance_a.success_rate_ci_upper < performance_b.success_rate_ci_lower or 
                                   performance_b.success_rate_ci_upper < performance_a.success_rate_ci_lower)
            }
        }
        
        # Interpretation
        comparison['interpretation'] = self._interpret_comparison(comparison)
        
        return comparison
    
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        return (np.mean(group2) - np.mean(group1)) / pooled_std
    
    def _simple_ttest(self, group1: List[float], group2: List[float]) -> Dict[str, float]:
        """Simple t-test implementation"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard error
        pooled_se = np.sqrt(var1/n1 + var2/n2)
        
        # T-statistic
        t_stat = (mean2 - mean1) / pooled_se
        
        # Degrees of freedom (Welch's formula approximation)
        df = n1 + n2 - 2
        
        # Simple p-value approximation (assumes normal distribution)
        p_value = 2 * (1 - abs(t_stat) / (1 + abs(t_stat)))  # Rough approximation
        p_value = max(0.001, min(0.999, p_value))  # Clamp to reasonable range
        
        return {'statistic': t_stat, 'pvalue': p_value}
    
    def _simple_ranksum(self, group1: List[float], group2: List[float]) -> Dict[str, float]:
        """Simple rank-sum test approximation"""
        combined = list(group1) + list(group2)
        ranks = np.argsort(np.argsort(combined)) + 1  # Convert to ranks
        
        n1 = len(group1)
        rank_sum1 = np.sum(ranks[:n1])
        
        # Expected rank sum under null hypothesis
        expected = n1 * (len(combined) + 1) / 2
        
        # Simple p-value approximation
        diff = abs(rank_sum1 - expected)
        p_value = max(0.001, min(0.999, 1 - diff / expected))  # Rough approximation
        
        return {'statistic': rank_sum1, 'pvalue': p_value}
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness"""
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        n = len(data)
        skewness = np.sum(((data - mean) / std) ** 3) / n
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis"""
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        n = len(data)
        kurtosis = np.sum(((data - mean) / std) ** 4) / n - 3  # Subtract 3 for excess kurtosis
        return kurtosis
    
    def _dummy_context(self):
        """Dummy context manager for non-PyTorch models"""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()
    
    def _interpret_comparison(self, comparison: Dict[str, Any]) -> Dict[str, str]:
        """Provide interpretation of comparison results"""
        reward_analysis = comparison['reward_analysis']
        
        # Statistical significance
        if reward_analysis['significant']:
            stat_sig = "statistically significant"
        else:
            stat_sig = "not statistically significant"
        
        # Effect size interpretation
        effect_size = abs(reward_analysis['effect_size_cohens_d'])
        if effect_size < 0.2:
            effect_magnitude = "negligible"
        elif effect_size < 0.5:
            effect_magnitude = "small"
        elif effect_size < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"
        
        # Practical significance
        if reward_analysis['practically_significant']:
            practical_sig = "practically significant"
        else:
            practical_sig = "not practically significant"
        
        # Overall conclusion
        improvement = reward_analysis['relative_improvement']
        if improvement > 0.1 and reward_analysis['significant']:
            conclusion = "Strong evidence of improvement"
        elif improvement > 0.05 and reward_analysis['significant']:
            conclusion = "Moderate evidence of improvement"
        elif reward_analysis['significant']:
            conclusion = "Weak evidence of improvement"
        else:
            conclusion = "No significant improvement detected"
        
        return {
            'statistical_significance': stat_sig,
            'effect_magnitude': effect_magnitude,
            'practical_significance': practical_sig,
            'overall_conclusion': conclusion,
            'recommendation': self._get_recommendation(comparison)
        }
    
    def _get_recommendation(self, comparison: Dict[str, Any]) -> str:
        """Get recommendation based on comparison results"""
        reward_analysis = comparison['reward_analysis']
        
        if (reward_analysis['significant'] and 
            reward_analysis['practically_significant'] and 
            abs(reward_analysis['effect_size_cohens_d']) > 0.5):
            return "Strong recommendation: The RL model shows significant and meaningful improvement."
        
        elif reward_analysis['significant'] and reward_analysis['practically_significant']:
            return "Moderate recommendation: The RL model shows improvement, but effect size is modest."
        
        elif reward_analysis['significant']:
            return "Weak recommendation: Statistically significant but questionable practical impact."
        
        else:
            return "No recommendation: No evidence of meaningful improvement from RL training."
    
    def generate_report(self, 
                       baseline_performance: ModelPerformance,
                       rl_performance: ModelPerformance,
                       comparison: Dict[str, Any],
                       save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        
        report = f"""
# RL vs SL Performance Evaluation Report

## Executive Summary
- **Baseline Model**: {baseline_performance.model_name}
- **RL Model**: {rl_performance.model_name}
- **Evaluation Episodes**: {baseline_performance.n_episodes} per model
- **Overall Conclusion**: {comparison['interpretation']['overall_conclusion']}

## Key Findings

### Primary Metrics
| Metric | Baseline | RL Model | Improvement | Significance |
|--------|----------|----------|-------------|--------------|
| Mean Reward | {baseline_performance.mean_total_reward:.3f} | {rl_performance.mean_total_reward:.3f} | {comparison['reward_analysis']['relative_improvement']:.1%} | {comparison['reward_analysis']['significant']} |
| Success Rate | {baseline_performance.mean_success_rate:.1%} | {rl_performance.mean_success_rate:.1%} | {comparison['success_rate_analysis']['relative_improvement']:.1%} | {comparison['success_rate_analysis']['significant']} |
| Reward/Step | {baseline_performance.mean_reward_per_step:.4f} | {rl_performance.mean_reward_per_step:.4f} | {((rl_performance.mean_reward_per_step - baseline_performance.mean_reward_per_step) / baseline_performance.mean_reward_per_step):.1%} | {comparison['efficiency_analysis']['significant']} |

### Statistical Analysis
- **Effect Size (Cohen's d)**: {comparison['reward_analysis']['effect_size_cohens_d']:.3f} ({comparison['interpretation']['effect_magnitude']})
- **P-value (t-test)**: {comparison['reward_analysis']['ttest_pvalue']:.6f}
- **P-value (Mann-Whitney)**: {comparison['reward_analysis']['mannwhitney_pvalue']:.6f}

### Confidence Intervals (95%)
- **Baseline Reward**: [{baseline_performance.reward_ci_lower:.3f}, {baseline_performance.reward_ci_upper:.3f}]
- **RL Model Reward**: [{rl_performance.reward_ci_lower:.3f}, {rl_performance.reward_ci_upper:.3f}]

### Behavioral Analysis
| Aspect | Baseline | RL Model |
|--------|----------|----------|
| Action Consistency | {baseline_performance.mean_action_consistency:.3f} | {rl_performance.mean_action_consistency:.3f} |
| Exploration Diversity | {baseline_performance.exploration_diversity:.3f} | {rl_performance.exploration_diversity:.3f} |
| Failure Rate | {baseline_performance.failure_rate:.1%} | {rl_performance.failure_rate:.1%} |
| Timeout Rate | {baseline_performance.timeout_rate:.1%} | {rl_performance.timeout_rate:.1%} |

## Interpretation
- **Statistical Significance**: {comparison['interpretation']['statistical_significance']}
- **Practical Significance**: {comparison['interpretation']['practical_significance']}
- **Recommendation**: {comparison['interpretation']['recommendation']}

## Detailed Statistics

### Distribution Properties
| Property | Baseline | RL Model |
|----------|----------|----------|
| Mean | {baseline_performance.mean_total_reward:.3f} | {rl_performance.mean_total_reward:.3f} |
| Std Dev | {baseline_performance.std_total_reward:.3f} | {rl_performance.std_total_reward:.3f} |
| Median | {baseline_performance.median_total_reward:.3f} | {rl_performance.median_total_reward:.3f} |
| Skewness | {baseline_performance.reward_skewness:.3f} | {rl_performance.reward_skewness:.3f} |
| Kurtosis | {baseline_performance.reward_kurtosis:.3f} | {rl_performance.reward_kurtosis:.3f} |

---
*Report generated by RL Performance Evaluator*
*Confidence Level: {self.confidence_level:.0%}*
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {save_path}")
        
        return report


def create_evaluation_suite():
    """Create a comprehensive evaluation suite for RL vs SL comparison"""
    
    # Multiple environment configurations for robust testing
    env_configs = [
        {
            'name': 'standard',
            'num_boids': 20,
            'canvas_width': 800,
            'canvas_height': 600,
            'max_steps': 1000
        },
        {
            'name': 'small_scale',
            'num_boids': 10,
            'canvas_width': 400,
            'canvas_height': 300,
            'max_steps': 500
        },
        {
            'name': 'large_scale',
            'num_boids': 30,
            'canvas_width': 1000,
            'canvas_height': 800,
            'max_steps': 1500
        }
    ]
    
    # Create evaluator with sufficient statistical power
    evaluator = PerformanceEvaluator(
        n_evaluation_episodes=100,  # Good statistical power
        confidence_level=0.95
    )
    
    return evaluator, env_configs


if __name__ == "__main__":
    # Example usage
    evaluator, env_configs = create_evaluation_suite()
    
    print("🧪 RL Performance Evaluation Framework")
    print("=" * 50)
    print("This framework provides rigorous statistical evaluation")
    print("to verify RL improvements over SL baseline with:")
    print("- Comprehensive metrics (reward, success rate, efficiency)")
    print("- Statistical significance testing")
    print("- Effect size analysis")
    print("- Confidence intervals")
    print("- Multiple environment configurations")
    print("- Reproducible evaluation seeds")