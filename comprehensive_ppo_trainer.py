#!/usr/bin/env python3
"""
Comprehensive PPO Training System with Plateau Detection

Features:
- Trains until performance plateaus
- Saves checkpoints at regular intervals
- Tracks detailed metrics (rewards, losses, value losses)
- Periodic evaluation with low-variance evaluator
- Automatic plateau detection
- Comprehensive training report generation
"""

import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppo_with_value_pretraining import PPOWithValuePretraining
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state
from policy.transformer.transformer_policy import TransformerPolicy


@dataclass
class TrainingMetrics:
    """Comprehensive metrics for each training iteration"""
    iteration: int
    timestamp: float
    
    # Training metrics
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    
    # Episode metrics
    episode_reward: float
    episode_length: int
    episode_catch_rate: float
    
    # Value function metrics
    value_mean: float
    value_std: float
    advantage_mean: float
    advantage_std: float
    
    # Learning metrics
    gradient_norm: float
    learning_rate: float


@dataclass
class EvaluationMetrics:
    """Evaluation results with confidence intervals"""
    iteration: int
    mean_performance: float
    std_error: float
    confidence_lower: float
    confidence_upper: float
    improvement_vs_baseline: float
    p_value: Optional[float] = None
    is_best: bool = False


class ComprehensivePPOTrainer:
    """
    Production-grade PPO trainer with comprehensive monitoring
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints/ppo_comprehensive",
                 eval_frequency: int = 10,
                 checkpoint_frequency: int = 5,
                 plateau_patience: int = 20,
                 plateau_threshold: float = 0.001):
        """
        Initialize comprehensive PPO trainer
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            eval_frequency: Evaluate every N iterations
            checkpoint_frequency: Save checkpoint every N iterations
            plateau_patience: Iterations without improvement before stopping
            plateau_threshold: Minimum improvement to not count as plateau
        """
        self.checkpoint_dir = checkpoint_dir
        self.eval_frequency = eval_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize trainer with optimal settings
        self.trainer = PPOWithValuePretraining(
            sl_checkpoint_path="checkpoints/best_model.pt",
            learning_rate=3e-5,
            clip_epsilon=0.1,
            ppo_epochs=2,
            rollout_steps=512,
            max_episode_steps=5000,  # Optimal episode length
            gamma=0.99,
            gae_lambda=0.95,
            device='cpu',
            value_pretrain_iterations=20,  # Optimal value pre-training
            value_pretrain_lr=3e-4,
            value_pretrain_epochs=4
        )
        
        # Initialize evaluators
        self.quick_evaluator = PolicyEvaluator(num_episodes=5, base_seed=5000)  # For trend
        self.precise_evaluator = PolicyEvaluator(num_episodes=15, base_seed=6000)  # For validation
        
        # Metrics storage
        self.training_metrics: List[TrainingMetrics] = []
        self.evaluation_metrics: List[EvaluationMetrics] = []
        self.value_pretrain_losses: List[float] = []
        
        # Plateau detection
        self.best_performance = -float('inf')
        self.iterations_without_improvement = 0
        self.performance_history = deque(maxlen=plateau_patience)
        
        # Training state
        self.start_time = time.time()
        self.iteration = 0
        self.is_training = True
        
        print(f"📊 Comprehensive PPO Trainer initialized")
        print(f"   Checkpoint dir: {checkpoint_dir}")
        print(f"   Eval frequency: every {eval_frequency} iterations")
        print(f"   Plateau detection: {plateau_patience} iterations, {plateau_threshold} threshold")
    
    def establish_baseline(self) -> EvaluationMetrics:
        """Establish SL baseline with confidence intervals"""
        print("\n🎯 Establishing SL baseline with low-variance evaluation...")
        
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        result = self.precise_evaluator.evaluate_policy(sl_policy, "SL_Baseline")
        
        baseline = EvaluationMetrics(
            iteration=-1,
            mean_performance=result.overall_catch_rate,
            std_error=result.std_error,
            confidence_lower=result.confidence_95_lower,
            confidence_upper=result.confidence_95_upper,
            improvement_vs_baseline=0.0
        )
        
        self.baseline = baseline
        self.evaluation_metrics.append(baseline)
        
        print(f"\n📊 Baseline established: {baseline.mean_performance:.4f} ± {baseline.std_error*1.96:.4f}")
        print(f"   95% CI: [{baseline.confidence_lower:.4f}, {baseline.confidence_upper:.4f}]")
        
        return baseline
    
    def pretrain_value_function(self):
        """Pre-train value function with detailed tracking"""
        print("\n🎓 Pre-training value function...")
        
        self.value_pretrain_losses = self.trainer.pretrain_value_function()
        
        print(f"   Initial loss: {self.value_pretrain_losses[0]:.4f}")
        print(f"   Final loss: {self.value_pretrain_losses[-1]:.4f}")
        print(f"   Reduction: {(1 - self.value_pretrain_losses[-1]/self.value_pretrain_losses[0])*100:.1f}%")
    
    def train_iteration(self) -> TrainingMetrics:
        """Run one training iteration with comprehensive metrics"""
        # Generate initial state
        initial_state = generate_random_state(12, 400, 300)
        
        # Get current learning rate
        current_lr = self.trainer.optimizer.param_groups[0]['lr']
        
        # Train iteration
        start_time = time.time()
        metrics = self.trainer.train_iteration(initial_state)
        
        # Extract detailed metrics
        training_metrics = TrainingMetrics(
            iteration=self.iteration,
            timestamp=time.time() - self.start_time,
            policy_loss=metrics['policy_loss'],
            value_loss=metrics['value_loss'],
            entropy=metrics.get('entropy', 0.0),
            approx_kl=metrics.get('approx_kl', 0.0),
            clip_fraction=metrics.get('clip_fraction', 0.0),
            episode_reward=metrics.get('episode_reward', 0.0),
            episode_length=metrics.get('episode_length', 0),
            episode_catch_rate=metrics.get('episode_catch_rate', 0.0),
            value_mean=metrics.get('value_mean', 0.0),
            value_std=metrics.get('value_std', 0.0),
            advantage_mean=metrics.get('advantage_mean', 0.0),
            advantage_std=metrics.get('advantage_std', 0.0),
            gradient_norm=metrics.get('gradient_norm', 0.0),
            learning_rate=current_lr
        )
        
        self.training_metrics.append(training_metrics)
        return training_metrics
    
    def evaluate_performance(self, use_precise: bool = False) -> EvaluationMetrics:
        """Evaluate current performance with confidence intervals"""
        evaluator = self.precise_evaluator if use_precise else self.quick_evaluator
        
        result = evaluator.evaluate_policy(self.trainer.policy, f"PPO_Iter{self.iteration}")
        
        # Calculate improvement vs baseline
        improvement = (result.overall_catch_rate - self.baseline.mean_performance) / self.baseline.mean_performance * 100
        
        # Check if confidence intervals don't overlap (significant improvement)
        is_significant = result.confidence_95_lower > self.baseline.confidence_95_upper
        
        eval_metrics = EvaluationMetrics(
            iteration=self.iteration,
            mean_performance=result.overall_catch_rate,
            std_error=result.std_error,
            confidence_lower=result.confidence_95_lower,
            confidence_upper=result.confidence_95_upper,
            improvement_vs_baseline=improvement,
            is_best=result.overall_catch_rate > self.best_performance
        )
        
        # Update best performance
        if eval_metrics.is_best:
            self.best_performance = result.overall_catch_rate
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        
        self.evaluation_metrics.append(eval_metrics)
        self.performance_history.append(result.overall_catch_rate)
        
        return eval_metrics
    
    def check_plateau(self) -> bool:
        """Check if training has plateaued"""
        if len(self.performance_history) < self.plateau_patience:
            return False
        
        # Check if performance improvement is below threshold
        recent_performances = list(self.performance_history)
        performance_std = np.std(recent_performances)
        performance_range = max(recent_performances) - min(recent_performances)
        
        # Plateau if:
        # 1. No improvement for patience iterations
        # 2. Performance variance is very low
        # 3. Performance range is below threshold
        has_plateaued = (
            self.iterations_without_improvement >= self.plateau_patience or
            (performance_std < self.plateau_threshold and 
             performance_range < self.plateau_threshold * 2)
        )
        
        return has_plateaued
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.trainer.policy.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'performance': self.evaluation_metrics[-1].mean_performance if self.evaluation_metrics else 0,
            'training_metrics': [asdict(m) for m in self.training_metrics[-100:]],  # Last 100
            'config': {
                'learning_rate': self.trainer.learning_rate,
                'episode_length': self.trainer.max_episode_steps,
                'value_pretrain_iterations': 20
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_iter{self.iteration}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best checkpoint: {self.evaluation_metrics[-1].mean_performance:.4f}")
    
    def train_until_plateau(self, max_iterations: int = 200):
        """Train until performance plateaus"""
        print("\n🚀 Starting PPO training until plateau...")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Episode length: {self.trainer.max_episode_steps}")
        print(f"   Learning rate: {self.trainer.learning_rate}")
        
        # Establish baseline
        self.establish_baseline()
        
        # Pre-train value function
        self.pretrain_value_function()
        
        # Initial evaluation
        print("\n📊 Initial evaluation...")
        eval_metrics = self.evaluate_performance(use_precise=True)
        print(f"   Initial: {eval_metrics.mean_performance:.4f} ({eval_metrics.improvement_vs_baseline:+.1f}%)")
        
        # Training loop
        print("\n🏃 Training progress:")
        while self.iteration < max_iterations and self.is_training:
            self.iteration += 1
            
            # Train iteration
            train_metrics = self.train_iteration()
            
            # Progress logging
            if self.iteration % 5 == 0:
                print(f"   Iter {self.iteration:3d}: "
                      f"Policy loss: {train_metrics.policy_loss:6.3f}, "
                      f"Value loss: {train_metrics.value_loss:6.3f}, "
                      f"Catch rate: {train_metrics.episode_catch_rate:.3f}")
            
            # Evaluation
            if self.iteration % self.eval_frequency == 0:
                print(f"\n   📊 Evaluation at iteration {self.iteration}:")
                eval_metrics = self.evaluate_performance(use_precise=(self.iteration % 20 == 0))
                
                print(f"      Performance: {eval_metrics.mean_performance:.4f} "
                      f"[{eval_metrics.confidence_lower:.4f}, {eval_metrics.confidence_upper:.4f}]")
                print(f"      Improvement: {eval_metrics.improvement_vs_baseline:+.1f}%"
                      f"{' 🌟 NEW BEST!' if eval_metrics.is_best else ''}")
                
                # Check plateau
                if self.check_plateau():
                    print(f"\n   🏁 Performance plateaued after {self.iteration} iterations!")
                    self.is_training = False
            
            # Save checkpoint
            if self.iteration % self.checkpoint_frequency == 0 or not self.is_training:
                self.save_checkpoint(is_best=eval_metrics.is_best if 'eval_metrics' in locals() else False)
        
        # Final evaluation with high precision
        print("\n📊 Final evaluation with high precision...")
        self.precise_evaluator = PolicyEvaluator(num_episodes=30, base_seed=7000)
        final_eval = self.evaluate_performance(use_precise=True)
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   Performance: {final_eval.mean_performance:.4f} ± {final_eval.std_error*1.96:.4f}")
        print(f"   95% CI: [{final_eval.confidence_lower:.4f}, {final_eval.confidence_upper:.4f}]")
        print(f"   Improvement: {final_eval.improvement_vs_baseline:+.1f}%")
        print(f"   Total iterations: {self.iteration}")
        print(f"   Training time: {(time.time() - self.start_time)/60:.1f} minutes")
    
    def generate_training_report(self):
        """Generate comprehensive training report with visualizations"""
        print("\n📝 Generating comprehensive training report...")
        
        # Prepare data
        iterations = [m.iteration for m in self.training_metrics]
        policy_losses = [m.policy_loss for m in self.training_metrics]
        value_losses = [m.value_loss for m in self.training_metrics]
        catch_rates = [m.episode_catch_rate for m in self.training_metrics]
        
        eval_iterations = [m.iteration for m in self.evaluation_metrics if m.iteration >= 0]
        eval_performances = [m.mean_performance for m in self.evaluation_metrics if m.iteration >= 0]
        eval_lower = [m.confidence_lower for m in self.evaluation_metrics if m.iteration >= 0]
        eval_upper = [m.confidence_upper for m in self.evaluation_metrics if m.iteration >= 0]
        
        # Create visualization
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Training losses
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(iterations, policy_losses, 'b-', alpha=0.7, label='Policy Loss')
        ax1.plot(iterations, value_losses, 'r-', alpha=0.7, label='Value Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Episode performance
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(iterations, catch_rates, 'g-', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Episode Catch Rate')
        ax2.set_title('Training Episode Performance')
        ax2.grid(True, alpha=0.3)
        
        # 3. Evaluation performance with confidence intervals
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(eval_iterations, eval_performances, 'b-o', linewidth=2, markersize=6)
        ax3.fill_between(eval_iterations, eval_lower, eval_upper, alpha=0.2, color='blue')
        ax3.axhline(y=self.baseline.mean_performance, color='red', linestyle='--', 
                    label=f'Baseline: {self.baseline.mean_performance:.3f}')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Performance')
        ax3.set_title('Evaluation Performance with 95% CI')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Value pre-training
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(range(len(self.value_pretrain_losses)), self.value_pretrain_losses, 'purple', linewidth=2)
        ax4.set_xlabel('Pre-training Iteration')
        ax4.set_ylabel('Value Loss')
        ax4.set_title('Value Function Pre-training')
        ax4.grid(True, alpha=0.3)
        
        # 5. Improvement over baseline
        improvements = [m.improvement_vs_baseline for m in self.evaluation_metrics if m.iteration >= 0]
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(eval_iterations, improvements, 'g-s', linewidth=2, markersize=6)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Improvement (%)')
        ax5.set_title('Improvement Over Baseline')
        ax5.grid(True, alpha=0.3)
        
        # 6. Learning metrics
        if len(self.training_metrics) > 0:
            ax6 = plt.subplot(3, 2, 6)
            approx_kls = [m.approx_kl for m in self.training_metrics]
            clip_fractions = [m.clip_fraction for m in self.training_metrics]
            ax6.plot(iterations, approx_kls, 'r-', alpha=0.7, label='Approx KL')
            ax6.plot(iterations, clip_fractions, 'b-', alpha=0.7, label='Clip Fraction')
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('Value')
            ax6.set_title('PPO Diagnostic Metrics')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        report_path = os.path.join(self.checkpoint_dir, 'training_report.png')
        plt.savefig(report_path, dpi=150)
        print(f"   📈 Saved visualization: {report_path}")
        
        # Save detailed metrics
        metrics_data = {
            'configuration': {
                'episode_length': self.trainer.max_episode_steps,
                'learning_rate': self.trainer.learning_rate,
                'value_pretrain_iterations': 20,
                'total_iterations': self.iteration,
                'training_time_minutes': (time.time() - self.start_time) / 60
            },
            'baseline': {
                'mean': self.baseline.mean_performance,
                'confidence_lower': self.baseline.confidence_lower,
                'confidence_upper': self.baseline.confidence_upper
            },
            'final_performance': {
                'mean': self.evaluation_metrics[-1].mean_performance,
                'confidence_lower': self.evaluation_metrics[-1].confidence_lower,
                'confidence_upper': self.evaluation_metrics[-1].confidence_upper,
                'improvement_percent': self.evaluation_metrics[-1].improvement_vs_baseline
            },
            'training_metrics': [asdict(m) for m in self.training_metrics],
            'evaluation_metrics': [asdict(m) for m in self.evaluation_metrics],
            'value_pretrain_losses': self.value_pretrain_losses
        }
        
        metrics_path = os.path.join(self.checkpoint_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"   💾 Saved metrics: {metrics_path}")
        
        # Statistical summary
        print("\n📊 STATISTICAL SUMMARY:")
        print(f"   Baseline: {self.baseline.mean_performance:.4f} ± {self.baseline.std_error*1.96:.4f}")
        print(f"   Final: {self.evaluation_metrics[-1].mean_performance:.4f} ± {self.evaluation_metrics[-1].std_error*1.96:.4f}")
        print(f"   Improvement: {self.evaluation_metrics[-1].improvement_vs_baseline:+.1f}%")
        print(f"   Peak performance: {max(eval_performances):.4f} at iteration {eval_iterations[np.argmax(eval_performances)]}")
        print(f"   Plateau detected at: iteration {self.iteration}")
        
        return metrics_data


def main():
    """Run comprehensive PPO training"""
    print("🚀 COMPREHENSIVE PPO TRAINING SYSTEM")
    print("="*70)
    print("Training PPO until plateau with detailed metrics and analysis")
    print("="*70)
    
    # Create trainer
    trainer = ComprehensivePPOTrainer(
        checkpoint_dir=f"checkpoints/ppo_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        eval_frequency=10,
        checkpoint_frequency=10,
        plateau_patience=30,
        plateau_threshold=0.001
    )
    
    # Train until plateau
    trainer.train_until_plateau(max_iterations=200)
    
    # Generate comprehensive report
    report = trainer.generate_training_report()
    
    print("\n✅ Training complete!")
    print(f"   Best checkpoint saved at: {trainer.checkpoint_dir}/best_checkpoint.pt")
    
    return trainer, report


if __name__ == "__main__":
    trainer, report = main()