#!/usr/bin/env python3
"""
Train PPO Comprehensively Until Plateau

Uses optimal settings discovered through extensive experimentation:
- Episode length: 5000 steps
- Value pre-training: 20 iterations
- Low-variance evaluation
- Comprehensive metrics tracking
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training import PPOTrainer
from rl_training.ppo_transformer_model import PPOTransformerModel
from evaluation import PolicyEvaluator
from simulation.random_state_generator import generate_random_state


@dataclass
class TrainingIteration:
    """Metrics for a single training iteration"""
    iteration: int
    timestamp: float
    policy_loss: float
    value_loss: float
    total_loss: float
    episode_reward: float
    episode_length: int
    episode_catch_rate: float
    learning_rate: float


@dataclass
class EvaluationPoint:
    """Evaluation results at a specific iteration"""
    iteration: int
    mean_performance: float
    std_error: float
    confidence_lower: float
    confidence_upper: float
    improvement_percent: float
    is_significant: bool
    is_best: bool


class PPOComprehensiveTrainer:
    """
    Comprehensive PPO trainer with plateau detection and detailed metrics
    """
    
    def __init__(self,
                 save_dir: str = None,
                 episode_length: int = 5000,
                 value_pretrain_iters: int = 20,
                 eval_frequency: int = 10,
                 plateau_patience: int = 30):
        """Initialize comprehensive trainer"""
        
        # Create save directory
        if save_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = f"checkpoints/ppo_comprehensive_{timestamp}"
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training configuration (optimal settings)
        self.episode_length = episode_length
        self.value_pretrain_iters = value_pretrain_iters
        self.eval_frequency = eval_frequency
        self.plateau_patience = plateau_patience
        
        # Initialize model
        print("🚀 Initializing PPO Comprehensive Trainer")
        print(f"   Save directory: {save_dir}")
        print(f"   Episode length: {episode_length}")
        print(f"   Value pre-train: {value_pretrain_iters} iterations")
        
        # Create PPO model
        self.model = PPOTransformerModel(
            d_model=128,
            n_heads=8,
            n_layers=4,
            ffn_hidden=512,
            max_boids=50,
            dropout=0.1
        )
        
        # Load SL checkpoint
        self._load_sl_checkpoint("checkpoints/best_model.pt")
        
        # Optimizer
        self.learning_rate = 3e-5
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # PPO parameters
        self.clip_epsilon = 0.1
        self.ppo_epochs = 2
        self.rollout_steps = 512
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Initialize evaluators
        self.quick_eval = PolicyEvaluator(num_episodes=5, base_seed=8000)
        self.precise_eval = PolicyEvaluator(num_episodes=15, base_seed=9000)
        
        # Metrics storage
        self.training_iterations: List[TrainingIteration] = []
        self.evaluation_points: List[EvaluationPoint] = []
        self.value_pretrain_losses: List[float] = []
        
        # Plateau detection
        self.best_performance = -float('inf')
        self.iterations_without_improvement = 0
        self.performance_history = deque(maxlen=plateau_patience)
        
        # Training state
        self.iteration = 0
        self.start_time = time.time()
        self.baseline_result = None
    
    def _load_sl_checkpoint(self, checkpoint_path):
        """Load supervised learning checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Filter out value_head parameters since SL model doesn't have them
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                      if not k.startswith('value_head')}
        
        # Load weights (partial load, ignoring missing value_head)
        self.model.load_state_dict(state_dict, strict=False)
        
        print(f"   ✓ Loaded SL checkpoint from {checkpoint_path}")
    
    def establish_baseline(self):
        """Establish SL baseline performance"""
        print("\n📊 Establishing SL baseline...")
        
        from policy.transformer.transformer_policy import TransformerPolicy
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        
        self.baseline_result = self.precise_eval.evaluate_policy(sl_policy, "SL_Baseline")
        
        print(f"   Baseline: {self.baseline_result.overall_catch_rate:.4f}")
        print(f"   95% CI: [{self.baseline_result.confidence_95_lower:.4f}, {self.baseline_result.confidence_95_upper:.4f}]")
        
        return self.baseline_result
    
    def pretrain_value_function(self):
        """Pre-train value function for stability"""
        print(f"\n🎓 Pre-training value function ({self.value_pretrain_iters} iterations)...")
        
        # Freeze policy parameters
        policy_params = []
        value_params = []
        
        for name, param in self.model.named_parameters():
            if 'value_head' in name:
                value_params.append(param)
            else:
                policy_params.append(param)
                param.requires_grad = False
        
        # Value-only optimizer
        value_optimizer = torch.optim.Adam(value_params, lr=3e-4)
        
        # Pre-training loop
        for i in range(self.value_pretrain_iters):
            # Generate episode
            initial_state = generate_random_state(12, 400, 300)
            
            # Collect experience
            states, actions, rewards, values, dones = self._collect_episode(initial_state)
            
            # Calculate returns
            returns = self._calculate_returns(rewards, values, dones)
            
            # Update value function
            value_loss = self._update_value_function(states, returns, value_optimizer)
            self.value_pretrain_losses.append(value_loss)
            
            if (i + 1) % 5 == 0:
                print(f"   Iteration {i+1}/{self.value_pretrain_iters}: Loss = {value_loss:.4f}")
        
        # Unfreeze policy parameters
        for param in policy_params:
            param.requires_grad = True
        
        # Reset main optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print(f"   Pre-training complete: {self.value_pretrain_losses[0]:.4f} → {self.value_pretrain_losses[-1]:.4f}")
    
    def _collect_episode(self, initial_state):
        """Collect one episode of experience"""
        from simulation.state_manager import StateManager
        
        state_manager = StateManager()
        
        # Create a policy wrapper for the state manager
        class ModelPolicy:
            def __init__(self, model):
                self.model = model
                
            def get_action(self, structured_inputs):
                with torch.no_grad():
                    action_logits, _ = self.model([structured_inputs])
                    # action_logits should be shape [2] for x,y
                    if action_logits.dim() == 0:
                        # Single value, duplicate for x,y
                        action = torch.tanh(action_logits).item()
                        return [action, action]
                    else:
                        # Convert to action in [-1, 1] range
                        action = torch.tanh(action_logits)
                        # Ensure we return a list of two floats
                        if action.shape[0] >= 2:
                            return [float(action[0]), float(action[1])]
                        else:
                            # If only one value, duplicate it
                            val = float(action[0]) if action.shape[0] > 0 else 0.0
                            return [val, val]
        
        policy = ModelPolicy(self.model)
        state_manager.init(initial_state, policy)
        
        states = []
        actions = []
        rewards = []
        values = []
        dones = []
        
        for _ in range(self.rollout_steps):
            state = state_manager.get_state()
            
            # Get structured input from state manager
            structured_input = state_manager._convert_state_to_structured_inputs(state)
            
            # Get action and value from model
            with torch.no_grad():
                action_logits, value = self.model([structured_input])
                
                # Convert to action in [-1, 1] range
                action = torch.tanh(action_logits[0])
            
            # Step environment
            result = state_manager.step()
            reward = len(result.get('caught_boids', []))
            done = len(state['boids_states']) == 0
            
            # Store experience
            states.append(structured_input)
            actions.append(action.numpy())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            
            if done:
                break
        
        return states, actions, rewards, values, dones
    
    def _calculate_returns(self, rewards, values, dones):
        """Calculate discounted returns"""
        returns = []
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns.insert(0, running_return)
        
        return torch.tensor(returns, dtype=torch.float32)
    
    def _update_value_function(self, states, returns, optimizer):
        """Update value function only"""
        _, values = self.model(states)
        values = values.squeeze()
        
        value_loss = F.mse_loss(values, returns)
        
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()
        
        return value_loss.item()
    
    def train_iteration(self):
        """Run one PPO training iteration"""
        # Generate initial state
        initial_state = generate_random_state(12, 400, 300)
        
        # Collect experience
        states, actions, rewards, values, dones = self._collect_episode(initial_state)
        
        # Calculate returns and advantages
        returns = self._calculate_returns(rewards, values, dones)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        advantages = returns - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Episode metrics
        episode_reward = sum(rewards)
        episode_length = len(rewards)
        episode_catch_rate = episode_reward / initial_state['n_boids']
        
        # PPO update
        # Convert actions list to tensor
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
        
        total_policy_loss = 0
        total_value_loss = 0
        num_updates = 0
        
        # Get old log probs (using continuous action space)
        with torch.no_grad():
            action_logits, _ = self.model(states)
            # Assume Gaussian policy for continuous actions
            old_action_mean = torch.tanh(action_logits)
            old_action_std = torch.ones_like(old_action_mean) * 0.1
            old_dist = torch.distributions.Normal(old_action_mean, old_action_std)
            old_log_probs = old_dist.log_prob(actions_tensor).sum(dim=-1)
        
        for _ in range(self.ppo_epochs):
            # Forward pass
            action_logits, new_values = self.model(states)
            new_values = new_values.squeeze()
            
            # Policy loss (continuous action space)
            new_action_mean = torch.tanh(action_logits)
            new_action_std = torch.ones_like(new_action_mean) * 0.1
            new_dist = torch.distributions.Normal(new_action_mean, new_action_std)
            new_log_probs = new_dist.log_prob(actions_tensor).sum(dim=-1)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_updates += 1
        
        # Record metrics
        iteration_metrics = TrainingIteration(
            iteration=self.iteration,
            timestamp=time.time() - self.start_time,
            policy_loss=total_policy_loss / num_updates,
            value_loss=total_value_loss / num_updates,
            total_loss=(total_policy_loss + total_value_loss) / num_updates,
            episode_reward=episode_reward,
            episode_length=episode_length,
            episode_catch_rate=episode_catch_rate,
            learning_rate=self.learning_rate
        )
        
        self.training_iterations.append(iteration_metrics)
        return iteration_metrics
    
    def evaluate_performance(self, use_precise=False):
        """Evaluate current performance"""
        evaluator = self.precise_eval if use_precise else self.quick_eval
        
        # Create policy wrapper for evaluation
        class PolicyWrapper:
            def __init__(self, model):
                self.model = model
            
            def get_action(self, structured_inputs):
                with torch.no_grad():
                    action_logits, _ = self.model([structured_inputs])
                    # Return continuous action in [-1, 1] range
                    action = torch.tanh(action_logits[0])
                    # Ensure we return a list of two floats
                    return [float(action[0]), float(action[1])]
        
        policy = PolicyWrapper(self.model)
        result = evaluator.evaluate_policy(policy, f"PPO_Iter{self.iteration}")
        
        # Calculate improvement
        improvement = (result.overall_catch_rate - self.baseline_result.overall_catch_rate) / self.baseline_result.overall_catch_rate * 100
        
        # Check significance
        is_significant = result.confidence_95_lower > self.baseline_result.confidence_95_upper
        
        # Check if best
        is_best = result.overall_catch_rate > self.best_performance
        if is_best:
            self.best_performance = result.overall_catch_rate
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        
        # Record evaluation
        eval_point = EvaluationPoint(
            iteration=self.iteration,
            mean_performance=result.overall_catch_rate,
            std_error=result.std_error,
            confidence_lower=result.confidence_95_lower,
            confidence_upper=result.confidence_95_upper,
            improvement_percent=improvement,
            is_significant=is_significant,
            is_best=is_best
        )
        
        self.evaluation_points.append(eval_point)
        self.performance_history.append(result.overall_catch_rate)
        
        return eval_point
    
    def check_plateau(self):
        """Check if training has plateaued"""
        if len(self.performance_history) < self.plateau_patience:
            return False
        
        # Check variance in recent performance
        recent_perfs = list(self.performance_history)
        perf_std = np.std(recent_perfs)
        perf_range = max(recent_perfs) - min(recent_perfs)
        
        # Plateau if no improvement for patience iterations
        # OR if performance variance is very low
        return (
            self.iterations_without_improvement >= self.plateau_patience or
            (perf_std < 0.001 and perf_range < 0.002)
        )
    
    def save_checkpoint(self, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_performance': self.best_performance,
            'config': {
                'episode_length': self.episode_length,
                'learning_rate': self.learning_rate,
                'value_pretrain_iters': self.value_pretrain_iters
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_iter{self.iteration}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
    
    def train_until_plateau(self, max_iterations=200):
        """Train until performance plateaus"""
        print(f"\n🏃 Training PPO until plateau (max {max_iterations} iterations)...")
        
        # Establish baseline
        self.establish_baseline()
        
        # Pre-train value function
        self.pretrain_value_function()
        
        # Initial evaluation
        print("\n📊 Initial evaluation...")
        eval_result = self.evaluate_performance(use_precise=True)
        print(f"   Initial: {eval_result.mean_performance:.4f} ({eval_result.improvement_percent:+.1f}%)")
        
        # Training loop
        print("\n🔄 Training progress:")
        
        while self.iteration < max_iterations:
            self.iteration += 1
            
            # Train iteration
            metrics = self.train_iteration()
            
            # Progress log
            if self.iteration % 5 == 0:
                print(f"   Iter {self.iteration:3d}: "
                      f"Loss P:{metrics.policy_loss:.3f} V:{metrics.value_loss:.3f} | "
                      f"Episode: R={metrics.episode_reward:.0f} CR={metrics.episode_catch_rate:.3f}")
            
            # Evaluation
            if self.iteration % self.eval_frequency == 0:
                print(f"\n   📊 Evaluation at iteration {self.iteration}:")
                
                # Use precise evaluation every 20 iterations
                use_precise = (self.iteration % 20 == 0)
                eval_result = self.evaluate_performance(use_precise=use_precise)
                
                print(f"      Performance: {eval_result.mean_performance:.4f} "
                      f"[{eval_result.confidence_lower:.4f}, {eval_result.confidence_upper:.4f}]")
                print(f"      Improvement: {eval_result.improvement_percent:+.1f}%"
                      f"{' ✅ Significant!' if eval_result.is_significant else ''}"
                      f"{' 🌟 NEW BEST!' if eval_result.is_best else ''}")
                
                # Save checkpoint if best
                if eval_result.is_best:
                    self.save_checkpoint(is_best=True)
                
                # Check plateau
                if self.check_plateau():
                    print(f"\n   🏁 Training plateaued after {self.iteration} iterations!")
                    break
            
            # Regular checkpoint
            if self.iteration % 20 == 0:
                self.save_checkpoint()
        
        # Final precise evaluation
        print("\n📊 Final evaluation (30 episodes for high precision)...")
        final_evaluator = PolicyEvaluator(num_episodes=30, base_seed=10000)
        
        class PolicyWrapper:
            def __init__(self, model):
                self.model = model
            
            def get_action(self, structured_inputs):
                with torch.no_grad():
                    action_logits, _ = self.model([structured_inputs])
                    # Return continuous action in [-1, 1] range
                    action = torch.tanh(action_logits[0])
                    # Ensure we return a list of two floats
                    return [float(action[0]), float(action[1])]
        
        policy = PolicyWrapper(self.model)
        final_result = final_evaluator.evaluate_policy(policy, "PPO_Final")
        
        final_improvement = (final_result.overall_catch_rate - self.baseline_result.overall_catch_rate) / self.baseline_result.overall_catch_rate * 100
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   Baseline: {self.baseline_result.overall_catch_rate:.4f} ± {self.baseline_result.std_error*1.96:.4f}")
        print(f"   Final: {final_result.overall_catch_rate:.4f} ± {final_result.std_error*1.96:.4f}")
        print(f"   Improvement: {final_improvement:+.1f}%")
        print(f"   Statistically significant: {'YES ✅' if final_result.confidence_95_lower > self.baseline_result.confidence_95_upper else 'NO ❌'}")
        print(f"   Total iterations: {self.iteration}")
        print(f"   Training time: {(time.time() - self.start_time)/60:.1f} minutes")
    
    def generate_report(self):
        """Generate comprehensive training report"""
        print("\n📝 Generating training report...")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Training losses
        iterations = [m.iteration for m in self.training_iterations]
        policy_losses = [m.policy_loss for m in self.training_iterations]
        value_losses = [m.value_loss for m in self.training_iterations]
        
        ax = axes[0, 0]
        ax.plot(iterations, policy_losses, 'b-', alpha=0.7, label='Policy Loss')
        ax.plot(iterations, value_losses, 'r-', alpha=0.7, label='Value Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Episode performance
        catch_rates = [m.episode_catch_rate for m in self.training_iterations]
        
        ax = axes[0, 1]
        ax.plot(iterations, catch_rates, 'g-', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Episode Catch Rate')
        ax.set_title('Training Episode Performance')
        ax.grid(True, alpha=0.3)
        
        # 3. Evaluation performance
        eval_iters = [e.iteration for e in self.evaluation_points]
        eval_means = [e.mean_performance for e in self.evaluation_points]
        eval_lower = [e.confidence_lower for e in self.evaluation_points]
        eval_upper = [e.confidence_upper for e in self.evaluation_points]
        
        ax = axes[0, 2]
        ax.plot(eval_iters, eval_means, 'b-o', linewidth=2)
        ax.fill_between(eval_iters, eval_lower, eval_upper, alpha=0.2)
        ax.axhline(y=self.baseline_result.overall_catch_rate, color='red', 
                   linestyle='--', label=f'Baseline: {self.baseline_result.overall_catch_rate:.3f}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Performance')
        ax.set_title('Evaluation Performance (95% CI)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Value pre-training
        ax = axes[1, 0]
        ax.plot(range(len(self.value_pretrain_losses)), self.value_pretrain_losses, 'purple', linewidth=2)
        ax.set_xlabel('Pre-training Iteration')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Function Pre-training')
        ax.grid(True, alpha=0.3)
        
        # 5. Improvement over baseline
        improvements = [e.improvement_percent for e in self.evaluation_points]
        
        ax = axes[1, 1]
        ax.plot(eval_iters, improvements, 'g-s', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Improvement Over Baseline')
        ax.grid(True, alpha=0.3)
        
        # 6. Statistical significance
        ax = axes[1, 2]
        significant_iters = [e.iteration for e in self.evaluation_points if e.is_significant]
        significant_perfs = [e.mean_performance for e in self.evaluation_points if e.is_significant]
        not_significant_iters = [e.iteration for e in self.evaluation_points if not e.is_significant]
        not_significant_perfs = [e.mean_performance for e in self.evaluation_points if not e.is_significant]
        
        ax.scatter(significant_iters, significant_perfs, c='green', s=50, label='Significant', alpha=0.7)
        ax.scatter(not_significant_iters, not_significant_perfs, c='red', s=50, label='Not Significant', alpha=0.7)
        ax.axhline(y=self.baseline_result.overall_catch_rate, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Performance')
        ax.set_title('Statistical Significance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        report_path = os.path.join(self.save_dir, 'training_report.png')
        plt.savefig(report_path, dpi=150)
        print(f"   📈 Saved visualization: {report_path}")
        
        # Save detailed metrics
        metrics = {
            'configuration': {
                'episode_length': self.episode_length,
                'learning_rate': self.learning_rate,
                'value_pretrain_iterations': self.value_pretrain_iters,
                'total_iterations': self.iteration,
                'training_time_minutes': (time.time() - self.start_time) / 60
            },
            'baseline': {
                'mean': self.baseline_result.overall_catch_rate,
                'confidence_lower': self.baseline_result.confidence_95_lower,
                'confidence_upper': self.baseline_result.confidence_95_upper
            },
            'best_performance': {
                'iteration': max((e.iteration for e in self.evaluation_points if e.is_best), default=0),
                'mean': self.best_performance,
                'improvement_percent': max((e.improvement_percent for e in self.evaluation_points), default=0)
            },
            'training_iterations': [asdict(m) for m in self.training_iterations],
            'evaluation_points': [asdict(e) for e in self.evaluation_points],
            'value_pretrain_losses': self.value_pretrain_losses
        }
        
        metrics_path = os.path.join(self.save_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   💾 Saved metrics: {metrics_path}")
        
        # Summary statistics
        print("\n📊 TRAINING SUMMARY:")
        print(f"   Peak performance: {self.best_performance:.4f}")
        print(f"   Peak improvement: {max((e.improvement_percent for e in self.evaluation_points), default=0):.1f}%")
        print(f"   Iterations to plateau: {self.iteration}")
        print(f"   Significant improvements: {sum(1 for e in self.evaluation_points if e.is_significant)}/{len(self.evaluation_points)}")
        
        return metrics


def main():
    """Run comprehensive PPO training"""
    print("🚀 COMPREHENSIVE PPO TRAINING WITH PLATEAU DETECTION")
    print("="*70)
    print("Using optimal configuration from extensive experiments")
    print("="*70)
    
    # Create trainer
    trainer = PPOComprehensiveTrainer(
        episode_length=5000,  # Optimal from experiments
        value_pretrain_iters=20,  # Optimal from experiments
        eval_frequency=10,
        plateau_patience=30
    )
    
    # Train until plateau
    trainer.train_until_plateau(max_iterations=200)
    
    # Generate report
    report = trainer.generate_report()
    
    print(f"\n✅ Training complete!")
    print(f"   Results saved to: {trainer.save_dir}")
    
    return trainer, report


if __name__ == "__main__":
    trainer, report = main()