#!/usr/bin/env python3
"""
Production PPO Training Script

Trains PPO with optimal settings discovered through extensive experimentation:
- Episode length: 5000 steps (optimal for learning)
- Value pre-training: 20 iterations (for stability)
- Learning rate: 3e-5
- Plateau detection for automatic stopping
- Comprehensive metrics and checkpointing
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_training.ppo_transformer_model import PPOTransformerModel
from rl_training.ppo_experience_buffer import PPOExperienceBuffer, PPOExperience, PPORolloutCollector
from evaluation import PolicyEvaluator
from simulation.state_manager import StateManager
from simulation.random_state_generator import generate_random_state
from rewards.reward_processor import RewardProcessor
from policy.transformer.transformer_policy import TransformerPolicy


@dataclass
class TrainingMetrics:
    """Metrics for each training iteration"""
    iteration: int
    timestamp: float
    
    # Loss metrics
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    
    # Episode metrics  
    episode_reward: float
    episode_length: int
    episode_catch_rate: float
    
    # Learning metrics
    learning_rate: float
    clip_fraction: float
    value_pred_mean: float
    advantage_mean: float
    advantage_std: float


@dataclass
class EvaluationResult:
    """Evaluation result with confidence intervals"""
    iteration: int
    mean_performance: float
    std_error: float
    confidence_lower: float
    confidence_upper: float
    improvement_percent: float
    is_significant: bool
    is_best: bool


class ProductionPPOTrainer:
    """
    Production-grade PPO trainer with all optimizations
    """
    
    def __init__(self,
                 save_dir: str = None,
                 episode_length: int = 5000,  # Optimal from experiments
                 value_pretrain_iterations: int = 20,  # Optimal from experiments
                 learning_rate: float = 3e-5,  # Optimal from experiments
                 eval_frequency: int = 10,
                 checkpoint_frequency: int = 10,
                 plateau_patience: int = 30,
                 plateau_threshold: float = 0.002):
        """
        Initialize production PPO trainer
        
        Args:
            save_dir: Directory to save checkpoints and logs
            episode_length: Max steps per episode
            value_pretrain_iterations: Iterations for value function pre-training
            learning_rate: Learning rate for PPO
            eval_frequency: Evaluate every N iterations
            checkpoint_frequency: Save checkpoint every N iterations
            plateau_patience: Iterations without improvement before stopping
            plateau_threshold: Minimum improvement to not count as plateau
        """
        # Create save directory
        if save_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = f"checkpoints/ppo_production_{timestamp}"
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Save configuration
        self.config = {
            'episode_length': episode_length,
            'value_pretrain_iterations': value_pretrain_iterations,
            'learning_rate': learning_rate,
            'eval_frequency': eval_frequency,
            'checkpoint_frequency': checkpoint_frequency,
            'plateau_patience': plateau_patience,
            'plateau_threshold': plateau_threshold,
            'device': 'cpu',  # For stability
            'rollout_steps': 512,
            'ppo_epochs': 2,
            'mini_batch_size': 64,
            'clip_epsilon': 0.1,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'max_grad_norm': 0.5
        }
        
        # Save configuration
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print("🚀 Production PPO Trainer")
        print("=" * 70)
        print(f"Save directory: {save_dir}")
        print(f"Episode length: {episode_length} (optimal)")
        print(f"Value pre-train: {value_pretrain_iterations} iterations")
        print(f"Learning rate: {learning_rate}")
        print(f"Plateau detection: {plateau_patience} patience, {plateau_threshold} threshold")
        print("=" * 70)
        
        # Initialize model
        self.model = PPOTransformerModel.from_sl_checkpoint("checkpoints/best_model.pt")
        self.device = torch.device(self.config['device'])
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize simulation components
        self.state_manager = StateManager()
        self.reward_processor = RewardProcessor()
        
        # Initialize evaluators
        self.quick_evaluator = PolicyEvaluator(num_episodes=5, base_seed=10000)
        self.standard_evaluator = PolicyEvaluator(num_episodes=15, base_seed=11000)
        self.precise_evaluator = PolicyEvaluator(num_episodes=30, base_seed=12000)
        
        # Metrics storage
        self.training_metrics: List[TrainingMetrics] = []
        self.evaluation_results: List[EvaluationResult] = []
        self.value_pretrain_losses: List[float] = []
        
        # Plateau detection
        self.best_performance = -float('inf')
        self.iterations_without_improvement = 0
        self.performance_history = deque(maxlen=plateau_patience)
        
        # Training state
        self.iteration = 0
        self.total_timesteps = 0
        self.start_time = time.time()
        self.baseline_result = None
        
        # Log file
        self.log_file = open(os.path.join(save_dir, 'training.log'), 'w')
        self._log("Production PPO Training Started")
        self._log(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def _log(self, message: str):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.log_file.write(log_message + '\n')
        self.log_file.flush()
    
    def establish_baseline(self):
        """Establish SL baseline with confidence intervals"""
        self._log("\n📊 Establishing SL baseline...")
        
        sl_policy = TransformerPolicy("checkpoints/best_model.pt")
        self.baseline_result = self.standard_evaluator.evaluate_policy(sl_policy, "SL_Baseline")
        
        self._log(f"Baseline performance: {self.baseline_result.overall_catch_rate:.4f}")
        self._log(f"95% CI: [{self.baseline_result.confidence_95_lower:.4f}, {self.baseline_result.confidence_95_upper:.4f}]")
        self._log(f"Std error: {self.baseline_result.std_error:.4f}")
        
        return self.baseline_result
    
    def pretrain_value_function(self):
        """Pre-train value function for stability"""
        self._log(f"\n🎓 Pre-training value function ({self.config['value_pretrain_iterations']} iterations)...")
        
        # Separate parameters
        value_params = []
        policy_params = []
        
        for name, param in self.model.named_parameters():
            if 'value_head' in name:
                value_params.append(param)
            else:
                policy_params.append(param)
                param.requires_grad = False
        
        # Value-only optimizer
        value_optimizer = optim.Adam(value_params, lr=3e-4)
        
        # Pre-training loop
        for iter_idx in range(self.config['value_pretrain_iterations']):
            # Generate random initial state
            initial_state = generate_random_state(12, 400, 300)
            
            # Collect episode for value training
            buffer = self._collect_rollout(initial_state, self.config['rollout_steps'])
            
            # Get batch data
            batch_data = buffer.get_batch_data()
            if len(batch_data) == 0:
                continue
            
            # Train value function
            total_value_loss = 0.0
            num_updates = 0
            
            for epoch in range(4):  # Value pre-train epochs
                indices = torch.randperm(len(buffer))
                
                for start_idx in range(0, len(buffer), self.config['mini_batch_size']):
                    end_idx = min(start_idx + self.config['mini_batch_size'], len(buffer))
                    mb_indices = indices[start_idx:end_idx]
                    
                    # Mini-batch data
                    mb_inputs = [batch_data['structured_inputs'][i] for i in mb_indices]
                    mb_returns = batch_data['returns'][mb_indices].to(self.device)
                    
                    # Forward pass
                    _, values = self.model(mb_inputs)
                    values = values.squeeze()
                    
                    # Value loss
                    value_loss = F.mse_loss(values, mb_returns)
                    
                    # Update
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(value_params, self.config['max_grad_norm'])
                    value_optimizer.step()
                    
                    total_value_loss += value_loss.item()
                    num_updates += 1
            
            avg_loss = total_value_loss / num_updates if num_updates > 0 else 0
            self.value_pretrain_losses.append(avg_loss)
            
            if (iter_idx + 1) % 5 == 0:
                self._log(f"  Iteration {iter_idx + 1}/{self.config['value_pretrain_iterations']}: Loss = {avg_loss:.4f}")
        
        # Unfreeze policy parameters
        for param in policy_params:
            param.requires_grad = True
        
        # Reset main optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        self._log(f"Value pre-training complete: {self.value_pretrain_losses[0]:.4f} → {self.value_pretrain_losses[-1]:.4f}")
    
    def _collect_rollout(self, initial_state: Dict, num_steps: int) -> PPOExperienceBuffer:
        """Collect rollout using simulation"""
        buffer = PPOExperienceBuffer(gamma=self.config['gamma'], gae_lambda=self.config['gae_lambda'])
        
        # Create policy wrapper
        class PolicyWrapper:
            def __init__(self, model):
                self.model = model
            
            def get_action(self, structured_inputs):
                with torch.no_grad():
                    # StateManager already converts state to structured inputs
                    # Pass single input directly (model handles wrapping)
                    action_logits = self.model(structured_inputs, return_value=False)
                    # action_logits should be shape [2] for single input
                    action = torch.tanh(action_logits).cpu().numpy().tolist()
                    return action
        
        policy = PolicyWrapper(self.model)
        
        # Initialize simulation
        self.state_manager.init(initial_state, policy)
        episode_steps = 0
        
        for step in range(num_steps):
            # Get current state
            state = self.state_manager.get_state()
            structured_input = self._prepare_structured_input(state)
            
            # Get action, value, and log prob
            with torch.no_grad():
                # Use the model's proper get_action_and_value method
                action, log_prob, value = self.model.get_action_and_value(structured_input, deterministic=False)
            
            # Step environment
            step_result = self.state_manager.step()
            
            # Calculate reward
            reward = len(step_result.get('caught_boids', []))
            
            # Check if done
            done = len(step_result['boids_states']) == 0 or episode_steps >= self.config['episode_length']
            
            # Add experience
            experience = PPOExperience(
                structured_input=structured_input,
                action=action.detach(),
                log_prob=log_prob.detach(),
                value=value.detach(),
                reward=reward,
                done=done
            )
            buffer.add_experience(experience)
            
            episode_steps += 1
            
            if done:
                if step < num_steps - 1:
                    # Reset for new episode
                    self.state_manager.init(generate_random_state(12, 400, 300), policy)
                    episode_steps = 0
        
        return buffer
    
    def _prepare_structured_input(self, state: Dict) -> Dict:
        """Convert state to structured input format"""
        predator_state = state['predator_state']
        boids_states = state['boids_states']
        
        # Normalize coordinates
        canvas_width = 400
        canvas_height = 300
        
        structured_input = {
            'context': {
                'canvasWidth': canvas_width / 500,  # Normalize
                'canvasHeight': canvas_height / 500
            },
            'predator': {
                'velX': predator_state['velocity']['x'] / 10,  # Normalize velocity
                'velY': predator_state['velocity']['y'] / 10
            },
            'boids': []
        }
        
        # Add boids (relative positions)
        for boid in boids_states:
            rel_x = (boid['position']['x'] - predator_state['position']['x']) / canvas_width
            rel_y = (boid['position']['y'] - predator_state['position']['y']) / canvas_height
            
            structured_input['boids'].append({
                'relX': rel_x,
                'relY': rel_y,
                'velX': boid['velocity']['x'] / 10,
                'velY': boid['velocity']['y'] / 10
            })
        
        return structured_input
    
    def train_iteration(self) -> TrainingMetrics:
        """Run one PPO training iteration"""
        iteration_start = time.time()
        
        # Generate initial state
        initial_state = generate_random_state(12, 400, 300)
        
        # Collect rollout
        buffer = self._collect_rollout(initial_state, self.config['rollout_steps'])
        
        # Get batch data
        batch_data = buffer.get_batch_data()
        if len(batch_data) == 0:
            return None
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_clip_fraction = 0
        num_updates = 0
        
        # PPO epochs
        for epoch in range(self.config['ppo_epochs']):
            indices = torch.randperm(len(buffer))
            
            for start_idx in range(0, len(buffer), self.config['mini_batch_size']):
                end_idx = min(start_idx + self.config['mini_batch_size'], len(buffer))
                mb_indices = indices[start_idx:end_idx]
                
                # Mini-batch data
                mb_inputs = [batch_data['structured_inputs'][i] for i in mb_indices]
                mb_actions = batch_data['actions'][mb_indices].to(self.device)
                mb_old_log_probs = batch_data['log_probs'][mb_indices].to(self.device)
                mb_advantages = batch_data['advantages'][mb_indices].to(self.device)
                mb_returns = batch_data['returns'][mb_indices].to(self.device)
                mb_old_values = batch_data['values'][mb_indices].to(self.device)
                
                # Evaluate actions with current policy
                new_log_probs, values, entropy = self.model.evaluate_actions(mb_inputs, mb_actions)
                values = values.squeeze()
                
                # Policy loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon']) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -self.config['entropy_coef'] * entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.config['value_loss_coef'] * value_loss + entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_clip_fraction += ((ratio - 1).abs() > self.config['clip_epsilon']).float().mean().item()
                num_updates += 1
        
        # Get episode stats
        stats = buffer.get_statistics()
        
        # Create metrics
        metrics = TrainingMetrics(
            iteration=self.iteration,
            timestamp=time.time() - self.start_time,
            policy_loss=total_policy_loss / num_updates,
            value_loss=total_value_loss / num_updates,
            entropy_loss=total_entropy_loss / num_updates,
            total_loss=(total_policy_loss + total_value_loss + total_entropy_loss) / num_updates,
            episode_reward=stats.get('total_reward', 0),
            episode_length=stats.get('rollout_length', 0),
            episode_catch_rate=stats.get('total_reward', 0) / 12,  # 12 boids
            learning_rate=self.config['learning_rate'],
            clip_fraction=total_clip_fraction / num_updates,
            value_pred_mean=batch_data['values'].mean().item(),
            advantage_mean=batch_data['advantages'].mean().item(),
            advantage_std=batch_data['advantages'].std().item()
        )
        
        self.training_metrics.append(metrics)
        self.total_timesteps += stats.get('rollout_length', 0)
        
        return metrics
    
    def evaluate_performance(self, use_precise: bool = False) -> EvaluationResult:
        """Evaluate current performance"""
        evaluator = self.precise_evaluator if use_precise else self.standard_evaluator
        
        # Create policy wrapper
        class PolicyWrapper:
            def __init__(self, model):
                self.model = model
            
            def get_action(self, structured_inputs):
                with torch.no_grad():
                    # StateManager already converts state to structured inputs
                    # Pass single input directly (model handles wrapping)
                    action_logits = self.model(structured_inputs, return_value=False)
                    # action_logits should be shape [2] for single input
                    action = torch.tanh(action_logits).cpu().numpy().tolist()
                    return action
        
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
        
        # Create evaluation result
        eval_result = EvaluationResult(
            iteration=self.iteration,
            mean_performance=result.overall_catch_rate,
            std_error=result.std_error,
            confidence_lower=result.confidence_95_lower,
            confidence_upper=result.confidence_95_upper,
            improvement_percent=improvement,
            is_significant=is_significant,
            is_best=is_best
        )
        
        self.evaluation_results.append(eval_result)
        self.performance_history.append(result.overall_catch_rate)
        
        return eval_result
    
    def check_plateau(self) -> bool:
        """Check if training has plateaued"""
        if len(self.performance_history) < self.config['plateau_patience']:
            return False
        
        # Check recent performance variance
        recent_perfs = list(self.performance_history)
        perf_range = max(recent_perfs) - min(recent_perfs)
        
        # Plateau if no improvement for patience iterations
        # OR if performance range is very small
        return (
            self.iterations_without_improvement >= self.config['plateau_patience'] or
            perf_range < self.config['plateau_threshold']
        )
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': self.iteration,
            'total_timesteps': self.total_timesteps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_performance': self.best_performance,
            'config': self.config,
            'training_time': time.time() - self.start_time
        }
        
        # Regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_iter{self.iteration}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            self._log(f"💾 Saved best checkpoint: {self.best_performance:.4f}")
    
    def train_until_plateau(self, max_iterations: int = 300):
        """Train until performance plateaus"""
        self._log(f"\n🏃 Training PPO until plateau (max {max_iterations} iterations)...")
        
        # Establish baseline
        self.establish_baseline()
        
        # Pre-train value function
        self.pretrain_value_function()
        
        # Initial evaluation
        self._log("\n📊 Initial evaluation...")
        eval_result = self.evaluate_performance(use_precise=True)
        self._log(f"Initial performance: {eval_result.mean_performance:.4f} ({eval_result.improvement_percent:+.1f}%)")
        
        # Training loop
        self._log("\n🔄 Starting training loop...")
        
        while self.iteration < max_iterations:
            self.iteration += 1
            
            # Train iteration
            metrics = self.train_iteration()
            
            if metrics is None:
                continue
            
            # Progress log
            if self.iteration % 5 == 0:
                self._log(
                    f"Iter {self.iteration:3d}: "
                    f"Loss P:{metrics.policy_loss:.3f} V:{metrics.value_loss:.3f} | "
                    f"Episode: R={metrics.episode_reward:.0f} CR={metrics.episode_catch_rate:.3f}"
                )
            
            # Evaluation
            if self.iteration % self.config['eval_frequency'] == 0:
                self._log(f"\n📊 Evaluation at iteration {self.iteration}:")
                
                # Use precise evaluation every 30 iterations
                use_precise = (self.iteration % 30 == 0)
                eval_result = self.evaluate_performance(use_precise=use_precise)
                
                self._log(
                    f"Performance: {eval_result.mean_performance:.4f} "
                    f"[{eval_result.confidence_lower:.4f}, {eval_result.confidence_upper:.4f}]"
                )
                self._log(
                    f"Improvement: {eval_result.improvement_percent:+.1f}%"
                    f"{' ✅ Significant!' if eval_result.is_significant else ''}"
                    f"{' 🌟 NEW BEST!' if eval_result.is_best else ''}"
                )
                
                # Check plateau
                if self.check_plateau():
                    self._log(f"\n🏁 Training plateaued after {self.iteration} iterations!")
                    break
            
            # Save checkpoint
            if self.iteration % self.config['checkpoint_frequency'] == 0:
                self.save_checkpoint(is_best=eval_result.is_best if 'eval_result' in locals() else False)
        
        # Final evaluation with high precision
        self._log("\n📊 Final evaluation (30 episodes)...")
        final_result = self.evaluate_performance(use_precise=True)
        
        self._log(f"\n🏆 FINAL RESULTS:")
        self._log(f"Baseline: {self.baseline_result.overall_catch_rate:.4f} ± {self.baseline_result.std_error*1.96:.4f}")
        self._log(f"Final: {final_result.mean_performance:.4f} ± {final_result.std_error*1.96:.4f}")
        self._log(f"Improvement: {final_result.improvement_percent:+.1f}%")
        self._log(f"Statistically significant: {'YES ✅' if final_result.is_significant else 'NO ❌'}")
        self._log(f"Total iterations: {self.iteration}")
        self._log(f"Total timesteps: {self.total_timesteps:,}")
        self._log(f"Training time: {(time.time() - self.start_time)/60:.1f} minutes")
        
        # Save final checkpoint
        self.save_checkpoint(is_best=True)
    
    def generate_report(self):
        """Generate comprehensive training report"""
        self._log("\n📝 Generating comprehensive report...")
        
        # Create visualizations
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # 1. Training losses
        iterations = [m.iteration for m in self.training_metrics]
        policy_losses = [m.policy_loss for m in self.training_metrics]
        value_losses = [m.value_loss for m in self.training_metrics]
        
        ax = axes[0, 0]
        ax.plot(iterations, policy_losses, 'b-', alpha=0.7, label='Policy Loss')
        ax.plot(iterations, value_losses, 'r-', alpha=0.7, label='Value Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Episode performance
        catch_rates = [m.episode_catch_rate for m in self.training_metrics]
        
        ax = axes[0, 1]
        ax.plot(iterations, catch_rates, 'g-', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Episode Catch Rate')
        ax.set_title('Training Episode Performance')
        ax.grid(True, alpha=0.3)
        
        # 3. Evaluation performance with CI
        eval_iters = [e.iteration for e in self.evaluation_results]
        eval_means = [e.mean_performance for e in self.evaluation_results]
        eval_lower = [e.confidence_lower for e in self.evaluation_results]
        eval_upper = [e.confidence_upper for e in self.evaluation_results]
        
        ax = axes[1, 0]
        ax.plot(eval_iters, eval_means, 'b-o', linewidth=2, markersize=6)
        ax.fill_between(eval_iters, eval_lower, eval_upper, alpha=0.2, color='blue')
        ax.axhline(y=self.baseline_result.overall_catch_rate, color='red', linestyle='--', 
                   label=f'Baseline: {self.baseline_result.overall_catch_rate:.3f}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Performance')
        ax.set_title('Evaluation Performance with 95% CI')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Value pre-training
        ax = axes[1, 1]
        ax.plot(range(len(self.value_pretrain_losses)), self.value_pretrain_losses, 'purple', linewidth=2)
        ax.set_xlabel('Pre-training Iteration')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Function Pre-training')
        ax.grid(True, alpha=0.3)
        
        # 5. Improvement over baseline
        improvements = [e.improvement_percent for e in self.evaluation_results]
        
        ax = axes[2, 0]
        ax.plot(eval_iters, improvements, 'g-s', linewidth=2, markersize=6)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Improvement Over Baseline')
        ax.grid(True, alpha=0.3)
        
        # 6. Learning metrics
        clip_fractions = [m.clip_fraction for m in self.training_metrics]
        
        ax = axes[2, 1]
        ax.plot(iterations, clip_fractions, 'orange', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Clip Fraction')
        ax.set_title('PPO Clip Fraction (Policy Change Indicator)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        report_path = os.path.join(self.save_dir, 'training_report.png')
        plt.savefig(report_path, dpi=150)
        self._log(f"📈 Saved visualization: {report_path}")
        
        # Save detailed metrics
        metrics_data = {
            'configuration': self.config,
            'training_summary': {
                'total_iterations': self.iteration,
                'total_timesteps': self.total_timesteps,
                'training_time_minutes': (time.time() - self.start_time) / 60,
                'plateau_detected': self.check_plateau()
            },
            'baseline': {
                'mean': self.baseline_result.overall_catch_rate,
                'confidence_lower': self.baseline_result.confidence_95_lower,
                'confidence_upper': self.baseline_result.confidence_95_upper
            },
            'final_performance': {
                'mean': self.evaluation_results[-1].mean_performance,
                'confidence_lower': self.evaluation_results[-1].confidence_lower,
                'confidence_upper': self.evaluation_results[-1].confidence_upper,
                'improvement_percent': self.evaluation_results[-1].improvement_percent,
                'is_significant': self.evaluation_results[-1].is_significant
            },
            'best_performance': {
                'value': self.best_performance,
                'iteration': max((e.iteration for e in self.evaluation_results if e.is_best), default=0)
            },
            'training_metrics': [asdict(m) for m in self.training_metrics[-1000:]],  # Last 1000
            'evaluation_results': [asdict(e) for e in self.evaluation_results],
            'value_pretrain_losses': self.value_pretrain_losses
        }
        
        metrics_path = os.path.join(self.save_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        self._log(f"💾 Saved metrics: {metrics_path}")
        
        # Close log file
        self.log_file.close()
        
        return metrics_data


def main():
    """Run production PPO training"""
    print("🚀 PRODUCTION PPO TRAINING")
    print("=" * 80)
    print("Using optimal configuration from extensive experiments:")
    print("- Episode length: 5000 steps")
    print("- Value pre-training: 20 iterations")
    print("- Learning rate: 3e-5")
    print("- Plateau detection for automatic stopping")
    print("=" * 80)
    
    # Create trainer
    trainer = ProductionPPOTrainer(
        episode_length=5000,
        value_pretrain_iterations=20,
        learning_rate=3e-5,
        eval_frequency=10,
        checkpoint_frequency=10,
        plateau_patience=30,
        plateau_threshold=0.002
    )
    
    # Train until plateau
    trainer.train_until_plateau(max_iterations=300)
    
    # Generate report
    report = trainer.generate_report()
    
    print(f"\n✅ Training complete!")
    print(f"Results saved to: {trainer.save_dir}")
    print(f"Best performance: {trainer.best_performance:.4f}")
    print(f"Improvement: {report['final_performance']['improvement_percent']:+.1f}%")
    
    return trainer, report


if __name__ == "__main__":
    trainer, report = main()