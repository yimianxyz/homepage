#!/usr/bin/env python3
"""
Extended RL Training - Run comprehensive RL vs baseline comparison with optimal parameters

This script runs extended RL training with better hyperparameters and longer episodes
based on the episode length analysis.
"""

import sys
import os
import argparse
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.evaluation.performance_evaluator import PerformanceEvaluator
from rl.models import TransformerModel, TransformerModelLoader
from rl.training import PPOTrainer, TrainingConfig
from rl.utils import set_seed
from stable_baselines3 import PPO


class ExtendedTrainingEvaluator:
    """Extended training with optimal parameters"""
    
    def __init__(self, output_dir: str = None, episode_length: int = 3000):
        """Initialize with optimal parameters"""
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="rl_extended_")
        self.episode_length = episode_length
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🚀 Extended Training Evaluator")
        print(f"Output directory: {self.output_dir}")
        print(f"Episode length: {self.episode_length}")
    
    def create_consistent_baseline(self, env_config: dict) -> tuple:
        """Create a consistent baseline model with proper architecture"""
        
        print(f"🔧 Creating consistent baseline model...")
        
        # Try to load SL model first
        sl_model_path = "/home/iotcat/homepage2/best_model.pt"
        
        if os.path.exists(sl_model_path):
            print(f"📚 Attempting to load SL baseline from {sl_model_path}")
            try:
                loader = TransformerModelLoader()
                sl_model = loader.load_model(sl_model_path)
                
                # Wrap in PPO for consistent interface
                # Create PPO with same architecture
                config = TrainingConfig(
                    num_boids=env_config['num_boids'],
                    canvas_width=env_config['canvas_width'],
                    canvas_height=env_config['canvas_height'],
                    max_episode_steps=self.episode_length,
                    total_timesteps=1000,  # Minimal training
                    n_steps=128,
                    batch_size=32,
                    n_epochs=2,
                    n_envs=1,
                    log_dir=os.path.join(self.output_dir, "sl_baseline_logs"),
                    save_dir=os.path.join(self.output_dir, "sl_baseline_models"),
                    tensorboard_log=None
                )
                
                trainer = PPOTrainer(config)
                trainer.transformer_model = sl_model
                trainer.setup_environment()
                trainer.setup_ppo()
                
                # Minimal training to create PPO wrapper
                print("Wrapping SL model in PPO interface...")
                trainer.ppo_agent.learn(total_timesteps=100, progress_bar=False)
                
                return trainer.ppo_agent, "SL_Baseline", sl_model.get_architecture_info()
                
            except Exception as e:
                print(f"⚠️  Failed to load SL model: {e}")
        
        # Create trained baseline with consistent architecture
        print("🔧 Creating trained transformer baseline...")
        
        # Use smaller model for consistency
        baseline_model = TransformerModel(
            d_model=64,
            n_heads=8, 
            n_layers=4,
            ffn_hidden=256,
            max_boids=env_config['num_boids']
        )
        
        # Train baseline with short training
        config = TrainingConfig(
            num_boids=env_config['num_boids'],
            canvas_width=env_config['canvas_width'],
            canvas_height=env_config['canvas_height'],
            max_episode_steps=self.episode_length,
            total_timesteps=5000,  # Short baseline training
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            n_envs=1,
            learning_rate=3e-4,
            log_dir=os.path.join(self.output_dir, "baseline_logs"),
            save_dir=os.path.join(self.output_dir, "baseline_models"),
            tensorboard_log=None
        )
        
        trainer = PPOTrainer(config)
        trainer.transformer_model = baseline_model
        trainer.setup_environment()
        trainer.setup_ppo()
        
        print("Training baseline model (5000 steps)...")
        trainer.ppo_agent.learn(total_timesteps=5000, progress_bar=True)
        
        return trainer.ppo_agent, "Trained_Baseline", baseline_model.get_architecture_info()
    
    def train_rl_model(self, env_config: dict, baseline_model, baseline_arch: dict) -> tuple:
        """Train RL model with extended training and optimal parameters"""
        
        print(f"🚀 Training RL model with extended parameters...")
        
        # Configure extended RL training
        config = TrainingConfig(
            num_boids=env_config['num_boids'],
            canvas_width=env_config['canvas_width'],
            canvas_height=env_config['canvas_height'],
            max_episode_steps=self.episode_length,
            
            # Extended training parameters
            total_timesteps=50_000,  # 10x longer training
            learning_rate=1e-4,      # Lower for fine-tuning
            n_steps=512,             # Longer rollouts
            batch_size=64,           # Larger batches
            n_epochs=10,             # More training per batch
            gamma=0.99,              # Standard discount
            gae_lambda=0.95,         # Standard GAE
            clip_range=0.2,          # Standard clipping
            
            # Environment and infrastructure
            n_envs=1,                # Keep single env for consistency
            
            # Monitoring
            save_freq=10000,
            eval_freq=5000,
            log_interval=1,
            
            # Paths
            log_dir=os.path.join(self.output_dir, "rl_logs"),
            save_dir=os.path.join(self.output_dir, "rl_models"),
            tensorboard_log=os.path.join(self.output_dir, "rl_tensorboard"),
            
            # Model configuration to match baseline
            model_config=baseline_arch
        )
        
        trainer = PPOTrainer(config)
        
        # Initialize from baseline if possible
        if hasattr(baseline_model, 'policy') and hasattr(baseline_model.policy, 'transformer'):
            print("🔄 Initializing RL training from baseline weights...")
            trainer.transformer_model = baseline_model.policy.transformer
        
        trainer.setup_environment()
        trainer.load_model()  # This will use our provided model or create new one
        trainer.setup_ppo()
        
        # Extended training with progress monitoring
        print(f"Training RL model ({config.total_timesteps:,} steps)...")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Rollout steps: {config.n_steps}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Training epochs: {config.n_epochs}")
        
        trainer.ppo_agent.learn(
            total_timesteps=config.total_timesteps,
            progress_bar=True
        )
        
        # Save trained model
        model_path = os.path.join(self.output_dir, "rl_final_model")
        trainer.ppo_agent.save(model_path)
        print(f"💾 RL model saved to: {model_path}")
        
        trainer.cleanup()
        return trainer.ppo_agent, "RL_Extended"
    
    def run_comprehensive_evaluation(self, baseline_model, baseline_name: str,
                                   rl_model, rl_name: str, env_config: dict) -> dict:
        """Run comprehensive evaluation with extended episodes"""
        
        print(f"\n🎯 Comprehensive Evaluation: {baseline_name} vs {rl_name}")
        print(f"Environment: {env_config['num_boids']} boids, {self.episode_length} max steps")
        
        # Create evaluator with more episodes for statistical power
        evaluator = PerformanceEvaluator(
            n_evaluation_episodes=50,  # Good statistical power
            confidence_level=0.95
        )
        
        # Update environment config for evaluation
        eval_env_config = env_config.copy()
        eval_env_config['max_steps'] = self.episode_length
        
        # Evaluate baseline
        print(f"\n📊 Evaluating {baseline_name}...")
        baseline_performance, baseline_episodes = evaluator.evaluate_model(
            model=baseline_model,
            model_name=baseline_name,
            env_config=eval_env_config,
            verbose=False
        )
        
        # Evaluate RL model
        print(f"\n📊 Evaluating {rl_name}...")
        rl_performance, rl_episodes = evaluator.evaluate_model(
            model=rl_model,
            model_name=rl_name,
            env_config=eval_env_config,
            verbose=False
        )
        
        # Statistical comparison
        comparison = evaluator.compare_models(
            baseline_performance, rl_performance,
            baseline_episodes, rl_episodes
        )
        
        # Generate detailed report
        report_path = os.path.join(self.output_dir, "extended_evaluation_report.md")
        report = evaluator.generate_report(
            baseline_performance, rl_performance, comparison, report_path
        )
        
        # Print key findings
        self._print_detailed_findings(comparison, baseline_performance, rl_performance)
        
        return {
            'baseline_performance': baseline_performance,
            'rl_performance': rl_performance,
            'comparison': comparison,
            'report_path': report_path,
            'evaluation_config': {
                'episode_length': self.episode_length,
                'n_episodes': 50,
                'env_config': env_config
            }
        }
    
    def _print_detailed_findings(self, comparison: dict, baseline_perf, rl_perf):
        """Print detailed findings with more context"""
        
        reward_analysis = comparison['reward_analysis']
        success_analysis = comparison['success_rate_analysis'] 
        interpretation = comparison['interpretation']
        
        print(f"\n" + "="*80)
        print(f"📊 DETAILED EVALUATION RESULTS")
        print(f"="*80)
        
        print(f"\n🎯 Primary Metrics:")
        print(f"  Baseline Reward:     {baseline_perf.mean_total_reward:.3f} ± {baseline_perf.std_total_reward:.3f}")
        print(f"  RL Model Reward:     {rl_perf.mean_total_reward:.3f} ± {rl_perf.std_total_reward:.3f}")
        print(f"  Absolute Difference: {reward_analysis['absolute_difference']:+.3f}")
        print(f"  Relative Improvement: {reward_analysis['relative_improvement']:+.1%}")
        
        print(f"\n🎪 Success Rates:")
        print(f"  Baseline Success:    {baseline_perf.mean_success_rate:.1%} ± {baseline_perf.std_success_rate:.1%}")
        print(f"  RL Model Success:    {rl_perf.mean_success_rate:.1%} ± {rl_perf.std_success_rate:.1%}")
        print(f"  Success Improvement: {success_analysis['relative_improvement']:+.1%}")
        
        print(f"\n📈 Efficiency Metrics:")
        print(f"  Baseline Reward/Step: {baseline_perf.mean_reward_per_step:.4f}")
        print(f"  RL Model Reward/Step: {rl_perf.mean_reward_per_step:.4f}")
        print(f"  Efficiency Improvement: {((rl_perf.mean_reward_per_step - baseline_perf.mean_reward_per_step) / baseline_perf.mean_reward_per_step):+.1%}")
        
        print(f"\n🔬 Statistical Analysis:")
        print(f"  P-value (t-test):     {reward_analysis['ttest_pvalue']:.6f}")
        print(f"  P-value (rank-sum):   {reward_analysis['mannwhitney_pvalue']:.6f}")
        print(f"  Effect Size (Cohen's d): {reward_analysis['effect_size_cohens_d']:+.3f}")
        print(f"  Effect Magnitude:     {interpretation['effect_magnitude']}")
        
        print(f"\n✅ Significance Tests:")
        print(f"  Statistically Significant: {reward_analysis['significant']}")
        print(f"  Practically Significant:   {reward_analysis['practically_significant']}")
        print(f"  CI Intervals Overlap:      {comparison['confidence_intervals_overlap']['reward']}")
        
        print(f"\n🏆 Final Assessment:")
        print(f"  Conclusion: {interpretation['overall_conclusion']}")
        print(f"  Recommendation: {interpretation['recommendation']}")
        
        # Additional behavioral insights
        print(f"\n🎭 Behavioral Analysis:")
        print(f"  Baseline Action Consistency: {baseline_perf.mean_action_consistency:.3f}")
        print(f"  RL Model Action Consistency: {rl_perf.mean_action_consistency:.3f}")
        print(f"  Baseline Failure Rate:       {baseline_perf.failure_rate:.1%}")
        print(f"  RL Model Failure Rate:       {rl_perf.failure_rate:.1%}")
        
        print(f"\n" + "="*80)


def main():
    """Main function for extended training evaluation"""
    
    parser = argparse.ArgumentParser(description="Run extended RL training evaluation")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--episode-length", type=int, default=3000, 
                       help="Episode length (default: 3000 based on analysis)")
    parser.add_argument("--env-scale", type=str, choices=['small', 'medium', 'large'], 
                       default='small', help="Environment scale")
    
    args = parser.parse_args()
    
    # Set deterministic seed
    set_seed(42)
    
    print("🚀 EXTENDED RL TRAINING EVALUATION")
    print("=" * 80)
    print("This evaluation runs extended RL training with optimal parameters:")
    print("- 50,000 timesteps (10x longer training)")
    print("- Lower learning rate (1e-4) for fine-tuning")
    print("- Longer episodes (3000 steps) based on analysis")
    print("- Larger rollouts and batches")
    print("")
    
    # Environment configurations
    env_configs = {
        'small': {
            'num_boids': 10,
            'canvas_width': 400,
            'canvas_height': 300
        },
        'medium': {
            'num_boids': 15,
            'canvas_width': 600,
            'canvas_height': 400
        },
        'large': {
            'num_boids': 20,
            'canvas_width': 800,
            'canvas_height': 600
        }
    }
    
    env_config = env_configs[args.env_scale]
    
    print(f"Environment Configuration:")
    print(f"  Scale: {args.env_scale}")
    print(f"  Boids: {env_config['num_boids']}")
    print(f"  Canvas: {env_config['canvas_width']}×{env_config['canvas_height']}")
    print(f"  Episode length: {args.episode_length}")
    
    # Create evaluator
    evaluator = ExtendedTrainingEvaluator(args.output_dir, args.episode_length)
    
    try:
        # Step 1: Create consistent baseline
        print(f"\n" + "="*60)
        print(f"STEP 1: BASELINE MODEL CREATION")
        print(f"="*60)
        
        baseline_model, baseline_name, baseline_arch = evaluator.create_consistent_baseline(env_config)
        
        # Step 2: Train RL model
        print(f"\n" + "="*60)
        print(f"STEP 2: EXTENDED RL TRAINING")
        print(f"="*60)
        
        rl_model, rl_name = evaluator.train_rl_model(env_config, baseline_model, baseline_arch)
        
        # Step 3: Comprehensive evaluation
        print(f"\n" + "="*60)
        print(f"STEP 3: COMPREHENSIVE EVALUATION")
        print(f"="*60)
        
        results = evaluator.run_comprehensive_evaluation(
            baseline_model, baseline_name,
            rl_model, rl_name,
            env_config
        )
        
        print(f"\n✅ Extended training evaluation completed successfully!")
        print(f"📁 All results saved to: {evaluator.output_dir}")
        print(f"📄 Detailed report: {results['report_path']}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Evaluation interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())