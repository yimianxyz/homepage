#!/usr/bin/env python3
"""
Performance Test Runner - Execute comprehensive RL vs SL evaluation

This script runs the rigorous performance evaluation to verify that RL training
produces meaningful improvements over the supervised learning baseline.
"""

import sys
import os
import argparse
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.evaluation.performance_evaluator import PerformanceEvaluator, create_evaluation_suite
from rl.models import TransformerModel, TransformerModelLoader
from rl.training import PPOTrainer, TrainingConfig
from rl.utils import set_seed
from stable_baselines3 import PPO


class RandomBaselineModel:
    """Random baseline model for comparison"""
    
    def __init__(self, action_space_shape=(2,)):
        self.action_space_shape = action_space_shape
        
    def predict(self, obs, deterministic=True):
        """Predict random actions"""
        import numpy as np
        action = np.random.uniform(-1, 1, size=self.action_space_shape).astype(np.float32)
        return action, None


class ModelEvaluationSuite:
    """Complete evaluation suite for comparing models"""
    
    def __init__(self, output_dir: str = None):
        """Initialize evaluation suite"""
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="rl_evaluation_")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create evaluator
        self.evaluator, self.env_configs = create_evaluation_suite()
        
        print(f"🔬 Model Evaluation Suite Initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Environment configurations: {len(self.env_configs)}")
    
    def load_or_train_baseline_model(self, env_config: dict) -> tuple:
        """Load SL baseline or create alternative baseline"""
        
        # Try to load best_model.pt (SL baseline)
        sl_model_path = "/home/iotcat/homepage2/best_model.pt"
        
        if os.path.exists(sl_model_path):
            print(f"📚 Loading SL baseline from {sl_model_path}")
            try:
                loader = TransformerModelLoader()
                sl_model = loader.load_model(sl_model_path)
                return sl_model, "SL_Baseline"
            except Exception as e:
                print(f"⚠️  Failed to load SL model: {e}")
                print("Falling back to trained transformer baseline...")
        
        # Create and minimally train a transformer model as baseline
        print("🔧 Creating trained transformer baseline...")
        baseline_model = TransformerModel(
            d_model=32, 
            n_heads=4, 
            n_layers=2, 
            max_boids=env_config['num_boids']
        )
        
        # Quick training to make it better than random
        config = TrainingConfig(
            num_boids=env_config['num_boids'],
            canvas_width=env_config['canvas_width'],
            canvas_height=env_config['canvas_height'],
            max_episode_steps=env_config['max_steps'],
            total_timesteps=1000,  # Very short training
            n_steps=64,
            batch_size=16,
            n_epochs=2,
            n_envs=1,
            log_dir=os.path.join(self.output_dir, "baseline_logs"),
            save_dir=os.path.join(self.output_dir, "baseline_models"),
            tensorboard_log=None
        )
        
        trainer = PPOTrainer(config)
        trainer.transformer_model = baseline_model
        trainer.setup_environment()
        trainer.setup_ppo()
        
        print("Training baseline model (1000 steps)...")
        trainer.ppo_agent.learn(total_timesteps=1000, progress_bar=False)
        
        return trainer.ppo_agent, "Trained_Baseline"
    
    def train_rl_model(self, env_config: dict, baseline_model=None) -> tuple:
        """Train RL model for comparison"""
        
        print(f"🚀 Training RL model...")
        
        # Configure RL training
        config = TrainingConfig(
            num_boids=env_config['num_boids'],
            canvas_width=env_config['canvas_width'],
            canvas_height=env_config['canvas_height'],
            max_episode_steps=env_config['max_steps'],
            total_timesteps=5000,  # Moderate training for clear improvement
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            n_envs=1,
            learning_rate=3e-4,
            log_dir=os.path.join(self.output_dir, "rl_logs"),
            save_dir=os.path.join(self.output_dir, "rl_models"),
            tensorboard_log=os.path.join(self.output_dir, "rl_tensorboard")
        )
        
        trainer = PPOTrainer(config)
        
        # Use baseline model as starting point if available
        if baseline_model and hasattr(baseline_model, 'policy'):
            print("🔄 Initializing RL training from baseline...")
            trainer.transformer_model = baseline_model.policy.transformer
        
        trainer.setup_environment()
        trainer.load_model()  # This will create new model if needed
        trainer.setup_ppo()
        
        # Train with progress monitoring
        print(f"Training RL model ({config.total_timesteps} steps)...")
        trainer.ppo_agent.learn(
            total_timesteps=config.total_timesteps,
            progress_bar=True
        )
        
        # Save trained model
        model_path = os.path.join(self.output_dir, "rl_final_model")
        trainer.ppo_agent.save(model_path)
        
        trainer.cleanup()
        return trainer.ppo_agent, "RL_Trained"
    
    def run_comprehensive_evaluation(self, 
                                   baseline_model, baseline_name: str,
                                   rl_model, rl_name: str,
                                   env_config: dict) -> dict:
        """Run comprehensive evaluation comparing baseline vs RL"""
        
        print(f"\n🎯 Comprehensive Evaluation: {baseline_name} vs {rl_name}")
        print(f"Environment: {env_config['name']} "
              f"({env_config['num_boids']} boids, {env_config['max_steps']} max steps)")
        
        # Evaluate baseline
        baseline_performance, baseline_episodes = self.evaluator.evaluate_model(
            model=baseline_model,
            model_name=baseline_name,
            env_config=env_config,
            verbose=False
        )
        
        # Evaluate RL model
        rl_performance, rl_episodes = self.evaluator.evaluate_model(
            model=rl_model,
            model_name=rl_name,
            env_config=env_config,
            verbose=False
        )
        
        # Statistical comparison
        comparison = self.evaluator.compare_models(
            baseline_performance, rl_performance,
            baseline_episodes, rl_episodes
        )
        
        # Generate detailed report
        report_path = os.path.join(self.output_dir, f"evaluation_report_{env_config['name']}.md")
        report = self.evaluator.generate_report(
            baseline_performance, rl_performance, comparison, report_path
        )
        
        # Print key findings
        self._print_key_findings(comparison)
        
        return {
            'baseline_performance': baseline_performance,
            'rl_performance': rl_performance,
            'comparison': comparison,
            'report_path': report_path,
            'env_config': env_config
        }
    
    def _print_key_findings(self, comparison: dict):
        """Print key findings in a clear format"""
        
        reward_analysis = comparison['reward_analysis']
        interpretation = comparison['interpretation']
        
        print(f"\n📊 KEY FINDINGS:")
        print(f"  Reward Improvement: {reward_analysis['relative_improvement']:.1%}")
        print(f"  Statistical Significance: {reward_analysis['significant']}")
        print(f"  Effect Size: {reward_analysis['effect_size_cohens_d']:.3f} ({interpretation['effect_magnitude']})")
        print(f"  P-value: {reward_analysis['ttest_pvalue']:.6f}")
        print(f"  Conclusion: {interpretation['overall_conclusion']}")
        print(f"  Recommendation: {interpretation['recommendation']}")
    
    def run_multi_environment_evaluation(self):
        """Run evaluation across multiple environment configurations"""
        
        print(f"\n🌍 Multi-Environment Evaluation")
        print(f"Testing across {len(self.env_configs)} different configurations")
        
        all_results = []
        
        for i, env_config in enumerate(self.env_configs):
            print(f"\n{'='*60}")
            print(f"ENVIRONMENT {i+1}/{len(self.env_configs)}: {env_config['name'].upper()}")
            print(f"{'='*60}")
            
            try:
                # Load/create baseline
                baseline_model, baseline_name = self.load_or_train_baseline_model(env_config)
                
                # Train RL model
                rl_model, rl_name = self.train_rl_model(env_config, baseline_model)
                
                # Evaluate
                results = self.run_comprehensive_evaluation(
                    baseline_model, baseline_name,
                    rl_model, rl_name,
                    env_config
                )
                
                all_results.append(results)
                
            except Exception as e:
                print(f"❌ Error in environment {env_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _generate_summary_report(self, all_results: list):
        """Generate summary report across all environments"""
        
        if not all_results:
            print("❌ No successful evaluations to summarize")
            return
        
        print(f"\n📋 MULTI-ENVIRONMENT SUMMARY")
        print(f"{'='*60}")
        
        summary_path = os.path.join(self.output_dir, "multi_environment_summary.md")
        
        with open(summary_path, 'w') as f:
            f.write("# Multi-Environment RL Evaluation Summary\n\n")
            
            # Overall statistics
            total_environments = len(all_results)
            significant_improvements = sum(1 for r in all_results 
                                         if r['comparison']['reward_analysis']['significant'])
            
            f.write(f"## Overview\n")
            f.write(f"- **Environments Tested**: {total_environments}\n")
            f.write(f"- **Statistically Significant Improvements**: {significant_improvements}/{total_environments}\n")
            f.write(f"- **Success Rate**: {significant_improvements/total_environments:.1%}\n\n")
            
            # Environment-by-environment results
            f.write(f"## Results by Environment\n\n")
            
            for result in all_results:
                env_name = result['env_config']['name']
                comparison = result['comparison']
                reward_analysis = comparison['reward_analysis']
                
                f.write(f"### {env_name.title()} Environment\n")
                f.write(f"- **Configuration**: {result['env_config']['num_boids']} boids, "
                       f"{result['env_config']['canvas_width']}×{result['env_config']['canvas_height']}\n")
                f.write(f"- **Improvement**: {reward_analysis['relative_improvement']:.1%}\n")
                f.write(f"- **Significant**: {reward_analysis['significant']}\n")
                f.write(f"- **Effect Size**: {reward_analysis['effect_size_cohens_d']:.3f}\n")
                f.write(f"- **Conclusion**: {comparison['interpretation']['overall_conclusion']}\n\n")
            
            # Meta-analysis
            improvements = [r['comparison']['reward_analysis']['relative_improvement'] 
                          for r in all_results]
            effect_sizes = [r['comparison']['reward_analysis']['effect_size_cohens_d'] 
                           for r in all_results]
            
            f.write(f"## Meta-Analysis\n")
            f.write(f"- **Mean Improvement**: {np.mean(improvements):.1%}\n")
            f.write(f"- **Improvement Range**: [{min(improvements):.1%}, {max(improvements):.1%}]\n")
            f.write(f"- **Mean Effect Size**: {np.mean(effect_sizes):.3f}\n")
            f.write(f"- **Consistent Improvement**: {significant_improvements >= total_environments * 0.7}\n\n")
            
            # Final recommendation
            if significant_improvements >= total_environments * 0.7:
                recommendation = "**STRONG EVIDENCE**: RL training consistently improves performance."
            elif significant_improvements >= total_environments * 0.5:
                recommendation = "**MODERATE EVIDENCE**: RL training shows improvement in most cases."
            else:
                recommendation = "**WEAK EVIDENCE**: RL training benefits are inconsistent or minimal."
            
            f.write(f"## Final Recommendation\n{recommendation}\n")
        
        print(f"📄 Summary report saved to: {summary_path}")
        
        # Print key summary
        print(f"  Significant improvements: {significant_improvements}/{total_environments}")
        print(f"  Mean improvement: {np.mean(improvements):.1%}")
        print(f"  Mean effect size: {np.mean(effect_sizes):.3f}")
        

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Run comprehensive RL vs SL evaluation")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation (fewer episodes)")
    parser.add_argument("--single-env", type=str, choices=['standard', 'small_scale', 'large_scale'],
                       help="Run evaluation on single environment only")
    
    args = parser.parse_args()
    
    # Set deterministic seed
    set_seed(42)
    
    print("🔬 RL PERFORMANCE VERIFICATION")
    print("=" * 60)
    print("This evaluation will rigorously test whether RL training")
    print("produces meaningful improvements over baseline performance.")
    print("")
    
    # Create evaluation suite
    suite = ModelEvaluationSuite(args.output_dir)
    
    if args.quick:
        # Override evaluator for quick testing
        suite.evaluator = PerformanceEvaluator(n_evaluation_episodes=20)
        print("⚡ Quick mode: Using 20 episodes per evaluation")
    
    try:
        if args.single_env:
            # Single environment evaluation
            env_config = next(cfg for cfg in suite.env_configs if cfg['name'] == args.single_env)
            print(f"🎯 Single environment evaluation: {env_config['name']}")
            
            baseline_model, baseline_name = suite.load_or_train_baseline_model(env_config)
            rl_model, rl_name = suite.train_rl_model(env_config, baseline_model)
            
            results = suite.run_comprehensive_evaluation(
                baseline_model, baseline_name,
                rl_model, rl_name,
                env_config
            )
            
            print(f"\n✅ Single environment evaluation completed")
            print(f"Report available at: {results['report_path']}")
            
        else:
            # Multi-environment evaluation
            all_results = suite.run_multi_environment_evaluation()
            
            print(f"\n✅ Multi-environment evaluation completed")
            print(f"All reports available in: {suite.output_dir}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Evaluation interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n🎉 Evaluation complete! Check results in: {suite.output_dir}")
    return 0


if __name__ == "__main__":
    import numpy as np  # For summary calculations
    exit(main())