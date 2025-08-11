"""
Quick Validation Script - Fast RL vs SL Proof of Concept

This script runs a minimal set of experiments to quickly validate that our
RL system can improve upon the SL baseline. Use this for rapid validation
before committing to longer experiment suites.

Features:
- Runs in 30-60 minutes
- 3 critical experiments with statistical validation
- Clear pass/fail results
- Actionable recommendations

Usage:
    python experiments/quick_validation.py
"""

import sys
import os
import time
from datetime import datetime

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.experimental_framework import ExperimentRunner, ExperimentConfig


def create_quick_experiments():
    """Create minimal experiments for rapid validation"""
    return [
        ExperimentConfig(
            name="quick_basic_improvement",
            description="Basic test that RL improves over SL baseline",
            hypothesis="PPO RL training improves catch rate over SL baseline",
            num_trials=5,
            num_iterations=10,
            rollout_steps=512,
            eval_episodes=20
        ),
        
        ExperimentConfig(
            name="quick_reproducibility",
            description="Test that RL improvement is reproducible",
            hypothesis="RL improvement is consistent across multiple independent runs",
            num_trials=6,
            num_iterations=12,
            rollout_steps=512,
            eval_episodes=20
        ),
        
        ExperimentConfig(
            name="quick_significance",
            description="Test that RL improvement is statistically significant",
            hypothesis="RL improvement has statistical significance (p < 0.05)",
            num_trials=8,
            num_iterations=15,
            rollout_steps=768,
            eval_episodes=25
        )
    ]


def run_quick_validation():
    """Run quick validation experiments"""
    
    print(f"🚀 Quick RL Validation")
    print(f"{'='*50}")
    print(f"  Purpose: Rapid proof that RL improves over SL")
    print(f"  Duration: ~30-60 minutes")
    print(f"  Experiments: 3 critical tests")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")
    
    # Check prerequisites
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: SL checkpoint not found: {checkpoint_path}")
        print(f"   Please train the supervised learning model first using transformer_training.ipynb")
        return False
    
    start_time = time.time()
    
    try:
        # Initialize runner
        runner = ExperimentRunner(
            sl_checkpoint_path=checkpoint_path,
            results_dir="experiments/quick_results",
            device='auto'
        )
        
        # Get quick experiments
        experiments = create_quick_experiments()
        
        # Run experiments
        results = []
        successful_experiments = 0
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n🧪 Running Experiment {i}/3: {experiment.name}")
            print(f"   Hypothesis: {experiment.hypothesis}")
            
            exp_start = time.time()
            result = runner.run_experiment(experiment)
            exp_time = time.time() - exp_start
            
            results.append(result)
            
            if result.hypothesis_confirmed:
                successful_experiments += 1
                print(f"   ✅ CONFIRMED in {exp_time/60:.1f}m")
            else:
                print(f"   ❌ REJECTED in {exp_time/60:.1f}m")
            
            # Show key metrics
            stats = result.aggregate_stats
            print(f"   📊 Improvement: {stats['mean_improvement']:+.3f} ± {stats['std_improvement']:.3f}")
            if 't_test_p_value' in result.statistical_tests:
                print(f"   📈 p-value: {result.statistical_tests['t_test_p_value']:.6f}")
        
        # Final assessment
        total_time = time.time() - start_time
        success_rate = successful_experiments / len(experiments)
        
        print(f"\n{'='*50}")
        print(f"🎯 QUICK VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"  Experiments: {len(experiments)}")
        print(f"  Confirmed: {successful_experiments}")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Total Time: {total_time/60:.1f} minutes")
        print(f"{'='*50}")
        
        # Interpretation and recommendations
        if success_rate >= 0.67:  # At least 2/3 experiments successful
            print(f"🎉 VALIDATION SUCCESSFUL!")
            print(f"   ✅ RL system demonstrates improvement over SL baseline")
            print(f"   ✅ Improvement is reproducible and statistically significant")
            print(f"   🚀 Recommended: Run full experiment suite for comprehensive validation")
            
            # Show aggregate improvement
            all_improvements = []
            for result in results:
                if result.hypothesis_confirmed:
                    all_improvements.append(result.aggregate_stats['mean_improvement'])
            
            if all_improvements:
                avg_improvement = sum(all_improvements) / len(all_improvements)
                print(f"   📊 Average improvement: {avg_improvement:+.3f} catch rate")
            
            return True
            
        elif success_rate >= 0.33:  # At least 1/3 experiments successful
            print(f"⚠️  VALIDATION INCONCLUSIVE")
            print(f"   ⚡ Some improvement detected but not consistent")
            print(f"   🔧 Recommended actions:")
            print(f"      - Check hyperparameters (learning rate, rollout size)")
            print(f"      - Try longer training (more iterations)")
            print(f"      - Verify reward function design")
            print(f"      - Run sensitivity experiments")
            
            return False
            
        else:
            print(f"❌ VALIDATION FAILED")
            print(f"   💥 No consistent RL improvement detected")
            print(f"   🔧 Recommended debugging:")
            print(f"      - Verify PPO implementation")
            print(f"      - Check reward function")
            print(f"      - Test with different hyperparameters")
            print(f"      - Analyze training curves for convergence issues")
            
            return False
    
    except Exception as e:
        print(f"\n💥 Quick validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    success = run_quick_validation()
    
    print(f"\n{'='*50}")
    if success:
        print(f"✅ Quick validation completed successfully!")
        print(f"🚀 Next steps:")
        print(f"   - Run critical path experiments: python run_experiments.py --suite critical")
        print(f"   - Run full validation suite: python run_experiments.py --suite full")
    else:
        print(f"❌ Quick validation needs attention")
        print(f"🔧 Debug before running larger experiment suites")
    print(f"{'='*50}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())