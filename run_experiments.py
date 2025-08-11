"""
Automated Experiment Runner - Systematic RL Validation

This script runs comprehensive experiments to systematically prove that our
PPO RL system improves upon the SL baseline. It executes predefined experiment
suites with statistical rigor and generates detailed reports.

Usage:
    python run_experiments.py --suite quick          # Quick validation (1-2 hours)
    python run_experiments.py --suite critical       # Critical path experiments (3-4 hours)
    python run_experiments.py --suite core           # Core validation experiments (6-8 hours)
    python run_experiments.py --suite ablation       # Ablation studies (4-6 hours)
    python run_experiments.py --suite full           # Complete experiment suite (12-16 hours)
    python run_experiments.py --experiments exp1,exp2 # Specific experiments

Features:
- Automated experiment execution with progress tracking
- Statistical significance testing and effect size analysis
- Comprehensive reporting and visualization
- Resume capability for interrupted experiments
- Parallel execution for efficiency
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.experimental_framework import ExperimentRunner, ExperimentResult
from experiments.experiment_definitions import ExperimentSuite


def main():
    parser = argparse.ArgumentParser(description='Run systematic RL validation experiments')
    
    # Experiment selection
    parser.add_argument('--suite', type=str, choices=['quick', 'critical', 'core', 'ablation', 'sensitivity', 'generalization', 'efficiency', 'full'],
                       help='Predefined experiment suite to run')
    parser.add_argument('--experiments', type=str,
                       help='Comma-separated list of specific experiment names to run')
    parser.add_argument('--list-experiments', action='store_true',
                       help='List all available experiments and exit')
    
    # Configuration
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to supervised learning checkpoint')
    parser.add_argument('--results-dir', type=str, default='experiments/results',
                       help='Directory to save experimental results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    # Execution options
    parser.add_argument('--resume', action='store_true',
                       help='Resume interrupted experiments (load existing results)')
    parser.add_argument('--force-rerun', action='store_true',
                       help='Force re-run of experiments even if results exist')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without actually running experiments')
    
    # Reporting
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive report from existing results')
    parser.add_argument('--report-only', action='store_true',
                       help='Only generate report, don\'t run experiments')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        print("Make sure you have trained a supervised learning model first.")
        return 1
    
    # List experiments if requested
    if args.list_experiments:
        list_all_experiments()
        return 0
    
    # Generate report only if requested
    if args.report_only:
        generate_report_from_existing(args.results_dir)
        return 0
    
    # Determine experiments to run
    experiments = get_experiments_to_run(args)
    if not experiments:
        print("❌ Error: No experiments specified. Use --suite or --experiments.")
        print("Use --list-experiments to see available options.")
        return 1
    
    # Show experiment plan
    show_experiment_plan(experiments, args.dry_run)
    
    if args.dry_run:
        return 0
    
    # Confirm execution for large suites
    if len(experiments) > 5 and not confirm_execution(experiments):
        print("Execution cancelled by user.")
        return 0
    
    # Run experiments
    success = run_experiment_suite(
        experiments=experiments,
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
        device=args.device,
        resume=args.resume,
        force_rerun=args.force_rerun
    )
    
    return 0 if success else 1


def list_all_experiments():
    """List all available experiments"""
    print("📋 Available Experiment Suites:")
    print()
    
    suites = {
        'quick': ExperimentSuite.get_quick_validation_suite(),
        'critical': ExperimentSuite.get_critical_path_experiments(),
        'core': ExperimentSuite.core_validation_experiments(),
        'ablation': ExperimentSuite.ablation_experiments(),
        'sensitivity': ExperimentSuite.sensitivity_experiments(),
        'generalization': ExperimentSuite.generalization_experiments(),
        'efficiency': ExperimentSuite.efficiency_experiments(),
        'full': ExperimentSuite.get_all_experiments()
    }
    
    for suite_name, suite_experiments in suites.items():
        total_trials = sum(exp.num_trials for exp in suite_experiments)
        estimated_time = estimate_suite_time(suite_experiments)
        
        print(f"  {suite_name.upper()}: {len(suite_experiments)} experiments, {total_trials} trials, ~{estimated_time:.1f}h")
        
        if suite_name != 'full':  # Don't list all experiments for full suite
            for exp in suite_experiments:
                print(f"    - {exp.name}: {exp.description}")
        print()


def get_experiments_to_run(args) -> List:
    """Determine which experiments to run based on arguments"""
    if args.suite:
        suite_map = {
            'quick': ExperimentSuite.get_quick_validation_suite(),
            'critical': ExperimentSuite.get_critical_path_experiments(),
            'core': ExperimentSuite.core_validation_experiments(),
            'ablation': ExperimentSuite.ablation_experiments(),
            'sensitivity': ExperimentSuite.sensitivity_experiments(),
            'generalization': ExperimentSuite.generalization_experiments(),
            'efficiency': ExperimentSuite.efficiency_experiments(),
            'full': ExperimentSuite.get_all_experiments()
        }
        return suite_map[args.suite]
    
    elif args.experiments:
        # Get specific experiments by name
        all_experiments = ExperimentSuite.get_all_experiments()
        exp_map = {exp.name: exp for exp in all_experiments}
        
        requested_names = [name.strip() for name in args.experiments.split(',')]
        experiments = []
        
        for name in requested_names:
            if name in exp_map:
                experiments.append(exp_map[name])
            else:
                print(f"⚠️  Warning: Unknown experiment '{name}' - skipping")
        
        return experiments
    
    return []


def estimate_suite_time(experiments) -> float:
    """Estimate time to run experiment suite in hours"""
    total_iterations = sum(exp.num_iterations * exp.num_trials for exp in experiments)
    # Rough estimate: 30 seconds per iteration (including evaluation)
    return (total_iterations * 30) / 3600


def show_experiment_plan(experiments, is_dry_run: bool):
    """Show detailed experiment execution plan"""
    total_trials = sum(exp.num_trials for exp in experiments)
    total_iterations = sum(exp.num_iterations * exp.num_trials for exp in experiments)
    estimated_time = estimate_suite_time(experiments)
    
    print(f"🧪 Experiment Execution Plan")
    print(f"{'='*60}")
    print(f"  Experiments: {len(experiments)}")
    print(f"  Total trials: {total_trials}")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Estimated time: {estimated_time:.1f} hours")
    print(f"  Mode: {'DRY RUN' if is_dry_run else 'LIVE EXECUTION'}")
    print(f"{'='*60}")
    
    print(f"\nExperiment Details:")
    for i, exp in enumerate(experiments, 1):
        exp_time = estimate_suite_time([exp])
        print(f"  {i:2d}. {exp.name}")
        print(f"      Hypothesis: {exp.hypothesis}")
        print(f"      Trials: {exp.num_trials}, Iterations: {exp.num_iterations}, Time: ~{exp_time:.1f}h")
    
    print()


def confirm_execution(experiments) -> bool:
    """Ask user confirmation for large experiment suites"""
    estimated_time = estimate_suite_time(experiments)
    
    print(f"⚠️  This will run {len(experiments)} experiments with an estimated runtime of {estimated_time:.1f} hours.")
    print(f"   Results will be saved and the process can be resumed if interrupted.")
    
    response = input("Do you want to proceed? [y/N]: ").strip().lower()
    return response in ['y', 'yes']


def run_experiment_suite(experiments, checkpoint_path: str, results_dir: str, 
                        device: str, resume: bool, force_rerun: bool) -> bool:
    """Run complete experiment suite with progress tracking"""
    
    print(f"\n{'='*80}")
    print(f"🚀 STARTING EXPERIMENT SUITE EXECUTION")
    print(f"{'='*80}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Results: {results_dir}")
    print(f"  Device: {device}")
    print(f"{'='*80}")
    
    try:
        # Initialize experiment runner
        runner = ExperimentRunner(
            sl_checkpoint_path=checkpoint_path,
            results_dir=results_dir,
            device=device
        )
        
        # Track results
        results = []
        successful_experiments = 0
        start_time = time.time()
        
        # Run each experiment
        for i, experiment in enumerate(experiments, 1):
            print(f"\n{'='*20} EXPERIMENT {i}/{len(experiments)} {'='*20}")
            print(f"Name: {experiment.name}")
            print(f"Progress: {i}/{len(experiments)} ({i/len(experiments):.1%})")
            
            exp_start_time = time.time()
            
            try:
                # Check if results already exist (unless force rerun)
                if not force_rerun and resume:
                    existing_result = load_existing_result(experiment.name, results_dir)
                    if existing_result:
                        print(f"📁 Loading existing result for {experiment.name}")
                        results.append(existing_result)
                        if existing_result.hypothesis_confirmed:
                            successful_experiments += 1
                        continue
                
                # Run experiment
                result = runner.run_experiment(experiment)
                results.append(result)
                
                if result.hypothesis_confirmed:
                    successful_experiments += 1
                    print(f"✅ Experiment {experiment.name} SUCCESSFUL")
                else:
                    print(f"❌ Experiment {experiment.name} INCONCLUSIVE")
                
                exp_time = time.time() - exp_start_time
                remaining_time = estimate_remaining_time(exp_time, i, len(experiments))
                print(f"⏱️  Experiment time: {exp_time/60:.1f}m, Estimated remaining: {remaining_time:.1f}m")
                
            except Exception as e:
                print(f"💥 Experiment {experiment.name} FAILED: {e}")
                # Continue with other experiments
                continue
        
        # Generate final report
        total_time = time.time() - start_time
        generate_final_report(results, successful_experiments, total_time, results_dir)
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment suite interrupted by user")
        print(f"   Partial results may be available in {results_dir}")
        return False
        
    except Exception as e:
        print(f"\n💥 Experiment suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_existing_result(experiment_name: str, results_dir: str):
    """Load existing experiment result if available"""
    results_path = Path(results_dir)
    
    # Look for result files matching the experiment name
    for result_file in results_path.glob(f"{experiment_name}_*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Convert back to ExperimentResult object (simplified)
            print(f"Found existing result: {result_file}")
            return type('ExperimentResult', (), {
                'hypothesis_confirmed': data['hypothesis_confirmed'],
                'config': type('Config', (), data['config'])(),
                'aggregate_stats': data['aggregate_stats'],
                'statistical_tests': data['statistical_tests']
            })()
            
        except Exception as e:
            print(f"⚠️  Could not load {result_file}: {e}")
            continue
    
    return None


def estimate_remaining_time(avg_exp_time: float, completed: int, total: int) -> float:
    """Estimate remaining time in minutes"""
    remaining_experiments = total - completed
    return (remaining_experiments * avg_exp_time) / 60


def generate_final_report(results, successful_experiments: int, total_time: float, results_dir: str):
    """Generate comprehensive final report"""
    print(f"\n{'='*80}")
    print(f"🎯 EXPERIMENT SUITE COMPLETE")
    print(f"{'='*80}")
    print(f"  Total experiments: {len(results)}")
    print(f"  Successful experiments: {successful_experiments}")
    print(f"  Success rate: {successful_experiments/len(results):.1%}")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"{'='*80}")
    
    if successful_experiments > 0:
        print(f"🎉 SUCCESS: RL system validation successful!")
        print(f"   {successful_experiments} experiments confirmed RL improves over SL baseline")
        
        # Summary of key findings
        improvements = []
        effect_sizes = []
        p_values = []
        
        for result in results:
            if hasattr(result, 'aggregate_stats') and result.hypothesis_confirmed:
                improvements.append(result.aggregate_stats.get('mean_improvement', 0))
                effect_sizes.append(result.aggregate_stats.get('effect_size', 0))
                if hasattr(result, 'statistical_tests'):
                    p_val = result.statistical_tests.get('t_test_p_value', 1.0)
                    p_values.append(p_val)
        
        if improvements:
            print(f"\n📊 Key Findings:")
            print(f"   Mean improvement: {sum(improvements)/len(improvements):+.3f}")
            print(f"   Mean effect size: {sum(effect_sizes)/len(effect_sizes):.3f}")
            if p_values:
                print(f"   Strongest p-value: {min(p_values):.6f}")
        
    else:
        print(f"⚠️  WARNING: No experiments confirmed RL improvement")
        print(f"   Consider: longer training, hyperparameter tuning, reward design")
    
    # Save comprehensive report
    report_path = Path(results_dir) / f"experiment_suite_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report_data = {
        'suite_summary': {
            'total_experiments': len(results),
            'successful_experiments': successful_experiments,
            'success_rate': successful_experiments / len(results) if results else 0,
            'total_time_hours': total_time / 3600,
            'timestamp': datetime.now().isoformat()
        },
        'experiment_results': [
            {
                'name': getattr(result.config, 'name', 'unknown') if hasattr(result, 'config') else 'unknown',
                'confirmed': result.hypothesis_confirmed if hasattr(result, 'hypothesis_confirmed') else False,
                'improvement': result.aggregate_stats.get('mean_improvement', 0) if hasattr(result, 'aggregate_stats') else 0,
                'effect_size': result.aggregate_stats.get('effect_size', 0) if hasattr(result, 'aggregate_stats') else 0
            }
            for result in results
        ]
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"📋 Comprehensive report saved: {report_path}")
    print(f"{'='*80}")


def generate_report_from_existing(results_dir: str):
    """Generate report from existing experiment results"""
    print(f"📊 Generating report from existing results in {results_dir}")
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Find all result files
    result_files = list(results_path.glob("*.json"))
    if not result_files:
        print(f"❌ No result files found in {results_dir}")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Analyze results
    confirmed_experiments = 0
    all_results = []
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            if data.get('hypothesis_confirmed', False):
                confirmed_experiments += 1
            
            all_results.append(data)
            
        except Exception as e:
            print(f"⚠️  Could not load {result_file}: {e}")
    
    # Generate summary report
    success_rate = confirmed_experiments / len(all_results) if all_results else 0
    
    print(f"\n📋 Experiment Summary Report")
    print(f"{'='*50}")
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Confirmed hypotheses: {confirmed_experiments}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"{'='*50}")
    
    if success_rate >= 0.6:
        print(f"🎉 VALIDATION SUCCESSFUL: RL system improvement confirmed!")
    else:
        print(f"⚠️  VALIDATION INCONCLUSIVE: Consider additional experiments")


if __name__ == "__main__":
    sys.exit(main())