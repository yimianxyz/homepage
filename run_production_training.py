#!/usr/bin/env python3
"""
Run Production PPO Training - Complete workflow demonstration

This script demonstrates the full production training workflow:
1. Setup and configuration
2. Launch training with monitoring
3. Handle interruptions gracefully
4. Analyze results
"""

import os
import sys
import time
import signal
import threading
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppo_production_trainer import ProductionPPOTrainer, TrainingConfig
from ppo_production_monitor import PPOProductionMonitor


class ProductionTrainingManager:
    """Manage production training with monitoring"""
    
    def __init__(self):
        self.trainer = None
        self.monitor = None
        self.training_thread = None
        self.monitoring_active = False
        
        print("🚀 PPO PRODUCTION TRAINING MANAGER")
        print("=" * 80)
        print("Complete production training workflow with:")
        print("• Automatic checkpointing")
        print("• Early stopping")
        print("• Real-time monitoring")
        print("• Graceful interruption handling")
        print("=" * 80)
    
    def setup_training(self) -> TrainingConfig:
        """Setup training configuration based on our analysis"""
        print("\n📋 SETTING UP TRAINING CONFIGURATION")
        
        # Based on our scaling analysis:
        # - 100-200 iterations for best performance
        # - Value pre-training for stability
        # - Conservative hyperparameters
        
        config = TrainingConfig(
            # Model paths
            sl_checkpoint_path="checkpoints/best_model.pt",
            checkpoint_dir="production_checkpoints",
            
            # Training scale (from scaling analysis)
            max_iterations=150,  # Sweet spot between performance and time
            value_pretrain_iterations=20,
            
            # Checkpoint strategy
            checkpoint_interval=5,   # Save every 5 iterations
            keep_best_k=5,          # Keep top 5 models
            
            # Validation strategy
            validation_interval=5,   # Validate every 5 iterations
            validation_runs=3,      # 3 runs for statistical validity
            
            # Early stopping (conservative)
            patience=25,            # Stop after 25 iterations without improvement
            min_delta=0.0005,      # 0.05% minimum improvement
            confidence_level=0.95,  # 95% confidence for decisions
            
            # Optimized hyperparameters
            learning_rate=0.00005,
            value_learning_rate=0.0005,
            clip_epsilon=0.1,
            ppo_epochs=2,
            rollout_steps=256,
            gamma=0.95,
            gae_lambda=0.9
        )
        
        print("✅ Configuration created:")
        print(f"   Max iterations: {config.max_iterations}")
        print(f"   Checkpoint every: {config.checkpoint_interval} iterations")
        print(f"   Early stop patience: {config.patience} iterations")
        print(f"   Expected time: ~{config.max_iterations * 2} minutes")
        
        return config
    
    def start_training(self, config: TrainingConfig):
        """Start training in a separate thread"""
        print("\n🚀 STARTING PRODUCTION TRAINING")
        
        # Create trainer
        self.trainer = ProductionPPOTrainer(config)
        
        # Create monitor
        self.monitor = PPOProductionMonitor(config.checkpoint_dir)
        
        # Start training in thread
        self.training_thread = threading.Thread(
            target=self.trainer.run_production_training,
            name="PPOTraining"
        )
        self.training_thread.start()
        
        print("✅ Training started in background")
        print("   Use Ctrl+C to safely interrupt")
    
    def monitor_training(self, update_interval: int = 30):
        """Monitor training progress"""
        print(f"\n📊 MONITORING TRAINING (updates every {update_interval}s)")
        print("Press Ctrl+C to stop monitoring (training continues)")
        
        self.monitoring_active = True
        
        try:
            while self.monitoring_active and self.training_thread.is_alive():
                # Clear screen for clean update
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("🚀 PPO PRODUCTION TRAINING - LIVE MONITOR")
                print("=" * 70)
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get current status
                self.monitor.print_live_status()
                
                # Show recent progress
                status = self.monitor.get_current_status()
                if status['status'] == 'running':
                    print(f"\nNext update in {update_interval} seconds...")
                    print("Press Ctrl+C to stop monitoring")
                
                # Wait for next update
                time.sleep(update_interval)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoring stopped (training continues in background)")
            self.monitoring_active = False
    
    def wait_for_completion(self):
        """Wait for training to complete"""
        if self.training_thread and self.training_thread.is_alive():
            print("\n⏳ Waiting for training to complete...")
            print("   Press Ctrl+C to interrupt training gracefully")
            
            try:
                self.training_thread.join()
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupting training gracefully...")
                # The trainer handles interruption internally
                self.training_thread.join(timeout=10)
        
        print("\n✅ Training completed")
    
    def analyze_results(self):
        """Analyze training results"""
        print("\n📊 ANALYZING TRAINING RESULTS")
        
        # Generate visualization
        print("\n1️⃣ Creating visualization...")
        self.monitor.create_training_visualization()
        
        # Generate report
        print("\n2️⃣ Generating comprehensive report...")
        report = self.monitor.generate_training_report()
        
        # Best model info
        best_model_path = f"{self.trainer.config.checkpoint_dir}/best/best_model.pt"
        if os.path.exists(best_model_path):
            print(f"\n✅ BEST MODEL SAVED: {best_model_path}")
            
            # Load best metadata
            import json
            with open(f"{self.trainer.config.checkpoint_dir}/best/best_model_metadata.json", 'r') as f:
                best_metadata = json.load(f)
            
            print(f"   Performance: {best_metadata['performance']:.4f}")
            print(f"   Improvement: {best_metadata['improvement_vs_baseline']:.1f}%")
            print(f"   Iteration: {best_metadata['iteration']}")
    
    def run_complete_workflow(self):
        """Run complete production training workflow"""
        print("\n" + "="*80)
        print("COMPLETE PRODUCTION TRAINING WORKFLOW")
        print("="*80)
        
        try:
            # 1. Setup
            config = self.setup_training()
            
            # 2. Start training
            self.start_training(config)
            
            # 3. Monitor progress
            time.sleep(5)  # Let training start
            self.monitor_training(update_interval=30)
            
            # 4. Wait for completion
            self.wait_for_completion()
            
            # 5. Analyze results
            self.analyze_results()
            
            print("\n" + "="*80)
            print("🎉 PRODUCTION TRAINING COMPLETE!")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False


def quick_demo():
    """Quick demonstration of production training setup"""
    print("📋 PRODUCTION TRAINING QUICK DEMO")
    print("=" * 70)
    
    # Show configuration
    config = TrainingConfig()
    
    print("\n1️⃣ DEFAULT CONFIGURATION:")
    print(f"   Training iterations: {config.max_iterations}")
    print(f"   Value pre-training: {config.value_pretrain_iterations} iterations")
    print(f"   Checkpoint interval: Every {config.checkpoint_interval} iterations")
    print(f"   Early stop patience: {config.patience} iterations")
    
    print("\n2️⃣ CHECKPOINT STRUCTURE:")
    print("   production_checkpoints/")
    print("   ├── best/              # Best model")
    print("   │   ├── best_model.pt")
    print("   │   └── best_model_metadata.json")
    print("   ├── checkpoint_iter_0005.pt")
    print("   ├── checkpoint_iter_0010.pt")
    print("   ├── training_state.json  # For resume")
    print("   └── final_report.json")
    
    print("\n3️⃣ MONITORING FEATURES:")
    print("   • Real-time performance tracking")
    print("   • Early stop risk assessment")
    print("   • Training curve analysis")
    print("   • Automatic report generation")
    
    print("\n4️⃣ ROBUSTNESS FEATURES:")
    print("   • Automatic resume from interruption")
    print("   • Statistical early stopping")
    print("   • Best model tracking")
    print("   • Comprehensive logging")
    
    print("\n5️⃣ EXPECTED RESULTS (based on analysis):")
    print("   • 10-12% improvement over SL baseline")
    print("   • Peak performance around iteration 80-120")
    print("   • Training time: 2-3 hours")
    print("   • All results statistically significant (p<0.05)")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO Production Training')
    parser.add_argument('--demo', action='store_true', help='Show demo information')
    parser.add_argument('--train', action='store_true', help='Start production training')
    parser.add_argument('--monitor', action='store_true', help='Monitor existing training')
    
    args = parser.parse_args()
    
    if args.demo:
        quick_demo()
    elif args.monitor:
        monitor = PPOProductionMonitor()
        monitor.print_live_status()
        monitor.create_training_visualization()
    elif args.train:
        manager = ProductionTrainingManager()
        manager.run_complete_workflow()
    else:
        print("🚀 PPO PRODUCTION TRAINING")
        print("\nOptions:")
        print("  --demo    : Show demo information")
        print("  --train   : Start production training")
        print("  --monitor : Monitor existing training")
        print("\nExample: python run_production_training.py --train")


if __name__ == "__main__":
    main()