#!/usr/bin/env python3
"""
PPO Production Monitor - Real-time monitoring and analysis tools

FEATURES:
1. Real-time training monitoring
2. Performance analysis and visualization
3. Early warning system for issues
4. Post-training analysis tools
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from datetime import datetime
from scipy import stats


class PPOProductionMonitor:
    """Monitor and analyze production PPO training"""
    
    def __init__(self, checkpoint_dir: str = "production_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.state_file = f"{checkpoint_dir}/training_state.json"
        
        print("📊 PPO PRODUCTION MONITOR")
        print("=" * 70)
        print(f"Monitoring: {checkpoint_dir}")
        print("=" * 70)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current training status"""
        if not os.path.exists(self.state_file):
            return {"status": "not_started"}
        
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        
        # Calculate current metrics
        current_iter = state['current_iteration']
        best_perf = state['best_performance']
        best_iter = state['best_iteration']
        
        if state['sl_baseline']:
            baseline = state['sl_baseline']['mean']
            improvement = (best_perf - baseline) / baseline * 100
        else:
            improvement = 0.0
        
        # Training progress
        if 'validation_history' in state and state['validation_history']:
            last_val = state['validation_history'][-1]
            current_perf = last_val['mean']
            current_improvement = last_val['improvement_vs_baseline']
        else:
            current_perf = 0.0
            current_improvement = 0.0
        
        # Early stopping risk
        early_stop_risk = self._assess_early_stop_risk(state)
        
        status = {
            'status': 'completed' if state.get('early_stop_triggered', False) else 'running',
            'current_iteration': current_iter,
            'best_performance': best_perf,
            'best_iteration': best_iter,
            'current_performance': current_perf,
            'improvement_vs_baseline': improvement,
            'current_improvement': current_improvement,
            'total_time_minutes': state.get('total_training_time', 0) / 60,
            'early_stop_triggered': state.get('early_stop_triggered', False),
            'early_stop_reason': state.get('early_stop_reason'),
            'early_stop_risk': early_stop_risk
        }
        
        return status
    
    def _assess_early_stop_risk(self, state: Dict) -> str:
        """Assess risk of early stopping"""
        if len(state.get('validation_history', [])) < 5:
            return "low"
        
        recent_vals = state['validation_history'][-5:]
        recent_perfs = [v['mean'] for v in recent_vals]
        
        # Check for plateau
        perf_range = max(recent_perfs) - min(recent_perfs)
        if perf_range < 0.001:  # Very small variation
            return "high"
        
        # Check for downward trend
        if all(recent_perfs[i] <= recent_perfs[i-1] for i in range(1, len(recent_perfs))):
            return "high"
        
        # Check iterations since best
        current_iter = state['current_iteration']
        best_iter = state['best_iteration']
        if current_iter - best_iter > 15:
            return "medium"
        
        return "low"
    
    def print_live_status(self):
        """Print current training status"""
        status = self.get_current_status()
        
        if status['status'] == 'not_started':
            print("❌ Training not started yet")
            return
        
        print(f"\n📊 TRAINING STATUS")
        print(f"{'='*50}")
        print(f"Status: {status['status'].upper()}")
        print(f"Current Iteration: {status['current_iteration']}")
        print(f"Training Time: {status['total_time_minutes']:.1f} minutes")
        print(f"\nPERFORMANCE:")
        print(f"  Current: {status['current_performance']:.4f} ({status['current_improvement']:+.1f}%)")
        print(f"  Best: {status['best_performance']:.4f} ({status['improvement_vs_baseline']:+.1f}%)")
        print(f"  Best at iteration: {status['best_iteration']}")
        
        # Early stop risk indicator
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        print(f"\nEARLY STOP RISK: {risk_emoji[status['early_stop_risk']]} {status['early_stop_risk'].upper()}")
        
        if status['early_stop_triggered']:
            print(f"\n🛑 EARLY STOP TRIGGERED: {status['early_stop_reason']}")
    
    def analyze_training_curves(self) -> Dict[str, Any]:
        """Analyze training curves and patterns"""
        if not os.path.exists(self.state_file):
            return {"error": "No training data found"}
        
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        
        analysis = {
            'performance_trend': self._analyze_performance_trend(state),
            'loss_analysis': self._analyze_losses(state),
            'efficiency_analysis': self._analyze_efficiency(state),
            'stability_analysis': self._analyze_stability(state)
        }
        
        return analysis
    
    def _analyze_performance_trend(self, state: Dict) -> Dict[str, Any]:
        """Analyze performance trend"""
        val_history = state.get('validation_history', [])
        if not val_history:
            return {"status": "no_data"}
        
        iterations = [v['iteration'] for v in val_history]
        performances = [v['mean'] for v in val_history]
        improvements = [v['improvement_vs_baseline'] for v in val_history]
        
        # Fit trend line
        if len(iterations) > 3:
            z = np.polyfit(iterations, performances, 2)  # Quadratic fit
            p = np.poly1d(z)
            trend_type = "increasing" if z[0] > 0 else "decreasing" if z[0] < 0 else "flat"
        else:
            trend_type = "insufficient_data"
            p = None
        
        # Find peak
        peak_idx = np.argmax(performances)
        peak_iter = iterations[peak_idx]
        peak_perf = performances[peak_idx]
        
        # Plateau detection
        if len(performances) > 5:
            recent_std = np.std(performances[-5:])
            plateau_detected = recent_std < 0.001
        else:
            plateau_detected = False
        
        return {
            'iterations': iterations,
            'performances': performances,
            'improvements': improvements,
            'trend_type': trend_type,
            'trend_function': p,
            'peak_iteration': peak_iter,
            'peak_performance': peak_perf,
            'plateau_detected': plateau_detected,
            'final_performance': performances[-1] if performances else 0,
            'total_improvement': improvements[-1] if improvements else 0
        }
    
    def _analyze_losses(self, state: Dict) -> Dict[str, Any]:
        """Analyze training losses"""
        train_history = state.get('training_history', [])
        if not train_history:
            return {"status": "no_data"}
        
        policy_losses = [h.get('policy_loss', 0) for h in train_history]
        value_losses = [h.get('value_loss', 0) for h in train_history]
        
        # Loss statistics
        return {
            'policy_loss': {
                'mean': np.mean(policy_losses),
                'std': np.std(policy_losses),
                'trend': 'decreasing' if len(policy_losses) > 5 and policy_losses[-1] < policy_losses[0] else 'stable',
                'final': policy_losses[-1] if policy_losses else 0
            },
            'value_loss': {
                'mean': np.mean(value_losses),
                'std': np.std(value_losses),
                'trend': 'decreasing' if len(value_losses) > 5 and value_losses[-1] < value_losses[0] else 'stable',
                'final': value_losses[-1] if value_losses else 0
            }
        }
    
    def _analyze_efficiency(self, state: Dict) -> Dict[str, Any]:
        """Analyze training efficiency"""
        val_history = state.get('validation_history', [])
        if not val_history:
            return {"status": "no_data"}
        
        # Calculate improvement per iteration
        efficiencies = []
        for i, val in enumerate(val_history):
            if val['iteration'] > 0:
                efficiency = val['improvement_vs_baseline'] / val['iteration']
                efficiencies.append({
                    'iteration': val['iteration'],
                    'efficiency': efficiency,
                    'improvement': val['improvement_vs_baseline']
                })
        
        # Find point of diminishing returns
        diminishing_point = None
        if len(efficiencies) > 5:
            for i in range(5, len(efficiencies)):
                if efficiencies[i]['efficiency'] < efficiencies[i-5]['efficiency'] * 0.5:
                    diminishing_point = efficiencies[i]['iteration']
                    break
        
        return {
            'efficiencies': efficiencies,
            'diminishing_returns_point': diminishing_point,
            'final_efficiency': efficiencies[-1]['efficiency'] if efficiencies else 0
        }
    
    def _analyze_stability(self, state: Dict) -> Dict[str, Any]:
        """Analyze training stability"""
        val_history = state.get('validation_history', [])
        if not val_history:
            return {"status": "no_data"}
        
        # Performance variance over time
        if len(val_history) > 10:
            early_perfs = [v['mean'] for v in val_history[:5]]
            late_perfs = [v['mean'] for v in val_history[-5:]]
            
            early_variance = np.var(early_perfs)
            late_variance = np.var(late_perfs)
            
            stability_improved = late_variance < early_variance
        else:
            stability_improved = None
            early_variance = None
            late_variance = None
        
        # Sudden drops
        sudden_drops = []
        for i in range(1, len(val_history)):
            if val_history[i]['mean'] < val_history[i-1]['mean'] * 0.95:
                sudden_drops.append({
                    'iteration': val_history[i]['iteration'],
                    'drop_percentage': (1 - val_history[i]['mean'] / val_history[i-1]['mean']) * 100
                })
        
        return {
            'early_variance': early_variance,
            'late_variance': late_variance,
            'stability_improved': stability_improved,
            'sudden_drops': sudden_drops,
            'num_sudden_drops': len(sudden_drops)
        }
    
    def create_training_visualization(self, save_path: str = "production_training_analysis.png"):
        """Create comprehensive training visualization"""
        if not os.path.exists(self.state_file):
            print("❌ No training data found")
            return
        
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        
        analysis = self.analyze_training_curves()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PPO Production Training Analysis', fontsize=16, fontweight='bold')
        
        # 1. Performance curve
        ax1 = axes[0, 0]
        perf_data = analysis['performance_trend']
        if perf_data.get('iterations'):
            ax1.plot(perf_data['iterations'], perf_data['performances'], 
                    'b-', linewidth=2, label='Validation')
            ax1.plot(perf_data['peak_iteration'], perf_data['peak_performance'], 
                    'r*', markersize=15, label='Best')
            
            # Add trend line
            if perf_data['trend_function']:
                x_smooth = np.linspace(min(perf_data['iterations']), 
                                     max(perf_data['iterations']), 100)
                ax1.plot(x_smooth, perf_data['trend_function'](x_smooth), 
                        'r--', alpha=0.5, label='Trend')
            
            # Add baseline
            if state.get('sl_baseline'):
                ax1.axhline(y=state['sl_baseline']['mean'], color='green', 
                          linestyle='--', label='SL Baseline')
            
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Performance')
            ax1.set_title('Performance Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Improvement percentage
        ax2 = axes[0, 1]
        if perf_data.get('iterations'):
            ax2.plot(perf_data['iterations'], perf_data['improvements'], 
                    'g-', linewidth=2)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.fill_between(perf_data['iterations'], 0, perf_data['improvements'], 
                           alpha=0.3, color='green')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Improvement vs Baseline (%)')
            ax2.set_title('Improvement Over Time')
            ax2.grid(True, alpha=0.3)
        
        # 3. Loss curves
        ax3 = axes[0, 2]
        train_history = state.get('training_history', [])
        if train_history:
            iterations = [h['iteration'] for h in train_history]
            policy_losses = [h.get('policy_loss', 0) for h in train_history]
            value_losses = [h.get('value_loss', 0) for h in train_history]
            
            ax3_twin = ax3.twinx()
            l1 = ax3.plot(iterations, policy_losses, 'b-', label='Policy Loss')
            l2 = ax3_twin.plot(iterations, value_losses, 'r-', label='Value Loss')
            
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Policy Loss', color='b')
            ax3_twin.set_ylabel('Value Loss', color='r')
            ax3.set_title('Training Losses')
            
            # Combine legends
            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            ax3.legend(lns, labs, loc='upper right')
            ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency curve
        ax4 = axes[1, 0]
        eff_data = analysis['efficiency_analysis']
        if eff_data.get('efficiencies'):
            effs = eff_data['efficiencies']
            ax4.plot([e['iteration'] for e in effs], 
                    [e['efficiency'] for e in effs], 
                    'orange', linewidth=2)
            
            if eff_data['diminishing_returns_point']:
                ax4.axvline(x=eff_data['diminishing_returns_point'], 
                          color='red', linestyle='--', 
                          label='Diminishing Returns')
            
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Improvement per Iteration (%)')
            ax4.set_title('Training Efficiency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Validation variance
        ax5 = axes[1, 1]
        val_history = state.get('validation_history', [])
        if len(val_history) > 2:
            iterations = [v['iteration'] for v in val_history]
            stds = [v['std'] for v in val_history]
            
            ax5.plot(iterations, stds, 'purple', linewidth=2)
            ax5.fill_between(iterations, 0, stds, alpha=0.3, color='purple')
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('Standard Deviation')
            ax5.set_title('Validation Uncertainty')
            ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = "TRAINING SUMMARY\n" + "="*25 + "\n\n"
        
        status = self.get_current_status()
        summary_text += f"Status: {status['status'].upper()}\n"
        summary_text += f"Iterations: {status['current_iteration']}\n"
        summary_text += f"Time: {status['total_time_minutes']:.1f} minutes\n\n"
        
        summary_text += f"PERFORMANCE:\n"
        summary_text += f"Best: {status['best_performance']:.4f} "
        summary_text += f"(+{status['improvement_vs_baseline']:.1f}%)\n"
        summary_text += f"Current: {status['current_performance']:.4f} "
        summary_text += f"(+{status['current_improvement']:.1f}%)\n\n"
        
        summary_text += f"ANALYSIS:\n"
        summary_text += f"Trend: {perf_data['trend_type']}\n"
        summary_text += f"Plateau: {'YES' if perf_data['plateau_detected'] else 'NO'}\n"
        
        loss_data = analysis['loss_analysis']
        if loss_data.get('value_loss'):
            summary_text += f"Value Loss: {loss_data['value_loss']['final']:.3f} "
            summary_text += f"({loss_data['value_loss']['trend']})\n"
        
        if status['early_stop_triggered']:
            summary_text += f"\nEARLY STOP: {status['early_stop_reason']}"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved: {save_path}")
        
        return fig
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        status = self.get_current_status()
        analysis = self.analyze_training_curves()
        
        report = {
            'status': status,
            'analysis': analysis,
            'recommendations': self._generate_recommendations(status, analysis),
            'timestamp': datetime.now().isoformat()
        }
        
        # Print report
        print("\n" + "="*70)
        print("PPO PRODUCTION TRAINING REPORT")
        print("="*70)
        
        print(f"\nSTATUS: {status['status'].upper()}")
        print(f"Best Performance: {status['best_performance']:.4f} "
              f"(+{status['improvement_vs_baseline']:.1f}% vs baseline)")
        
        print(f"\nKEY FINDINGS:")
        
        # Performance trend
        perf_trend = analysis['performance_trend']
        if perf_trend.get('trend_type'):
            print(f"• Performance trend: {perf_trend['trend_type']}")
            print(f"• Peak at iteration {perf_trend['peak_iteration']}")
            if perf_trend['plateau_detected']:
                print(f"• ⚠️  Performance plateau detected")
        
        # Efficiency
        eff_analysis = analysis['efficiency_analysis']
        if eff_analysis.get('diminishing_returns_point'):
            print(f"• Diminishing returns after iteration {eff_analysis['diminishing_returns_point']}")
        
        # Stability
        stability = analysis['stability_analysis']
        if stability.get('num_sudden_drops', 0) > 0:
            print(f"• ⚠️  {stability['num_sudden_drops']} sudden performance drops detected")
        
        print(f"\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"• {rec}")
        
        # Save report
        report_path = f"{self.checkpoint_dir}/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Report saved: {report_path}")
        
        return report
    
    def _generate_recommendations(self, status: Dict, analysis: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on performance trend
        perf_trend = analysis['performance_trend']
        if perf_trend.get('plateau_detected'):
            recommendations.append("Performance plateaued - consider stopping training")
        
        if perf_trend.get('trend_type') == 'decreasing':
            recommendations.append("Performance declining - check for overfitting")
        
        # Based on efficiency
        eff_analysis = analysis['efficiency_analysis']
        if eff_analysis.get('diminishing_returns_point'):
            if status['current_iteration'] > eff_analysis['diminishing_returns_point']:
                recommendations.append("Past diminishing returns point - minimal benefit from continued training")
        
        # Based on stability
        stability = analysis['stability_analysis']
        if stability.get('num_sudden_drops', 0) > 2:
            recommendations.append("Multiple sudden drops detected - consider reducing learning rate")
        
        # Based on early stop risk
        if status.get('early_stop_risk') == 'high':
            recommendations.append("High early stop risk - save current checkpoint if needed")
        
        # General recommendations
        if status['improvement_vs_baseline'] > 10:
            recommendations.append("Excellent improvement achieved - current model ready for deployment")
        elif status['improvement_vs_baseline'] > 5:
            recommendations.append("Good improvement achieved - consider production deployment")
        
        if not recommendations:
            recommendations.append("Training progressing normally - continue monitoring")
        
        return recommendations


def main():
    """Demonstrate production monitoring"""
    monitor = PPOProductionMonitor()
    
    print("\n1️⃣ CURRENT STATUS:")
    monitor.print_live_status()
    
    print("\n2️⃣ ANALYZING TRAINING:")
    analysis = monitor.analyze_training_curves()
    
    print("\n3️⃣ CREATING VISUALIZATION:")
    monitor.create_training_visualization()
    
    print("\n4️⃣ GENERATING REPORT:")
    report = monitor.generate_training_report()
    
    return monitor


if __name__ == "__main__":
    main()