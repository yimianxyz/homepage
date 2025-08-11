#!/usr/bin/env python3
"""
Quick analysis of training trajectory patterns
"""

import numpy as np
import matplotlib.pyplot as plt

# Data from Trial 1 Standard PPO
standard_trial1_iters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
standard_trial1_perfs = [0.8500, 0.7833, 0.7500, 0.6167, 0.8000, 0.5667, 0.8000, 0.8333, 0.8000, 0.7833, 0.8833, 0.7000, 0.6500, 0.7333]

# Data from Trial 1 Value Pre-trained PPO  
pretrained_trial1_iters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
pretrained_trial1_perfs = [0.7333, 0.7333, 0.7500, 0.7167, 0.8167, 0.7333, 0.7833, 0.8500, 0.8333, 0.7833, 0.7833, 0.8333, 0.7500, 0.7167, 0.7667]

# SL Baseline
sl_baseline = 0.8167

# Quick visualization
plt.figure(figsize=(12, 6))

# Plot trajectories
plt.plot(standard_trial1_iters, standard_trial1_perfs, 'b-o', label='Standard PPO', linewidth=2, markersize=8)
plt.plot(pretrained_trial1_iters, pretrained_trial1_perfs, 'g-s', label='Value Pre-trained PPO', linewidth=2, markersize=8)
plt.axhline(y=sl_baseline, color='red', linestyle='--', label='SL Baseline', linewidth=2)

# Add shaded regions
plt.axhspan(sl_baseline * 0.95, sl_baseline, alpha=0.2, color='yellow', label='Within 5% of baseline')
plt.axhspan(sl_baseline, 1.0, alpha=0.2, color='green', label='Beats baseline')

plt.xlabel('Training Iteration', fontsize=12)
plt.ylabel('Performance (Catch Rate)', fontsize=12)
plt.title('Training Trajectory Comparison - Trial 1', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 0.9)

# Add annotations for key points
# Standard PPO peaks
plt.annotate('Peak: 0.883', xy=(10, 0.8833), xytext=(12, 0.88),
            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))

# Value pre-trained peak
plt.annotate('Peak: 0.850', xy=(7, 0.8500), xytext=(5, 0.87),
            arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

plt.tight_layout()
plt.savefig('quick_trajectory_analysis.png', dpi=150)

# Print key insights
print("🔍 QUICK TRAJECTORY ANALYSIS")
print("=" * 50)

print("\n📊 STANDARD PPO:")
print(f"  Initial (iter 0): {standard_trial1_perfs[0]:.3f}")
print(f"  Best performance: {max(standard_trial1_perfs):.3f} at iter {standard_trial1_iters[np.argmax(standard_trial1_perfs)]}")
print(f"  Iterations beating baseline: {[i for i, p in zip(standard_trial1_iters, standard_trial1_perfs) if p > sl_baseline]}")
print(f"  High volatility: {np.std(standard_trial1_perfs):.3f}")

print("\n📊 VALUE PRE-TRAINED PPO:")
print(f"  Initial (iter 0): {pretrained_trial1_perfs[0]:.3f} (after pre-training)")
print(f"  Best performance: {max(pretrained_trial1_perfs):.3f} at iter {pretrained_trial1_iters[np.argmax(pretrained_trial1_perfs)]}")
print(f"  Iterations beating baseline: {[i for i, p in zip(pretrained_trial1_iters, pretrained_trial1_perfs) if p > sl_baseline]}")
print(f"  Lower volatility: {np.std(pretrained_trial1_perfs):.3f}")

print("\n💡 KEY INSIGHTS:")
print("1. Standard PPO shows extreme volatility (0.567 to 0.883)")
print("2. Value pre-training provides more stable performance")
print("3. Both methods can beat baseline but inconsistently")
print("4. Performance degrades after ~10-15 iterations")
print("5. Need early stopping or lower learning rate")

print("\n✅ Plot saved: quick_trajectory_analysis.png")