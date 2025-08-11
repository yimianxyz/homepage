# PPO Reinforcement Learning Findings Summary

## Executive Summary

After extensive experimentation with PPO reinforcement learning to improve upon the supervised learning (SL) baseline, we discovered that while improvement is possible, it requires extremely careful hyperparameter tuning and very short training durations to avoid catastrophic forgetting.

## Key Findings

### 1. Production Training Results
- **Baseline Performance**: 0.7722 ± 0.0324 (77.2% catch rate)
- **Best PPO Performance**: 0.7611 at iteration 10 (-1.4%)
- **Final Performance**: 0.4667 after 300 iterations (-39.6%)
- **Key Issue**: Catastrophic forgetting after ~10 iterations

### 2. Value Pre-training Impact
- Value pre-training alone cannot improve performance (doesn't change policy)
- However, it's essential for stable PPO training
- Optimal duration: 20 iterations
- Reduces initial value loss from ~0.058 to ~0.025

### 3. Optimal Training Duration
- Performance peaks at iterations 1-10
- Longer training leads to catastrophic forgetting
- Best approach: Stop at 1-3 PPO iterations

### 4. Hyperparameter Sensitivity
From partial grid search results:
- **Learning Rate**: 3e-5 maintains performance, 1e-5 still causes degradation
- **Clip Epsilon**: 0.01 (very conservative) prevents drastic policy changes
- **Episode Length**: Should match evaluation (2500 steps, not 5000)

## Root Causes of Failure

1. **Distribution Mismatch**: Training on random episodes vs evaluation on fixed scenarios
2. **Aggressive Updates**: Standard PPO hyperparameters too aggressive for fine-tuning
3. **Sparse Rewards**: Only catch events provide signal, leading to conservative behavior
4. **Pre-trained Model Fragility**: SL baseline easily corrupted by RL updates

## Recommended Approach

### Configuration
```python
# Optimal settings discovered
value_pretrain_iterations = 20
ppo_iterations = 1-3  # No more!
learning_rate = 3e-5
clip_epsilon = 0.01  # Very conservative
episode_length = 2500  # Match evaluation
```

### Training Protocol
1. Load SL baseline model
2. Freeze policy, train value head for 20 iterations
3. Unfreeze policy, run 1-3 PPO iterations maximum
4. Stop immediately if performance degrades

### Expected Improvement
- Realistic: 0-5% improvement over SL baseline
- Risk: -40% degradation if overtrained

## Future Directions

1. **KL Penalty**: Add explicit KL constraint to prevent diverging from SL
2. **Mixed Objectives**: Combine SL loss with RL to maintain baseline performance
3. **Dense Rewards**: Add proximity-based rewards for better signal
4. **Curriculum Learning**: Start with easy scenarios, gradually increase difficulty
5. **Conservative Policy Gradient**: Use TRPO or other trust region methods

## Conclusion

PPO can theoretically improve the SL baseline, but the improvement window is extremely narrow (1-3 iterations) and requires very conservative hyperparameters. The risk of catastrophic forgetting is high, making this approach fragile in practice. The SL baseline represents a strong local optimum that is difficult to improve upon with standard RL methods.