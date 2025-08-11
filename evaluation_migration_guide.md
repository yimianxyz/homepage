# Evaluation System Migration Guide

## What Changed

The evaluation system has been upgraded from a high-variance 5-episode system to a configurable low-variance system with statistical confidence measures.

### Old System Issues
- Only 5 episodes → ±0.08 std deviation (~10% of mean)
- No confidence intervals reported
- Baseline varied from 0.6944 to 0.8222 (just noise!)
- Could not reliably detect <15% improvements

### New System Benefits
- Default 15 episodes → ±0.023 std error (~3% of mean)
- 95% confidence intervals on all results
- 50% variance reduction
- Can reliably detect >9% improvements
- Statistical significance testing built-in

## Code Changes

### Imports
```python
# No change needed - same import
from evaluation import PolicyEvaluator
```

### Basic Usage
```python
# Old way (still works)
evaluator = PolicyEvaluator()
result = evaluator.evaluate_policy(policy)
performance = result.overall_catch_rate

# New way (with confidence)
evaluator = PolicyEvaluator()  # Now uses 15 episodes by default
result = evaluator.evaluate_policy(policy)
performance = result.overall_catch_rate
confidence_interval = (result.confidence_95_lower, result.confidence_95_upper)
```

### New Features

1. **Confidence Intervals**
```python
print(f"Performance: {result.overall_catch_rate:.4f}")
print(f"95% CI: [{result.confidence_95_lower:.4f}, {result.confidence_95_upper:.4f}]")
```

2. **Statistical Comparisons**
```python
# Compare two policies with p-values
comparison = evaluator.compare_policies([
    (baseline_policy, "Baseline"),
    (new_policy, "New")
])
# Automatically reports p-values and effect sizes
```

3. **Configurable Precision**
```python
# Choose your tradeoff
quick = PolicyEvaluator(num_episodes=10)      # ~60s, detects >11%
standard = PolicyEvaluator(num_episodes=15)   # ~90s, detects >9% (default)
precise = PolicyEvaluator(num_episodes=30)    # ~180s, detects >6%
```

## Migration Checklist

- [ ] Update any hardcoded expectations of 5 episodes
- [ ] Add confidence interval reporting to results
- [ ] Update any variance/threshold checks (new system has 50% less variance)
- [ ] Consider using statistical comparison for A/B tests
- [ ] Set appropriate base_seed for reproducibility

## Backward Compatibility

The new system is fully backward compatible:
- All old attributes still exist
- Default behavior gives better results
- Old code will work without changes
- New features are additive

## Example: Updating Experiments

```python
# Old experiment code
def run_experiment():
    evaluator = PolicyEvaluator()
    
    baseline = evaluator.evaluate_policy(sl_policy)
    improved = evaluator.evaluate_policy(rl_policy)
    
    improvement = (improved.overall_catch_rate - baseline.overall_catch_rate) / baseline.overall_catch_rate * 100
    print(f"Improvement: {improvement:.1f}%")

# Updated with confidence
def run_experiment_with_confidence():
    evaluator = PolicyEvaluator()  # Now 15 episodes
    
    # Use comparison method for automatic statistics
    results = evaluator.compare_policies([
        (sl_policy, "Baseline"),
        (rl_policy, "Improved")
    ])
    
    # Results include p-values and significance testing
```

## Performance Impact

- Old: 5 episodes × 6s = ~30s
- New: 15 episodes × 6s = ~90s
- Benefit: 50% variance reduction, reliable significance testing

The 60s increase in evaluation time provides:
- 3x more statistical power
- Confidence intervals
- Significance testing
- Reproducible results