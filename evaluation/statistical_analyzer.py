"""Statistical analysis utilities for policy evaluation."""

import numpy as np
from typing import Dict, List, Tuple, Any
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class StatisticalAnalyzer:
    """Provides statistical analysis for evaluation results."""
    
    @staticmethod
    def calculate_confidence_interval(
        data: List[float], 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for given data.
        
        Args:
            data: List of values
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        
        if SCIPY_AVAILABLE:
            std_err = stats.sem(data)
            interval = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        else:
            # Simple approximation using normal distribution
            std_err = np.std(data, ddof=1) / np.sqrt(n)
            z_score = 1.96 if confidence == 0.95 else 2.58  # Common z-scores
            interval = std_err * z_score
        
        return (mean - interval, mean + interval)
    
    @staticmethod
    def compare_distributions(
        data1: List[float], 
        data2: List[float],
        test: str = 'welch'
    ) -> Dict[str, float]:
        """Compare two distributions using statistical tests.
        
        Args:
            data1: First dataset
            data2: Second dataset
            test: Type of test ('welch' for Welch's t-test, 'mann-whitney' for Mann-Whitney U)
            
        Returns:
            Dictionary with test results
        """
        if not SCIPY_AVAILABLE:
            # Simple comparison without scipy
            mean1, mean2 = np.mean(data1), np.mean(data2)
            std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
            n1, n2 = len(data1), len(data2)
            
            # Simplified t-statistic
            pooled_std = np.sqrt((std1**2/n1) + (std2**2/n2))
            statistic = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            # Very rough p-value approximation
            p_value = 0.05 if abs(statistic) > 2 else 0.5
            test_name = "Approximate t-test"
        else:
            if test == 'welch':
                statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                test_name = "Welch's t-test"
            elif test == 'mann-whitney':
                statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                test_name = "Mann-Whitney U test"
            else:
                raise ValueError(f"Unknown test type: {test}")
            
        # Calculate effect size (Cohen's d)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        pooled_std = np.sqrt((np.std(data1)**2 + np.std(data2)**2) / 2)
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'mean_difference': mean1 - mean2
        }
    
    @staticmethod
    def analyze_performance_stability(
        episode_metrics: Dict[str, List[float]], 
        window_size: int = 10
    ) -> Dict[str, Any]:
        """Analyze performance stability over episodes.
        
        Args:
            episode_metrics: Dictionary of metric lists from evaluation
            window_size: Size of moving average window
            
        Returns:
            Dictionary with stability analysis
        """
        results = {}
        
        # Calculate moving averages
        for metric_name, values in episode_metrics.items():
            if metric_name == 'success':  # Convert boolean to float
                values = [float(v) for v in values]
                
            # Moving average
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            
            # Trend analysis
            x = np.arange(len(values))
            
            if SCIPY_AVAILABLE:
                slope, intercept, r_value, _, _ = stats.linregress(x, values)
                r_squared = r_value**2
            else:
                # Simple linear regression without scipy
                x_mean = np.mean(x)
                y_mean = np.mean(values)
                
                numerator = np.sum((x - x_mean) * (values - y_mean))
                denominator = np.sum((x - x_mean)**2)
                
                slope = numerator / denominator if denominator != 0 else 0
                intercept = y_mean - slope * x_mean
                
                # Calculate R-squared
                y_pred = slope * x + intercept
                ss_res = np.sum((values - y_pred)**2)
                ss_tot = np.sum((values - y_mean)**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            results[metric_name] = {
                'moving_average': moving_avg.tolist(),
                'trend_slope': slope,
                'trend_r_squared': r_squared,
                'variance': np.var(values),
                'improving': slope > 0
            }
        
        return results
    
    @staticmethod
    def calculate_percentiles(
        data: List[float], 
        percentiles: List[int] = [25, 50, 75, 90, 95, 99]
    ) -> Dict[int, float]:
        """Calculate percentiles for performance metrics.
        
        Args:
            data: List of values
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary mapping percentile to value
        """
        return {p: np.percentile(data, p) for p in percentiles}
    
    @staticmethod
    def test_normality(data: List[float]) -> Dict[str, Any]:
        """Test if data follows normal distribution.
        
        Args:
            data: List of values
            
        Returns:
            Dictionary with normality test results
        """
        if SCIPY_AVAILABLE:
            # Shapiro-Wilk test
            statistic, p_value = stats.shapiro(data)
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
        else:
            # Simple approximations without scipy
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            n = len(data)
            
            # Simplified skewness
            skewness = np.sum(((data - mean) / std)**3) / n if std > 0 else 0
            
            # Simplified kurtosis (excess kurtosis)
            kurtosis = (np.sum(((data - mean) / std)**4) / n - 3) if std > 0 else 0
            
            # Very rough normality check
            statistic = abs(skewness) + abs(kurtosis)
            p_value = 0.1 if statistic < 2 else 0.01
        
        return {
            'shapiro_statistic': statistic,
            'shapiro_p_value': p_value,
            'is_normal': p_value > 0.05,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'skewed': abs(skewness) > 1,
            'heavy_tailed': kurtosis > 3
        }