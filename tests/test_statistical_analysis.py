"""Tests for statistical analysis module."""

import pytest
import numpy as np
from src.analysis.statistical_analysis import SimulationStatistics, ConfidenceInterval, StatisticalTest


def test_bootstrap_ci():
    """Test bootstrap confidence interval calculation."""
    np.random.seed(42)
    data = np.random.normal(100, 10, 1000)
    
    ci = SimulationStatistics.bootstrap_ci(data, confidence_level=0.95, n_bootstrap=1000, seed=42)
    
    assert isinstance(ci, ConfidenceInterval)
    assert 95 < ci.mean < 105  # Should be near 100
    assert ci.lower < ci.mean < ci.upper
    assert ci.confidence_level == 0.95
    assert ci.n_samples == 1000


def test_bootstrap_ci_median():
    """Test bootstrap CI for median."""
    np.random.seed(42)
    data = np.random.exponential(5, 1000)
    
    ci = SimulationStatistics.bootstrap_ci(
        data, 
        statistic_func=np.median,
        confidence_level=0.90,
        n_bootstrap=1000,
        seed=42
    )
    
    assert ci.mean > 0
    assert ci.lower < ci.mean < ci.upper


def test_compare_distributions_mann_whitney():
    """Test Mann-Whitney U test for comparing distributions."""
    np.random.seed(42)
    data1 = np.random.normal(100, 10, 100)
    data2 = np.random.normal(110, 10, 100)  # Different mean
    
    test = SimulationStatistics.compare_distributions(data1, data2, test='mann-whitney')
    
    assert isinstance(test, StatisticalTest)
    assert test.reject_null  # Should detect difference
    assert 0 <= test.p_value <= 1


def test_compare_distributions_same():
    """Test that identical distributions are not flagged as different."""
    np.random.seed(42)
    data1 = np.random.normal(100, 10, 100)
    data2 = np.random.normal(100, 10, 100)  # Same distribution
    
    test = SimulationStatistics.compare_distributions(data1, data2, test='mann-whitney')
    
    # Should not reject null (though there's always a small chance)
    # Use p_value check instead of reject_null
    assert test.p_value > 0.01  # Very unlikely to get p < 0.01 for same distribution


def test_bonferroni_correction():
    """Test Bonferroni correction for multiple comparisons."""
    p_values = [0.01, 0.03, 0.001, 0.10]
    reject, adjusted_alpha = SimulationStatistics.bonferroni_correction(p_values, alpha=0.05)
    
    assert adjusted_alpha == 0.05 / 4
    assert len(reject) == 4
    assert reject[2]  # 0.001 < 0.0125, should reject
    assert not reject[3]  # 0.10 > 0.0125, should not reject


def test_propagate_uncertainty():
    """Test uncertainty propagation through calculation."""
    # Test with simple linear function: z = x + y
    values = {
        'x': (10.0, 1.0),  # mean=10, std=1
        'y': (20.0, 2.0)   # mean=20, std=2
    }
    
    def add(x, y):
        return x + y
    
    np.random.seed(42)
    mean_result, std_result = SimulationStatistics.propagate_uncertainty(values, add)
    
    # For addition, mean(x+y) = mean(x) + mean(y)
    assert 29 < mean_result < 31  # Should be ~30
    
    # For independent variables, var(x+y) = var(x) + var(y)
    # std(x+y) = sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.236
    assert 2.0 < std_result < 2.5


def test_goodness_of_fit_normal():
    """Test goodness of fit for normal distribution."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    
    test = SimulationStatistics.goodness_of_fit(data, 'norm')
    
    assert isinstance(test, StatisticalTest)
    # Normal data should not reject normality
    assert not test.reject_null or test.p_value > 0.01


def test_goodness_of_fit_not_normal():
    """Test that non-normal data is detected."""
    np.random.seed(42)
    data = np.random.exponential(1, 1000)  # Clearly not normal
    
    test = SimulationStatistics.goodness_of_fit(data, 'norm')
    
    # Should reject normality for exponential data
    assert test.reject_null


def test_effect_size_cohen_d():
    """Test Cohen's d effect size calculation."""
    np.random.seed(42)
    data1 = np.random.normal(100, 10, 100)
    data2 = np.random.normal(110, 10, 100)  # 1 SD difference
    
    d = SimulationStatistics.compute_effect_size(data1, data2, method='cohen_d')
    
    # Should be approximately -1.0 (negative because data2 > data1)
    assert -1.3 < d < -0.7


def test_effect_size_hedges_g():
    """Test Hedges' g effect size calculation."""
    np.random.seed(42)
    data1 = np.random.normal(100, 10, 50)
    data2 = np.random.normal(105, 10, 50)  # 0.5 SD difference
    
    g = SimulationStatistics.compute_effect_size(data1, data2, method='hedges_g')
    
    # Nominal effect is -0.5, but this particular seeded n=50 sample realizes a
    # larger negative effect (~-0.82); assert a moderate-to-large negative g.
    assert -1.0 < g < -0.2


def test_effect_size_cliff_delta():
    """Test Cliff's delta effect size calculation."""
    np.random.seed(42)
    data1 = np.random.normal(100, 10, 100)
    data2 = np.random.normal(110, 10, 100)
    
    delta = SimulationStatistics.compute_effect_size(data1, data2, method='cliff_delta')
    
    # Cliff's delta ranges from -1 to 1
    assert -1.0 <= delta <= 1.0
    # data2 > data1, so delta should be negative
    assert delta < 0


def test_sample_size_power_analysis():
    """Test sample size calculation for desired power."""
    # Medium effect size (0.5), standard alpha and power
    n = SimulationStatistics.sample_size_power_analysis(
        effect_size=0.5,
        alpha=0.05,
        power=0.80
    )
    
    assert n > 0
    # For Cohen's d = 0.5, power = 0.80, should need ~64 per group
    assert 50 < n < 80


def test_sample_size_large_effect():
    """Test that large effect sizes require smaller samples."""
    n_small = SimulationStatistics.sample_size_power_analysis(
        effect_size=0.2,  # Small effect
        power=0.80
    )
    
    n_large = SimulationStatistics.sample_size_power_analysis(
        effect_size=0.8,  # Large effect
        power=0.80
    )
    
    # Larger effect should require fewer samples
    assert n_large < n_small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
