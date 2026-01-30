import numpy as np

from spd_t4.metrics.thresholds import baseline_indices, theta_quantile


def test_threshold_quantile():
    mu = np.linspace(-1, 1, 10)
    eps = np.arange(10) * 0.1
    idx = baseline_indices(mu, 0.5)
    assert np.array_equal(idx, np.arange(5))
    theta = theta_quantile(eps, idx, 0.8)
    expected = np.quantile(eps[idx], 0.8)
    assert np.isclose(theta, expected)
