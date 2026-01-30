import numpy as np


def baseline_indices(mu_series, fraction, mu_cut=None):
    n = len(mu_series)
    cutoff = int(n * fraction)
    if mu_cut is None:
        return np.arange(cutoff)
    return np.where(mu_series <= mu_cut)[0]


def theta_quantile(eps, indices, quantile):
    baseline = eps[indices]
    return float(np.nanquantile(baseline, quantile))
