import numpy as np

from spd_t4.config import SimConfig
from spd_t4.simulate.hopf import simulate_hopf


def test_reproducibility():
    config = SimConfig(seed=123, T=200, dt=0.01)
    series_a = simulate_hopf(config)
    series_b = simulate_hopf(config)
    for key in ["x", "y", "mu"]:
        assert np.allclose(series_a[key], series_b[key])
