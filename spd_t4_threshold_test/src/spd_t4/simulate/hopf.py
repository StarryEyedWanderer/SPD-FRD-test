import numpy as np

from spd_t4.simulate.integrators import euler_maruyama_step
from spd_t4.simulate.schedules import linear_mu_schedule


def hopf_drift(state, mu, omega):
    x, y = state
    r2 = x**2 + y**2
    dx = mu * x - omega * y - r2 * x
    dy = omega * x + mu * y - r2 * y
    return np.array([dx, dy])


def simulate_hopf(config):
    steps = config.T
    dt = config.dt
    rng = np.random.default_rng(config.seed)
    mu_series = linear_mu_schedule(config.mu0, config.mu1, steps)
    state = np.zeros(2, dtype=float)
    xs = np.zeros(steps, dtype=float)
    ys = np.zeros(steps, dtype=float)

    for t in range(steps):
        mu = mu_series[t]
        drift = hopf_drift(state, mu, config.omega)
        state = euler_maruyama_step(state, drift, dt, config.sigma, rng)
        xs[t], ys[t] = state

    time = np.arange(steps) * dt
    return {
        "t": time,
        "mu": mu_series,
        "x": xs,
        "y": ys,
    }
