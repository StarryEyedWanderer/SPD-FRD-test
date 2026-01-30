import numpy as np


def linear_mu_schedule(mu0, mu1, steps):
    return np.linspace(mu0, mu1, steps)
