import numpy as np


def fit_linear_map(states, next_states):
    ones = np.ones((states.shape[0], 1))
    design = np.hstack([states, ones])
    coeffs, *_ = np.linalg.lstsq(design, next_states, rcond=None)
    A = coeffs[:2, :].T
    c = coeffs[2, :]
    return A, c


def predict(A, c, state):
    return A @ state + c
