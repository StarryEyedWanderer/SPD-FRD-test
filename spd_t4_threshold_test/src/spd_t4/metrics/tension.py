import numpy as np


def sse_tension(next_states, predicted_next):
    diff = next_states - predicted_next
    return np.sum(diff**2, axis=1)
