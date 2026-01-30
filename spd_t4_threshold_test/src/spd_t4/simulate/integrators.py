import numpy as np


def euler_maruyama_step(state, drift, dt, sigma, rng):
    noise = rng.normal(scale=np.sqrt(dt), size=state.shape)
    return state + drift * dt + sigma * noise
