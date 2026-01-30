import numpy as np

from spd_t4.models.linear_predictor import fit_linear_map, predict


def rolling_linear_predictions(states, window):
    n = states.shape[0]
    preds = np.full_like(states, np.nan)
    for idx in range(window, n - 1):
        window_states = states[idx - window : idx]
        window_next = states[idx - window + 1 : idx + 1]
        A, c = fit_linear_map(window_states, window_next)
        preds[idx + 1] = predict(A, c, states[idx])
    return preds
