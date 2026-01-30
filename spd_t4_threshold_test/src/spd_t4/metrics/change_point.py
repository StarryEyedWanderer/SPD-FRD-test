import numpy as np
import pandas as pd


def event_time_from_radius(radius, window, threshold):
    series = pd.Series(radius)
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    hits = np.where(rolling_mean >= threshold)[0]
    return int(hits[0]) if len(hits) else None
