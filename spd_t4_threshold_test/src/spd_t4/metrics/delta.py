import numpy as np


def find_delta(series, threshold, start, persist):
    if start >= len(series):
        return None
    if persist is None or persist <= 1:
        hits = np.where(series[start:] >= threshold)[0]
        return int(start + hits[0]) if len(hits) else None
    for idx in range(start, len(series) - persist + 1):
        window = series[idx : idx + persist]
        if np.all(window >= threshold):
            return int(idx)
    return None
