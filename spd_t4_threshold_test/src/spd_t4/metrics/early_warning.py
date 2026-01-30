import pandas as pd


def rolling_var(series, window):
    return pd.Series(series).rolling(window=window, min_periods=window).var().to_numpy()


def rolling_ac1(series, window):
    s = pd.Series(series)
    return s.rolling(window=window, min_periods=window).apply(
        lambda w: w.autocorr(lag=1),
        raw=False,
    ).to_numpy()
