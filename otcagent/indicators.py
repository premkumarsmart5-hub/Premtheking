"""Technical indicators implemented with pandas/numpy only (Termux friendly).

Every function takes a candle DataFrame with columns open, high, low, close
and returns Series aligned to it. Nothing here looks into the future: each
value at row t uses only rows <= t.
"""
import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(s: pd.Series, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(s: pd.Series, n=20, k=2.0):
    mid = sma(s, n)
    sd = s.rolling(n).std()
    upper, lower = mid + k * sd, mid - k * sd
    pct_b = (s - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / mid
    return pct_b, width


def stochastic(df: pd.DataFrame, n=14, d=3):
    lo = df["low"].rolling(n).min()
    hi = df["high"].rolling(n).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(d).mean()


def williams_r(df: pd.DataFrame, n=14) -> pd.Series:
    hi = df["high"].rolling(n).max()
    lo = df["low"].rolling(n).min()
    return -100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)


def cci(df: pd.DataFrame, n=20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def adx(df: pd.DataFrame, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(df), n)
    pdi = 100 * rma(pd.Series(plus_dm, index=df.index), n) / tr.replace(0, np.nan)
    mdi = 100 * rma(pd.Series(minus_dm, index=df.index), n) / tr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx.fillna(0), n), pdi, mdi


def supertrend_dir(df: pd.DataFrame, n=10, mult=3.0) -> pd.Series:
    """+1 when price is above the SuperTrend line, -1 when below."""
    a = atr(df, n).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).values
    close = df["close"].values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb = ub.copy(), lb.copy()
    direction = np.ones(len(df))
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
        if close[i] > fub[i - 1]:
            direction[i] = 1
        elif close[i] < flb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return pd.Series(direction, index=df.index)


def heikin_ashi_dir(df: pd.DataFrame) -> pd.Series:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    vals = ha_open.to_numpy(copy=True)
    hc = ha_close.to_numpy()
    for i in range(1, len(df)):
        vals[i] = (vals[i - 1] + hc[i - 1]) / 2
    return np.sign(ha_close - pd.Series(vals, index=df.index))


def donchian_pos(df: pd.DataFrame, n=20) -> pd.Series:
    hi = df["high"].rolling(n).max()
    lo = df["low"].rolling(n).min()
    return (df["close"] - lo) / (hi - lo).replace(0, np.nan)


def ichimoku(df: pd.DataFrame, tenkan=9, kijun=26, senkou_b=52):
    """Tenkan, Kijun, and the cloud as it stands at candle t (spans built 26 candles ago)."""
    def mid(n):
        return (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2
    t, k = mid(tenkan), mid(kijun)
    span_a = ((t + k) / 2).shift(kijun)
    span_b = mid(senkou_b).shift(kijun)
    return t, k, span_a, span_b


def parabolic_sar(df: pd.DataFrame, step=0.02, max_step=0.2) -> pd.Series:
    """Direction of Parabolic SAR: +1 = SAR below price (uptrend), -1 = above."""
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    d = np.ones(n)
    if n < 3:
        return pd.Series(d, index=df.index)
    up, af, ep, sar = True, step, hi[0], lo[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, lo[i - 1], lo[i - 2] if i > 1 else lo[i - 1])
            if lo[i] < sar:
                up, sar, ep, af = False, ep, lo[i], step
            elif hi[i] > ep:
                ep, af = hi[i], min(af + step, max_step)
        else:
            sar = max(sar, hi[i - 1], hi[i - 2] if i > 1 else hi[i - 1])
            if hi[i] > sar:
                up, sar, ep, af = True, ep, hi[i], step
            elif lo[i] < ep:
                ep, af = lo[i], min(af + step, max_step)
        d[i] = 1 if up else -1
    return pd.Series(d, index=df.index)


def keltner(df: pd.DataFrame, n=20, atr_n=10, mult=2.0):
    mid = ema(df["close"], n)
    a = atr(df, atr_n)
    return mid + mult * a, mid - mult * a
