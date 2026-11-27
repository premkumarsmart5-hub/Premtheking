"""Feature engineering: indicators + candlestick patterns + candle history."""
import numpy as np
import pandas as pd

from . import indicators as ind
from .patterns import detect_patterns

HORIZONS = (1, 2, 3, 5)  # expiry in minutes == number of 1-minute candles
WARMUP = 60              # rows needed before features are valid
N_LAGS = 10


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame aligned with df. Row t uses data up to candle t only."""
    c = df["close"]
    a = ind.atr(df, 14).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)

    # --- Trend / momentum indicators (normalised by ATR or bounded) ---
    for n in (5, 9, 21, 50):
        f[f"dist_ema{n}"] = (c - ind.ema(c, n)) / a
    f["ema9_21"] = (ind.ema(c, 9) - ind.ema(c, 21)) / a
    f["ema_slope9"] = ind.ema(c, 9).diff(3) / a
    m, s, hist = ind.macd(c)
    f["macd"] = m / a
    f["macd_sig"] = s / a
    f["macd_hist"] = hist / a
    f["macd_hist_chg"] = hist.diff() / a
    for n in (7, 14):
        f[f"rsi{n}"] = ind.rsi(c, n)
    f["rsi14_chg"] = f["rsi14"].diff()
    k, d = ind.stochastic(df)
    f["stoch_k"], f["stoch_d"] = k, d
    f["stoch_kd"] = k - d
    f["willr"] = ind.williams_r(df)
    f["cci"] = ind.cci(df)
    adx, pdi, mdi = ind.adx(df)
    f["adx"], f["di_diff"] = adx, pdi - mdi
    f["roc3"] = c.pct_change(3) * 1e4
    f["roc10"] = c.pct_change(10) * 1e4
    f["supertrend"] = ind.supertrend_dir(df)
    f["heikin_ashi"] = ind.heikin_ashi_dir(df)

    # --- Volatility / mean reversion ---
    pb, bw = ind.bollinger(c)
    f["bb_pctb"], f["bb_width"] = pb, bw * 1e4
    f["atr_rel"] = a / c * 1e4
    f["atr_ratio"] = a / ind.atr(df, 50).replace(0, np.nan)
    f["donchian"] = ind.donchian_pos(df)
    f["zscore20"] = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)

    # --- Candle anatomy + past history (the last N candles) ---
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = df["close"] - df["open"]
    f["body_rel"] = body / a
    f["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    f["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
    ret = c.diff() / a
    for i in range(N_LAGS):
        f[f"ret_lag{i}"] = ret.shift(i)
    for i in range(1, 4):
        f[f"body_lag{i}"] = f["body_rel"].shift(i)
    color = np.sign(body)
    # consecutive same-colour candles (signed)
    grp = (color != color.shift()).cumsum()
    f["streak"] = color * color.groupby(grp).cumcount().add(1)
    f["green_ratio10"] = (color > 0).rolling(10).mean()
    f["dist_high20"] = (df["high"].rolling(20).max() - c) / a
    f["dist_low20"] = (c - df["low"].rolling(20).min()) / a

    # --- Time features (OTC price generators can have intraday rhythms) ---
    if isinstance(df.index, pd.DatetimeIndex):
        mins = df.index.hour * 60 + df.index.minute
        f["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
        f["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
        f["minute_mod5"] = df.index.minute % 5

    # --- Candlestick patterns ---
    pats = detect_patterns(df)
    f = pd.concat([f, pats.add_prefix("pat_")], axis=1)
    f["pat_score_3"] = pats["pattern_score"].rolling(3).sum()

    return f.replace([np.inf, -np.inf], np.nan)


def make_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    """1 = CALL wins (close h candles later is higher), 0 = PUT wins, NaN = tie/unknown.

    Entry is at the close of candle t (that is when the signal is shown), expiry
    is at the close of candle t+h.
    """
    fut = df["close"].shift(-horizon)
    lab = pd.Series(np.where(fut > df["close"], 1.0, np.where(fut < df["close"], 0.0, np.nan)),
                    index=df.index)
    lab[fut.isna()] = np.nan
    return lab


def sequence_features(f: pd.DataFrame) -> pd.DataFrame:
    """The raw shape of the last N candles, used by the history-matching (kNN) model."""
    cols = [f"ret_lag{i}" for i in range(N_LAGS)] + ["body_rel", "upper_wick", "lower_wick"]
    return f[cols]
