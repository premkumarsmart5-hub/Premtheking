"""Candlestick pattern detection (vectorised).

Each pattern column is +1 (bullish), -1 (bearish) or 0. The value at row t
uses only candles t, t-1, t-2 (the pattern completes on candle t).
Definitions follow the common textbook / TA-Lib style rules, simplified.
"""
import numpy as np
import pandas as pd


def _parts(df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    avg_body = body.rolling(10, min_periods=3).mean().replace(0, np.nan)
    return o, h, l, c, body, rng, upper, lower, avg_body


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, body, rng, upper, lower, avg_body = _parts(df)
    bull = c > o
    bear = c < o
    # Short-term trend context: change of close over the 5 previous candles.
    prior = c.shift(1) - c.shift(6)
    downtrend = prior < 0
    uptrend = prior > 0

    o1, c1, h1, l1, b1 = o.shift(1), c.shift(1), h.shift(1), l.shift(1), body.shift(1)
    o2, c2, b2 = o.shift(2), c.shift(2), body.shift(2)
    bull1, bear1 = c1 > o1, c1 < o1
    bull2, bear2 = c2 > o2, c2 < o2

    small = body <= 0.3 * rng
    long_body = body >= 1.2 * avg_body
    doji = body <= 0.1 * rng

    p = {}
    p["doji_flag"] = doji.astype(int)
    p["dragonfly_doji"] = (doji & (lower >= 0.6 * rng) & (upper <= 0.1 * rng)).astype(int)
    p["gravestone_doji"] = -(doji & (upper >= 0.6 * rng) & (lower <= 0.1 * rng)).astype(int)

    hammer_shape = small & (lower >= 2 * body) & (upper <= 0.25 * rng)
    inv_shape = small & (upper >= 2 * body) & (lower <= 0.25 * rng)
    p["hammer"] = (hammer_shape & downtrend).astype(int)
    p["hanging_man"] = -(hammer_shape & uptrend).astype(int)
    p["inverted_hammer"] = (inv_shape & downtrend).astype(int)
    p["shooting_star"] = -(inv_shape & uptrend).astype(int)

    maru = (body >= 0.9 * rng) & long_body
    p["marubozu"] = np.where(maru & bull, 1, np.where(maru & bear, -1, 0))
    p["spinning_top"] = (small & ~doji & (upper > body) & (lower > body)).astype(int)

    p["engulfing"] = np.where(
        bear1 & bull & (o <= c1) & (c >= o1) & (body > b1), 1,
        np.where(bull1 & bear & (o >= c1) & (c <= o1) & (body > b1), -1, 0))
    p["harami"] = np.where(
        bear1 & bull & (o > c1) & (c < o1) & (b1 > avg_body), 1,
        np.where(bull1 & bear & (o < c1) & (c > o1) & (b1 > avg_body), -1, 0))
    p["piercing_line"] = (bear1 & bull & (o <= c1) & (c > (o1 + c1) / 2) & (c < o1)).astype(int)
    p["dark_cloud_cover"] = -(bull1 & bear & (o >= c1) & (c < (o1 + c1) / 2) & (c > o1)).astype(int)

    star_small = b1 <= 0.3 * (h1 - l1).replace(0, np.nan)
    p["morning_star"] = (bear2 & (b2 > avg_body) & star_small & bull & (c > (o2 + c2) / 2)).astype(int)
    p["evening_star"] = -(bull2 & (b2 > avg_body) & star_small & bear & (c < (o2 + c2) / 2)).astype(int)

    p["three_soldiers_crows"] = np.where(
        bull & bull1 & bull2 & (c > c1) & (c1 > c2) & long_body, 1,
        np.where(bear & bear1 & bear2 & (c < c1) & (c1 < c2) & long_body, -1, 0))

    tol = 0.1 * avg_body
    p["tweezer"] = np.where(
        downtrend & bear1 & bull & ((l - l1).abs() <= tol), 1,
        np.where(uptrend & bull1 & bear & ((h - h1).abs() <= tol), -1, 0))

    harami_up = bear2 & bull1 & (o1 > c2) & (c1 < o2)
    harami_dn = bull2 & bear1 & (o1 < c2) & (c1 > o2)
    p["three_inside"] = np.where(harami_up & bull & (c > o2), 1,
                                 np.where(harami_dn & bear & (c < o2), -1, 0))
    engulf_up = bear2 & bull1 & (o1 <= c2) & (c1 >= o2)
    engulf_dn = bull2 & bear1 & (o1 >= c2) & (c1 <= o2)
    p["three_outside"] = np.where(engulf_up & bull & (c > c1), 1,
                                  np.where(engulf_dn & bear & (c < c1), -1, 0))
    p["kicker"] = np.where(bear1 & bull & (o > o1) & long_body, 1,
                           np.where(bull1 & bear & (o < o1) & long_body, -1, 0))
    p["belt_hold"] = np.where(bull & (lower <= 0.05 * rng) & long_body & downtrend, 1,
                              np.where(bear & (upper <= 0.05 * rng) & long_body & uptrend, -1, 0))

    out = pd.DataFrame(p, index=df.index).fillna(0).astype(int)
    out["pattern_score"] = out.drop(columns=["doji_flag", "spinning_top"]).sum(axis=1)
    return out


PATTERN_NAMES = [
    "dragonfly_doji", "gravestone_doji", "hammer", "hanging_man", "inverted_hammer",
    "shooting_star", "marubozu", "engulfing", "harami", "piercing_line",
    "dark_cloud_cover", "morning_star", "evening_star", "three_soldiers_crows",
    "tweezer", "three_inside", "three_outside", "kicker", "belt_hold",
]


def describe_latest(pats: pd.DataFrame) -> list:
    """Human-readable list of patterns active on the last candle."""
    row = pats.iloc[-1]
    found = []
    for name in PATTERN_NAMES:
        v = int(row[name])
        if v:
            found.append(f"{name.replace('_', ' ')} ({'bull' if v > 0 else 'bear'})")
    if row["doji_flag"]:
        found.append("doji")
    return found
