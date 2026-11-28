"""Forex feature set for the big agent.

For every candle (using only that candle and earlier ones) it computes:
  * the core indicator values (from features.build_features)
  * ~56 candlestick patterns (candlesticks.py)
  * "indicator setups": each classic indicator read the textbook way -> vote +1/-1/0
  * higher-timeframe context (e.g. 15m and 60m trend while trading 1m)
  * trading session (Asia / London / New York / overlap) and day of week
  * classic daily pivot points (P, R1, S1, R2, S2) from the previous UTC day
  * market regime (trend strength, volatility rank, efficiency ratio)
"""
import numpy as np
import pandas as pd

from . import indicators as ind
from .candlesticks import detect_all
from .features import build_features

# indicator name -> (setting, what the vote means), used in reports and the README
SETUPS = {
    "rsi_cross": "RSI(14): crosses up through 30 = buy, down through 70 = sell",
    "rsi_zone": "RSI(14): below 30 oversold = buy, above 70 overbought = sell",
    "macd_cross": "MACD(12,26,9): line crosses signal up = buy, down = sell",
    "macd_zero": "MACD(12,26,9): above zero = bullish, below = bearish",
    "bb_reentry": "Bollinger(20,2): close back inside from below = buy, from above = sell",
    "stoch_cross": "Stochastic(14,3,3): %K crosses %D below 20 = buy, above 80 = sell",
    "adx_trend": "ADX(14) > 25: +DI > -DI = bullish trend, -DI > +DI = bearish",
    "ema_cross": "EMA 9 / EMA 21 cross",
    "ema_trend": "EMA 50 vs EMA 200 with price on the same side",
    "ichimoku_tk": "Ichimoku(9,26,52): Tenkan crosses Kijun",
    "ichimoku_cloud": "Ichimoku: price above cloud = bullish, below = bearish",
    "psar": "Parabolic SAR(0.02, 0.2): dots below price = bullish",
    "cci": "CCI(20): crosses up -100 = buy, down +100 = sell",
    "williams": "Williams %R(14): crosses up -80 = buy, down -20 = sell",
    "supertrend": "SuperTrend(10,3) direction",
    "keltner": "Keltner(20, 2xATR10): close back inside channel = reversal",
    "donchian": "Donchian(20) breakout of previous high / low",
    "heikin_ashi": "Heikin-Ashi colour change",
    "pivot": "Daily pivots: rejection at S1 = buy, at R1 = sell",
}
TREND_VOTES = ["macd_zero", "adx_trend", "ema_trend", "ichimoku_cloud", "psar", "supertrend"]
REVERSAL_VOTES = ["rsi_cross", "rsi_zone", "bb_reentry", "stoch_cross", "cci", "williams",
                  "keltner", "pivot"]


def _cross_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_dn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def _v(buy, sell):
    buy = pd.Series(buy).fillna(False).astype(bool)
    sell = pd.Series(sell).fillna(False).astype(bool)
    return pd.Series(np.where(buy & ~sell, 1, np.where(sell & ~buy, -1, 0)), index=buy.index)


def daily_pivots(df: pd.DataFrame) -> pd.DataFrame:
    day = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna().shift(1)
    P = (day["high"] + day["low"] + day["close"]) / 3
    piv = pd.DataFrame({"P": P, "R1": 2 * P - day["low"], "S1": 2 * P - day["high"],
                        "R2": P + (day["high"] - day["low"]), "S2": P - (day["high"] - day["low"])})
    keys = df.index.floor("1D")
    return piv.reindex(keys).set_axis(df.index)


def indicator_votes(df: pd.DataFrame, piv: pd.DataFrame = None) -> pd.DataFrame:
    c = df["close"]
    v = {}
    r = ind.rsi(c, 14)
    v["rsi_cross"] = _v(_cross_up(r, pd.Series(30, index=c.index)), _cross_dn(r, pd.Series(70, index=c.index)))
    v["rsi_zone"] = _v(r < 30, r > 70)
    m, s, _ = ind.macd(c)
    v["macd_cross"] = _v(_cross_up(m, s), _cross_dn(m, s))
    v["macd_zero"] = _v(m > 0, m < 0)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    lo, hi = mid - 2 * sd, mid + 2 * sd
    v["bb_reentry"] = _v((c.shift(1) < lo.shift(1)) & (c > lo), (c.shift(1) > hi.shift(1)) & (c < hi))
    k, d = ind.stochastic(df, 14, 3)
    k = k.rolling(3).mean()  # slowing 3
    d = k.rolling(3).mean()
    v["stoch_cross"] = _v(_cross_up(k, d) & (k < 20), _cross_dn(k, d) & (k > 80))
    adx, pdi, mdi = ind.adx(df, 14)
    v["adx_trend"] = _v((adx > 25) & (pdi > mdi), (adx > 25) & (mdi > pdi))
    e9, e21, e50, e200 = (ind.ema(c, n) for n in (9, 21, 50, 200))
    v["ema_cross"] = _v(_cross_up(e9, e21), _cross_dn(e9, e21))
    v["ema_trend"] = _v((e50 > e200) & (c > e50), (e50 < e200) & (c < e50))
    t, kj, sa, sb = ind.ichimoku(df)
    v["ichimoku_tk"] = _v(_cross_up(t, kj), _cross_dn(t, kj))
    top = pd.concat([sa, sb], axis=1).max(axis=1)
    bot = pd.concat([sa, sb], axis=1).min(axis=1)
    v["ichimoku_cloud"] = _v(c > top, c < bot)
    v["psar"] = ind.parabolic_sar(df).astype(int)
    cc = ind.cci(df, 20)
    v["cci"] = _v(_cross_up(cc, pd.Series(-100, index=c.index)), _cross_dn(cc, pd.Series(100, index=c.index)))
    w = ind.williams_r(df, 14)
    v["williams"] = _v(_cross_up(w, pd.Series(-80, index=c.index)), _cross_dn(w, pd.Series(-20, index=c.index)))
    v["supertrend"] = ind.supertrend_dir(df).astype(int)
    ku, kl = ind.keltner(df)
    v["keltner"] = _v((c.shift(1) < kl.shift(1)) & (c > kl), (c.shift(1) > ku.shift(1)) & (c < ku))
    dh = df["high"].rolling(20).max().shift(1)
    dl = df["low"].rolling(20).min().shift(1)
    v["donchian"] = _v(c > dh, c < dl)
    ha = ind.heikin_ashi_dir(df)
    v["heikin_ashi"] = _v((ha > 0) & (ha.shift(1) <= 0), (ha < 0) & (ha.shift(1) >= 0))
    if piv is None:
        piv = daily_pivots(df)
    v["pivot"] = _v((df["low"] <= piv["S1"]) & (c > piv["S1"]), (df["high"] >= piv["R1"]) & (c < piv["R1"]))
    out = pd.DataFrame(v, index=df.index).fillna(0).astype(np.int8)
    out["votes_trend"] = out[TREND_VOTES].sum(axis=1)
    out["votes_reversal"] = out[REVERSAL_VOTES].sum(axis=1)
    out["votes_total"] = out[list(SETUPS)].sum(axis=1)
    return out


def htf_features(df: pd.DataFrame, tf: int, htf: int) -> pd.DataFrame:
    """Indicators on a higher timeframe, using only HTF candles that have CLOSED."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    h = df.resample(f"{htf}min", label="right", closed="left").agg(agg).dropna()
    c = h["close"]
    a = ind.atr(h, 14).replace(0, np.nan)
    e21 = ind.ema(c, 21)
    adx, pdi, mdi = ind.adx(h, 14)
    _, _, hist = ind.macd(c)
    f = pd.DataFrame({
        "trend": np.sign(e21.diff(3)),
        "dist_ema21": (c - e21) / a,
        "rsi": ind.rsi(c, 14),
        "adx": adx,
        "di": pdi - mdi,
        "macd_hist": hist / a,
        "supertrend": ind.supertrend_dir(h),
    }, index=h.index)
    # an LTF candle opened at t closes at t+tf; it may use HTF candles labelled <= t+tf
    keys = df.index + pd.Timedelta(minutes=tf)
    out = f.reindex(keys, method="ffill").set_axis(df.index)
    return out.add_prefix(f"htf{htf}_")


def session_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    hr = idx.hour + idx.minute / 60
    return pd.DataFrame({
        "s_asia": ((hr >= 0) & (hr < 7)).astype(np.int8),
        "s_london": ((hr >= 7) & (hr < 16)).astype(np.int8),
        "s_newyork": ((hr >= 12) & (hr < 21)).astype(np.int8),
        "s_overlap": ((hr >= 12) & (hr < 16)).astype(np.int8),
        "dow": idx.dayofweek,
    }, index=idx)


def session_name(ts) -> str:
    hr = ts.hour + ts.minute / 60
    if 12 <= hr < 16:
        return "London-New York overlap"
    if 7 <= hr < 12:
        return "London"
    if 16 <= hr < 21:
        return "New York"
    return "Asia / late"


def htf_for(tf: int):
    return (15, 60) if tf == 1 else (60, 240)


def build_forex_features(df: pd.DataFrame, tf: int = 1) -> pd.DataFrame:
    base = build_features(df)
    base = base[[c for c in base.columns if not c.startswith("pat_")]]
    a = ind.atr(df, 14).replace(0, np.nan)
    c = df["close"]
    parts = [base, detect_all(df).add_prefix("cdl_")]
    piv = daily_pivots(df)
    parts.append(indicator_votes(df, piv).add_prefix("vote_"))
    parts.append(pd.DataFrame({f"pv_{k}": (c - piv[k]) / a for k in piv.columns}, index=df.index))
    for htf in htf_for(tf):
        parts.append(htf_features(df, tf, htf))
    parts.append(session_features(df.index))
    pb, bw = ind.bollinger(c)
    absdiff = c.diff().abs().rolling(20).sum()
    parts.append(pd.DataFrame({
        "rg_bbw_rank": bw.rolling(200, min_periods=50).rank(pct=True),
        "rg_efficiency": (c - c.shift(20)).abs() / absdiff.replace(0, np.nan),
        "rg_trending": (ind.adx(df, 14)[0] > 25).astype(np.int8),
    }, index=df.index))
    f = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    return f.astype(np.float32)
