"""Rule-based CALL/PUT strategies and an honest tester for them.

Each strategy is a function df -> Series of +1 (CALL), -1 (PUT) or 0 (no trade)
for every candle. The signal on candle t uses only candles <= t, and the trade
is entered at the close of candle t.

To add your own idea: write a function like the ones below and add it to
STRATEGIES at the bottom with a one-line description.
"""
import numpy as np
import pandas as pd

from . import indicators as ind
from .backtest import breakeven, wilson_lower
from .features import make_labels
from .patterns import detect_patterns


def _sig(call, put) -> pd.Series:
    call = pd.Series(call).fillna(False).astype(bool)
    put = pd.Series(put).fillna(False).astype(bool)
    return pd.Series(np.where(call & ~put, 1, np.where(put & ~call, -1, 0)), index=call.index)


def _cross_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_dn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def _color(df):
    return np.sign(df["close"] - df["open"])


# ----------------------------------------------------------------- strategies

def rsi_bb_reversal(df):
    c = df["close"]
    r = ind.rsi(c, 14)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    return _sig((c < mid - 2 * sd) & (r < 30), (c > mid + 2 * sd) & (r > 70))


def ema_cross(df):
    c = df["close"]
    f, s = ind.ema(c, 9), ind.ema(c, 21)
    return _sig(_cross_up(f, s), _cross_dn(f, s))


def three_candle_reversal(df):
    col = _color(df)
    up3 = (col == 1) & (col.shift(1) == 1) & (col.shift(2) == 1)
    dn3 = (col == -1) & (col.shift(1) == -1) & (col.shift(2) == -1)
    return _sig(dn3, up3)


def follow_candle(df):
    col = _color(df)
    body = (df["close"] - df["open"]).abs()
    strong = body > body.rolling(20).mean() * 1.5
    return _sig((col == 1) & strong, (col == -1) & strong)


def engulfing_with_trend(df):
    p = detect_patterns(df)["engulfing"]
    trend = df["close"] > ind.ema(df["close"], 50)
    return _sig((p == 1) & trend, (p == -1) & ~trend)


def stoch_cross(df):
    k, d = ind.stochastic(df)
    return _sig(_cross_up(k, d) & (k < 20), _cross_dn(k, d) & (k > 80))


def macd_cross(df):
    m, s, _ = ind.macd(df["close"])
    return _sig(_cross_up(m, s), _cross_dn(m, s))


def pin_bar_at_band(df):
    p = detect_patterns(df)
    c = df["close"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    bull_pin = (p["hammer"] == 1) | (p["dragonfly_doji"] == 1)
    bear_pin = (p["shooting_star"] == -1) | (p["gravestone_doji"] == -1)
    return _sig(bull_pin & (df["low"] <= mid - 2 * sd), bear_pin & (df["high"] >= mid + 2 * sd))


def supertrend_flip(df):
    st = ind.supertrend_dir(df)
    return _sig((st == 1) & (st.shift(1) == -1), (st == -1) & (st.shift(1) == 1))


def all_patterns_vote(df):
    score = detect_patterns(df)["pattern_score"]
    return _sig(score >= 2, score <= -2)


STRATEGIES = {
    "rsi_bb_reversal": (rsi_bb_reversal, "Close outside Bollinger(20,2) + RSI<30 CALL / RSI>70 PUT"),
    "ema_cross": (ema_cross, "EMA 9 crosses EMA 21: up = CALL, down = PUT"),
    "three_candle_reversal": (three_candle_reversal, "3 same-colour candles in a row -> trade the opposite"),
    "follow_candle": (follow_candle, "Big candle (1.5x avg body) -> trade the same direction"),
    "engulfing_with_trend": (engulfing_with_trend, "Engulfing pattern in the direction of EMA 50 trend"),
    "stoch_cross": (stoch_cross, "Stochastic K crosses D below 20 CALL / above 80 PUT"),
    "macd_cross": (macd_cross, "MACD line crosses signal line"),
    "pin_bar_at_band": (pin_bar_at_band, "Hammer/shooting star touching Bollinger band -> reversal"),
    "supertrend_flip": (supertrend_flip, "SuperTrend(10,3) changes direction"),
    "patterns_vote": (all_patterns_vote, "2+ bullish candlestick patterns CALL / 2+ bearish PUT"),
}


# ----------------------------------------------------------------- tester

def test_strategy(df, signal: pd.Series, horizons, tf: int, payout: float,
                  exam_frac: float = 0.2, min_trades: int = 100, min_exam: int = 30):
    """Win rate of the rule on all history plus a separate check on the newest 20%.
    PASS only if overall win rate beats break-even with 95% confidence AND the
    newest 20% (exam) also beats break-even."""
    be = breakeven(payout)
    exam_start = int(len(df) * (1 - exam_frac))
    pos = np.arange(len(df))
    out = []
    for h in horizons:
        y = make_labels(df, h, tf).to_numpy()
        s = signal.to_numpy()
        traded = (s != 0) & ~np.isnan(y)
        win = traded & (((s == 1) & (y == 1)) | ((s == -1) & (y == 0)))
        n, w = int(traded.sum()), int(win.sum())
        ex = pos >= exam_start
        ne, we = int((traded & ex).sum()), int((win & ex).sum())
        wr = w / n if n else 0.0
        ewr = we / ne if ne else 0.0
        passed = n >= min_trades and wilson_lower(w, n) > be and ne >= min_exam and ewr > be
        out.append({"expiry": f"{h * tf}m", "trades": n, "wins": w, "win_rate": wr,
                    "exam_trades": ne, "exam_win_rate": ewr, "pass": passed})
    return out
