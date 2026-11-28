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


def channel_bounce(df, window=30, direction="up", touch_tol=0.15, min_touches=2,
                   mode="bounce"):
    """Trend-channel bounce (your drawing).

    The channel is built from the `window` candles BEFORE candle t:
      * a least-squares trend line through the closes
      * upper line = parallel line through the highest high, lower = through the lowest low
      * a valid channel needs >= `min_touches` swing highs near the upper line and
        swing lows near the lower line (price zig-zagging between them)
      * direction: "up" = only rising channels (as drawn), "down" = falling, "any"
    Candle t reaches the top zone of the channel (within touch_tol of the upper line)
    and closes red, still inside -> PUT. Reaches the bottom zone and closes green -> CALL.

    mode="breakout": candle t CLOSES above the upper line -> CALL, closes below the
    lower line -> PUT (trade in the direction of the break).
    direction="flat": horizontal support/resistance levels (slope forced to 0); for a
    breakout only the side being broken needs the touches (e.g. a flat top with
    rising lows below it, as in an ascending triangle).
    """
    from numpy.lib.stride_tricks import sliding_window_view as win
    n = len(df)
    sig = np.zeros(n, dtype=int)
    if n <= window + 2:
        return pd.Series(sig, index=df.index)
    hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    # swing points (needs the next candle, so the window's last candle is never a swing)
    sh = np.zeros(n, bool)
    sl = np.zeros(n, bool)
    sh[1:-1] = (hi[1:-1] >= hi[:-2]) & (hi[1:-1] >= hi[2:])
    sl[1:-1] = (lo[1:-1] <= lo[:-2]) & (lo[1:-1] <= lo[2:])
    # windows ending at t-1 for t = window .. n-1
    Wc, Wh, Wl = win(cl[:-1], window), win(hi[:-1], window), win(lo[:-1], window)
    Wsh, Wsl = win(sh[:-1], window).copy(), win(sl[:-1], window).copy()
    Wsh[:, -1] = False
    Wsl[:, -1] = False
    x = np.arange(window, dtype=float)
    xm = x.mean()
    slope = ((Wc - Wc.mean(1, keepdims=True)) * (x - xm)).sum(1) / ((x - xm) ** 2).sum()
    if direction == "flat":
        slope = np.zeros_like(slope)
    icpt = Wc.mean(1) - slope * xm
    line = icpt[:, None] + slope[:, None] * x
    up_off = (Wh - line).max(1)
    lo_off = (Wl - line).min(1)
    width = up_off - lo_off
    near_up = Wsh & ((Wh - line) >= (up_off - touch_tol * width)[:, None])
    near_lo = Wsl & ((Wl - line) <= (lo_off + touch_tol * width)[:, None])
    ok_up = near_up.sum(1) >= min_touches
    ok_lo = near_lo.sum(1) >= min_touches
    valid = ok_up & ok_lo & (width > 0)
    # slope must be meaningful: channel rises/falls by >= 25% of its width over the window
    rise = slope * window / np.where(width > 0, width, np.nan)
    if direction == "up":
        valid &= rise >= 0.25
    elif direction == "down":
        valid &= rise <= -0.25
    proj = icpt + slope * window  # trend line projected to candle t
    U, L = proj + up_off, proj + lo_off
    t = np.arange(window, n)
    if mode == "breakout":
        if direction == "flat":  # each side only needs its own touches
            call = ok_up & (width > 0) & (cl[t] > U)
            put = ok_lo & (width > 0) & (cl[t] < L)
        else:
            call = valid & (cl[t] > U)
            put = valid & (cl[t] < L)
        sig[t] = np.where(call & ~put, 1, np.where(put & ~call, -1, 0))
        return pd.Series(sig, index=df.index)
    op = df["open"].to_numpy(float)
    band = touch_tol * width
    # reaches the line zone and shows rejection: red candle at the top, green at the bottom
    put = valid & (hi[t] >= U - band) & (cl[t] < op[t]) & (cl[t] < U)
    call = valid & (lo[t] <= L + band) & (cl[t] > op[t]) & (cl[t] > L)
    sig[t] = np.where(put & ~call, -1, np.where(call & ~put, 1, 0))
    return pd.Series(sig, index=df.index)


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
    "channel_up": (lambda df: channel_bounce(df, 30, "up"),
                   "YOUR IDEA: rising channel (30 candles), touch top = PUT, touch bottom = CALL"),
    "channel_up_50": (lambda df: channel_bounce(df, 50, "up"),
                      "YOUR IDEA, longer channel (50 candles)"),
    "channel_any": (lambda df: channel_bounce(df, 30, "any"),
                    "YOUR IDEA on rising, falling or flat channels (30 candles)"),
    "breakout_channel": (lambda df: channel_bounce(df, 30, "any", mode="breakout"),
                         "YOUR IDEA 2: close above channel = CALL, below = PUT (30 candles)"),
    "breakout_channel_50": (lambda df: channel_bounce(df, 50, "any", mode="breakout"),
                            "YOUR IDEA 2, longer channel (50 candles)"),
    "breakout_level": (lambda df: channel_bounce(df, 30, "flat", mode="breakout"),
                       "YOUR IDEA 2: break of flat resistance (2+ touches) = CALL, support = PUT"),
    "breakout_level_50": (lambda df: channel_bounce(df, 50, "flat", mode="breakout"),
                          "YOUR IDEA 2, flat levels over 50 candles"),
    # Fade = trade AGAINST the break (bet on a false breakout / fakeout).
    "fade_channel": (lambda df: -channel_bounce(df, 30, "any", mode="breakout"),
                     "FADE: close above channel = PUT, below = CALL (30 candles)"),
    "fade_channel_50": (lambda df: -channel_bounce(df, 50, "any", mode="breakout"),
                        "FADE, channel over 50 candles"),
    "fade_level": (lambda df: -channel_bounce(df, 30, "flat", mode="breakout"),
                   "FADE: break above flat resistance = PUT, below support = CALL"),
    "fade_level_50": (lambda df: -channel_bounce(df, 50, "flat", mode="breakout"),
                      "FADE, flat levels over 50 candles"),
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
