"""Full candlestick pattern library (~50 patterns, modelled on TA-Lib's CDL* set).

Every column is +1 (bullish), -1 (bearish) or 0. "Flag" patterns (indecision:
doji, spinning top, ...) are 1 when present. The value on candle t uses only
candles t, t-1, ... t-4: the pattern completes on candle t.

Forex note: 1-5 minute forex candles almost never gap (open == previous close),
so the classic "gap" conditions are checked on candle BODIES (body above / below
the previous body), which is how most forex traders adapt these patterns.
"""
import numpy as np
import pandas as pd

FLAGS = ["doji", "long_legged_doji", "rickshaw_man", "spinning_top", "high_wave"]


def _s(x, k):
    return x.shift(k)


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    bt = np.maximum(o, c)          # body top
    bb = np.minimum(o, c)          # body bottom
    up = h - bt                    # upper shadow
    dn = bb - l                    # lower shadow
    avgb = body.rolling(10, min_periods=5).mean().shift(1)   # average body (TA-Lib style)
    avgr = rng.rolling(10, min_periods=5).mean().shift(1)    # average range
    bull, bear = c > o, c < o
    long_ = body >= avgb
    vlong = body >= 1.5 * avgb
    small = body <= 0.5 * avgb
    doji = body <= 0.1 * avgr
    near = 0.05 * avgr                                       # "equal" tolerance
    trend = c.shift(1) - c.shift(6)
    upt, dnt = trend > 0, trend < 0

    S = lambda x, k=1: _s(x, k)                              # noqa: E731
    p = {}

    def pm(cond_bull, cond_bear):
        return np.where(cond_bull & ~cond_bear, 1, np.where(cond_bear & ~cond_bull, -1, 0))

    # ---------------- single-candle ----------------
    p["doji"] = doji.astype(int)
    lld = doji & (up >= 0.35 * rng) & (dn >= 0.35 * rng)
    p["long_legged_doji"] = lld.astype(int)
    p["rickshaw_man"] = (lld & (((bt + bb) / 2 - (h + l) / 2).abs() <= 0.1 * rng)).astype(int)
    p["dragonfly_doji"] = (doji & (up <= 0.1 * rng) & (dn >= 0.6 * rng)).astype(int)
    p["gravestone_doji"] = -(doji & (dn <= 0.1 * rng) & (up >= 0.6 * rng)).astype(int)
    ham = (body <= 0.35 * rng) & (dn >= 2 * body) & (up <= 0.15 * rng)
    inv = (body <= 0.35 * rng) & (up >= 2 * body) & (dn <= 0.15 * rng)
    p["hammer"] = (ham & dnt).astype(int)
    p["hanging_man"] = -(ham & upt).astype(int)
    p["takuri"] = (ham & (dn >= 3 * body) & (up <= 0.05 * rng) & dnt).astype(int)
    p["inverted_hammer"] = (inv & dnt).astype(int)
    p["shooting_star"] = -(inv & upt).astype(int)
    maru = long_ & (up <= 0.05 * rng) & (dn <= 0.05 * rng)
    p["marubozu"] = pm(maru & bull, maru & bear)
    p["closing_marubozu"] = pm(long_ & bull & (up <= 0.05 * rng), long_ & bear & (dn <= 0.05 * rng))
    st = (body <= 0.3 * rng) & (up > body) & (dn > body) & ~doji
    p["spinning_top"] = st.astype(int)
    p["high_wave"] = ((body <= 0.3 * rng) & (up >= 3 * body) & (dn >= 3 * body) & ~doji).astype(int)
    ll = vlong & (up <= 0.25 * rng) & (dn <= 0.25 * rng)
    p["long_line"] = pm(ll & bull, ll & bear)
    sl = small & ~doji & (up <= 0.25 * rng) & (dn <= 0.25 * rng)
    p["short_line"] = pm(sl & bull, sl & bear)
    p["belt_hold"] = pm(bull & (dn <= 0.05 * rng) & long_ & dnt, bear & (up <= 0.05 * rng) & long_ & upt)

    # ---------------- two-candle ----------------
    o1, h1, l1, c1, b1 = S(o), S(h), S(l), S(c), S(body)
    bt1, bb1 = S(bt), S(bb)
    bull1, bear1, long1 = S(bull), S(bear), S(long_)
    p["engulfing"] = pm(bear1 & bull & (o <= c1) & (c >= o1) & (body > b1),
                        bull1 & bear & (o >= c1) & (c <= o1) & (body > b1))
    inside = (bt <= bt1) & (bb >= bb1) & (body < b1)  # body inside previous body
    p["harami"] = pm(long1 & bear1 & bull & inside, long1 & bull1 & bear & inside)
    p["harami_cross"] = pm(long1 & bear1 & doji & inside, long1 & bull1 & doji & inside)
    mid1 = (o1 + c1) / 2
    p["piercing"] = (long1 & bear1 & bull & (o <= c1) & (c > mid1) & (c < o1)).astype(int)
    p["dark_cloud_cover"] = -(long1 & bull1 & bear & (o >= c1) & (c < mid1) & (c > o1)).astype(int)
    p["doji_star"] = pm(long1 & bear1 & doji & (bt <= bb1), long1 & bull1 & doji & (bb >= bt1))
    maru1 = S(maru)
    p["kicking"] = pm(maru1 & bear1 & maru & bull & (bb >= bt1), maru1 & bull1 & maru & bear & (bt <= bb1))
    p["counterattack"] = pm(long1 & bear1 & long_ & bull & ((c - c1).abs() <= near),
                            long1 & bull1 & long_ & bear & ((c - c1).abs() <= near))
    p["homing_pigeon"] = (dnt & long1 & bear1 & bear & inside).astype(int)
    p["matching_low"] = (dnt & bear1 & bear & ((c - c1).abs() <= near)).astype(int)
    p["matching_high"] = -(upt & bull1 & bull & ((c - c1).abs() <= near)).astype(int)
    p["separating_lines"] = pm(upt & bear1 & bull & long_ & ((o - o1).abs() <= near),
                               dnt & bull1 & bear & long_ & ((o - o1).abs() <= near))
    p["on_neck"] = -(dnt & long1 & bear1 & bull & ((c - l1).abs() <= near)).astype(int)
    p["in_neck"] = -(dnt & long1 & bear1 & bull & (c >= c1) & (c <= c1 + 0.1 * b1)).astype(int)
    p["thrusting"] = -(dnt & long1 & bear1 & bull & (c > c1 + 0.1 * b1) & (c < mid1)).astype(int)
    p["tweezer"] = pm(dnt & bear1 & bull & ((l - l1).abs() <= near),
                      upt & bull1 & bear & ((h - h1).abs() <= near))

    # ---------------- three-candle ----------------
    o2, c2, b2, h2, l2 = S(o, 2), S(c, 2), S(body, 2), S(h, 2), S(l, 2)
    bt2, bb2 = S(bt, 2), S(bb, 2)
    bull2, bear2, long2 = S(bull, 2), S(bear, 2), S(long_, 2)
    doji1 = S(doji)
    small1 = b1 <= 0.3 * (h1 - l1).replace(0, np.nan)
    mid2 = (o2 + c2) / 2
    p["morning_star"] = (long2 & bear2 & small1 & bull & (c > mid2)).astype(int)
    p["evening_star"] = -(long2 & bull2 & small1 & bear & (c < mid2)).astype(int)
    p["morning_doji_star"] = (long2 & bear2 & doji1 & bull & (c > mid2)).astype(int)
    p["evening_doji_star"] = -(long2 & bull2 & doji1 & bear & (c < mid2)).astype(int)
    p["abandoned_baby"] = pm(bear2 & doji1 & (bt1 < bb2) & bull & (bb > bt1),
                             bull2 & doji1 & (bb1 > bt2) & bear & (bt < bb1))
    soldiers = (bull & bull1 & bull2 & (c > c1) & (c1 > c2) & (o >= bb1) & (o <= bt1)
                & (S(o) >= bb2) & (S(o) <= bt2) & (up <= 0.3 * body) & long_ & long1 & long2)
    crows = (bear & bear1 & bear2 & (c < c1) & (c1 < c2) & (o <= bt1) & (o >= bb1)
             & (S(o) <= bt2) & (S(o) >= bb2) & (dn <= 0.3 * body) & long_ & long1 & long2)
    p["three_white_soldiers"] = soldiers.astype(int)
    p["three_black_crows"] = -crows.astype(int)
    p["identical_three_crows"] = -(bear & bear1 & bear2 & long_ & long1 & long2 & (c < c1) & (c1 < c2)
                                   & ((o - c1).abs() <= near)
                                   & ((S(o) - c2).abs() <= near)).astype(int)
    h_up = bear2 & long2 & bull1 & (bt1 <= bt2) & (bb1 >= bb2) & (b1 < b2)
    h_dn = bull2 & long2 & bear1 & (bt1 <= bt2) & (bb1 >= bb2) & (b1 < b2)
    p["three_inside"] = pm(h_up & bull & (c > o2), h_dn & bear & (c < o2))
    e_up = bear2 & bull1 & (o1 <= c2) & (c1 >= o2)
    e_dn = bull2 & bear1 & (o1 >= c2) & (c1 <= o2)
    p["three_outside"] = pm(e_up & bull & (c > c1), e_dn & bear & (c < c1))
    p["two_crows"] = -(long2 & bull2 & bear1 & (bb1 > bt2) & bear & (o < bt1) & (o > bb1)
                       & (c < c2) & (c > o2)).astype(int)
    p["upside_gap_two_crows"] = -(long2 & bull2 & bear1 & (bb1 > bt2) & bear & (o > o1)
                                  & (c < c1) & (c > c2)).astype(int)
    up1 = S(up)
    p["advance_block"] = -(upt & bull & bull1 & bull2 & (c > c1) & (c1 > c2)
                           & (body < b1) & (b1 < b2) & (up > up1)).astype(int)
    p["stalled_pattern"] = -(upt & bull2 & long2 & bull1 & long1 & bull & small
                             & (c > c1) & (o >= c1 - 0.1 * b1)).astype(int)
    p["unique_3_river"] = (long2 & bear2 & bear1 & (bb1 >= bb2) & (l1 < l2) & bull & small
                           & (c < c1)).astype(int)
    p["stick_sandwich"] = (bear2 & bull1 & (c1 > c2) & bear & ((c - c2).abs() <= near)).astype(int)
    p["tasuki_gap"] = pm(bull2 & bull1 & (bb1 > bt2) & bear & (o < c1) & (o > o1) & (c < o1) & (c > c2),
                         bear2 & bear1 & (bt1 < bb2) & bull & (o > c1) & (o < o1) & (c > o1) & (c < c2))
    doji2 = S(doji, 2)
    p["tristar"] = pm(doji2 & doji1 & doji & (bt1 < bb2) & (bb > bt1),
                      doji2 & doji1 & doji & (bb1 > bt2) & (bt < bb1))

    # ---------------- four / five-candle ----------------
    c3, o3 = S(c, 3), S(o, 3)
    bull3, bear3 = S(bull, 3), S(bear, 3)
    p["three_line_strike"] = pm(bull3 & bull2 & bull1 & (c2 > c3) & (c1 > c2) & bear & (o >= c1) & (c < o3),
                                bear3 & bear2 & bear1 & (c2 < c3) & (c1 < c2) & bull & (o <= c1) & (c > o3))
    c4, o4, h4, l4 = S(c, 4), S(o, 4), S(h, 4), S(l, 4)
    long4, bull4, bear4 = S(long_, 4), S(bull, 4), S(bear, 4)
    small3 = S(small, 3)
    smalls = small3 & S(small, 2) & S(small, 1)
    inside4 = ((pd.concat([S(h, 3), h2, h1], axis=1).max(axis=1) <= h4)
               & (pd.concat([S(l, 3), l2, l1], axis=1).min(axis=1) >= l4))
    p["rise_fall_three_methods"] = pm(long4 & bull4 & smalls & inside4 & bull & long_ & (c > c4),
                                      long4 & bear4 & smalls & inside4 & bear & long_ & (c < c4))
    p["ladder_bottom"] = (bear4 & bear3 & bear2 & (c3 < c4) & (c2 < c3) & bear1 & (up1 > b1)
                          & bull & (o > bt1)).astype(int)
    p["breakaway"] = pm(long4 & bear4 & (S(bt, 3) < S(bb, 4)) & bull & (c > S(bt, 3)) & (c < S(bb, 4)),
                        long4 & bull4 & (S(bb, 3) > S(bt, 4)) & bear & (c < S(bb, 3)) & (c > S(bt, 4)))
    inside_bar = (h1 < h2) & (l1 > l2)
    p["hikkake"] = pm(S(inside_bar) & (l1 < S(l, 2)) & (c > S(h, 2)),
                      S(inside_bar) & (h1 > S(h, 2)) & (c < S(l, 2)))

    out = pd.DataFrame(p, index=df.index).fillna(0).astype(np.int8)
    directional = [k for k in out.columns if k not in FLAGS]
    out["bull_count"] = (out[directional] > 0).sum(axis=1).astype(np.int8)
    out["bear_count"] = (out[directional] < 0).sum(axis=1).astype(np.int8)
    out["pattern_score"] = (out["bull_count"] - out["bear_count"]).astype(np.int8)
    return out


def names(pats: pd.DataFrame) -> list:
    return [k for k in pats.columns if k not in ("bull_count", "bear_count", "pattern_score")]


def describe_row(row: pd.Series) -> list:
    out = []
    for k, v in row.items():
        if k in ("bull_count", "bear_count", "pattern_score") or not v:
            continue
        nice = k.replace("_", " ")
        out.append(nice if k in FLAGS else f"{nice} ({'bull' if v > 0 else 'bear'})")
    return out
