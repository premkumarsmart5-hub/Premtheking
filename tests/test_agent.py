import numpy as np
import pytest

from otcagent import data
from otcagent.backtest import breakeven, train_and_validate, wilson_lower
from otcagent.features import WARMUP, build_features, make_labels
from otcagent.patterns import detect_patterns
from otcagent.signals import generate


def test_features_do_not_look_ahead():
    df = data.synthetic(800, seed=5)
    full = build_features(df)
    cut = build_features(df.iloc[:500])
    a = full.iloc[WARMUP:500].fillna(-999).values
    b = cut.iloc[WARMUP:500].fillna(-999).values
    assert np.allclose(a, b, atol=1e-9), "a feature uses future candles"


def test_labels():
    df = data.synthetic(50, seed=2)
    lab = make_labels(df, 3)
    assert lab.iloc[-3:].isna().all()
    i = 10
    exp = 1.0 if df.close.iloc[i + 3] > df.close.iloc[i] else 0.0
    assert lab.iloc[i] == exp


def test_patterns_values():
    pats = detect_patterns(data.synthetic(2000, seed=4))
    assert set(np.unique(pats.drop(columns="pattern_score").values)) <= {-1, 0, 1}
    assert (pats["engulfing"] != 0).any()


def test_math():
    assert breakeven(0.85) == pytest.approx(0.5405, abs=1e-4)
    assert wilson_lower(60, 100) < 0.6


def test_finds_planted_edge_and_rejects_noise():
    logs = []
    edge = train_and_validate(data.synthetic(9000, edge=0.7, seed=3), horizons=(1,),
                              n_folds=3, log=logs.append)
    assert edge["horizons"][1]["threshold"] is not None
    noise = train_and_validate(data.synthetic(9000, edge=0.0, seed=9), horizons=(1,),
                               n_folds=3, log=logs.append)
    assert noise["horizons"][1]["threshold"] is None
    sig = generate(data.synthetic(9000, edge=0.0, seed=9), noise)
    assert sig["signals"]["1m"]["action"] == "NO TRADE"


def test_csv_roundtrip_keeps_real_times(tmp_path):
    df = data.synthetic(300, seed=1)
    path = tmp_path / "c.csv"
    data.save_csv(df, path)
    back = data.load_csv(path)
    assert back.index.equals(df.index.as_unit(back.index.unit))
    assert back.index[0].year == 2026


def test_labels_skip_gaps():
    df = data.synthetic(40, seed=2)
    df = df.drop(df.index[20:25])  # a 5-minute hole, like a market close
    lab = make_labels(df, 2)
    i = df.index.get_loc(df.index[19])
    assert lab.iloc[i] != lab.iloc[i] and lab.iloc[i - 1] != lab.iloc[i - 1]  # NaN across gap
    assert lab.iloc[5] == lab.iloc[5]  # normal rows still labelled


def test_strategies_find_planted_rule_only():
    from otcagent.strategies import STRATEGIES, test_strategy
    for edge, expect in ((0.7, True), (0.0, False)):
        df = data.synthetic(8000, edge=edge, seed=3)
        fn = STRATEGIES["three_candle_reversal"][0]
        res = test_strategy(df, fn(df), (1,), 1, 0.85)
        assert res[0]["pass"] is expect
    df = data.synthetic(3000, seed=5)
    for name, (fn, _) in STRATEGIES.items():
        s = fn(df)
        assert len(s) == len(df) and set(s.unique()) <= {-1, 0, 1}, name


def test_channel_bounce_no_lookahead_and_finds_channel():
    import pandas as pd
    from otcagent.strategies import channel_bounce, test_strategy
    df = data.synthetic(5000, seed=5)
    assert (channel_bounce(df, 30, "any").iloc[:3500].values
            == channel_bounce(df.iloc[:3500], 30, "any").values).all()
    n = 6000
    t = np.arange(n)
    c = 1.1 + t * 3e-6 + np.sin(t / 2.2) * 3e-4 + np.random.default_rng(1).normal(0, 4e-5, n)
    o = np.r_[c[0], c[:-1]]
    ch = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 1e-5, "low": np.minimum(o, c) - 1e-5,
                       "close": c}, index=pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"))
    r = test_strategy(ch, channel_bounce(ch, 30, "up"), (2,), 1, 0.85)[0]
    assert r["pass"] and r["win_rate"] > 0.7
