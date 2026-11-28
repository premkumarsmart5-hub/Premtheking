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
