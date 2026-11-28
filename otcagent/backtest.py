"""Honest walk-forward validation and confidence-threshold selection.

The model is always tested on candles it has never seen (expanding window,
with a purge gap so future labels never leak into training). A horizon is
only marked TRADABLE when its out-of-sample win rate is statistically above
the break-even win rate for your payout.
"""
import math

import numpy as np
import pandas as pd

from .features import HORIZONS, WARMUP, build_features, make_labels
from .model import EnsembleModel


def breakeven(payout: float) -> float:
    """Win rate needed to break even. payout=0.85 means +85% on a win, -100% on a loss."""
    return 1.0 / (1.0 + payout)


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def dataset(df: pd.DataFrame, horizon: int, feats: pd.DataFrame = None):
    feats = build_features(df) if feats is None else feats
    y = make_labels(df, horizon)
    mask = y.notna()
    mask.iloc[:WARMUP] = False
    return feats[mask], y[mask]


def walk_forward(X: pd.DataFrame, y: pd.Series, horizon: int, n_folds: int = 5,
                 initial_frac: float = 0.5):
    n = len(X)
    start = int(n * initial_frac)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    probs = pd.Series(np.nan, index=X.index)
    parts = {}
    for i in range(n_folds):
        a, b = edges[i], edges[i + 1]
        train_end = max(0, a - horizon)  # purge: drop rows whose label overlaps the test window
        if train_end < 500 or b <= a:
            continue
        m = EnsembleModel().fit(X.iloc[:train_end], y.iloc[:train_end])
        pp = m.predict_proba_parts(X.iloc[a:b])
        probs.iloc[a:b] = np.mean(np.vstack(list(pp.values())), axis=0)
        for k, v in pp.items():
            parts.setdefault(k, []).append(pd.Series(v, index=X.index[a:b]))
    parts = {k: pd.concat(v) for k, v in parts.items()}
    return probs.dropna(), parts


def evaluate_thresholds(probs: pd.Series, y: pd.Series, payout: float,
                        min_trades: int = 100) -> pd.DataFrame:
    y = y.loc[probs.index]
    rows = []
    for t in np.round(np.arange(0.50, 0.801, 0.01), 2):
        call = probs >= t if t > 0.5 else probs > 0.5
        put = probs <= 1 - t if t > 0.5 else probs < 0.5
        n = int(call.sum() + put.sum())
        wins = int(((call) & (y == 1)).sum() + ((put) & (y == 0)).sum())
        wr = wins / n if n else 0.0
        rows.append({
            "threshold": t, "trades": n, "wins": wins, "win_rate": wr,
            "wilson_low": wilson_lower(wins, n),
            "profit_per_trade": wr * payout - (1 - wr) if n else 0.0,
            "coverage": n / len(probs),
        })
    tab = pd.DataFrame(rows)
    be = breakeven(payout)
    tab["passes"] = (tab["trades"] >= min_trades) & (tab["wilson_low"] > be)
    return tab


def choose_threshold(tab: pd.DataFrame):
    ok = tab[tab["passes"]]
    if ok.empty:
        return None
    # best total expected profit among statistically valid thresholds
    best = ok.loc[(ok["profit_per_trade"] * ok["trades"]).idxmax()]
    return best.to_dict()


def trade_stats(probs: np.ndarray, y: np.ndarray, thr: float) -> tuple:
    call, put = probs >= thr, probs <= 1 - thr
    n = int(call.sum() + put.sum())
    wins = int((call & (y == 1)).sum() + (put & (y == 0)).sum())
    return n, wins


def train_and_validate(df: pd.DataFrame, payout: float = 0.85, horizons=HORIZONS,
                       n_folds: int = 5, min_trades: int = 100, holdout_frac: float = 0.2,
                       min_holdout_trades: int = 30, log=print) -> dict:
    """Walk-forward on the older 80% picks a confidence threshold; the newest 20%
    (never used for any choice) must then confirm it. Only confirmed expiries trade."""
    feats = build_features(df)
    be = breakeven(payout)
    bundle = {"payout": payout, "breakeven": be, "horizons": {},
              "trained_on_rows": len(df),
              "trained_until": str(df.index[-1]) if len(df) else None}
    for h in horizons:
        X, y = dataset(df, h, feats)
        n_hold = int(len(X) * holdout_frac)
        dev_end = len(X) - n_hold - h  # purge gap so dev labels never peek into holdout
        Xd, yd = X.iloc[:dev_end], y.iloc[:dev_end]
        Xh, yh = X.iloc[len(X) - n_hold:], y.iloc[len(X) - n_hold:]
        log(f"\n=== Expiry {h} min | {len(Xd)} dev + {len(Xh)} final-exam samples ===")
        probs, parts = walk_forward(Xd, yd, h, n_folds=n_folds)
        yy = yd.loc[probs.index]
        acc = float(np.mean((probs.to_numpy() > 0.5) == (yy.to_numpy() == 1)))
        log(f"Out-of-sample direction accuracy (every candle): {acc:.2%}")
        for k, v in parts.items():
            a_k = float(np.mean((v.to_numpy() > 0.5) == (yd.loc[v.index].to_numpy() == 1)))
            log(f"   {k:<12} alone: {a_k:.2%}")
        tab = evaluate_thresholds(probs, yd, payout, min_trades)
        best = choose_threshold(tab)
        log(f"Break-even win rate at {payout:.0%} payout: {be:.2%}")
        show = tab[tab["threshold"].isin([0.5, 0.55, 0.6, 0.65, 0.7])]
        log(show[["threshold", "trades", "win_rate", "wilson_low", "profit_per_trade"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        holdout = None
        threshold = None
        if not best:
            log("NOT TRADABLE: no confidence level beat break-even with statistical "
                "confidence. The agent will show NO TRADE for this expiry.")
        elif n_hold == 0:
            threshold = best["threshold"]
        else:
            m = EnsembleModel().fit(Xd, yd)
            n, w = trade_stats(m.predict_proba(Xh), yh.to_numpy(), best["threshold"])
            wr = w / n if n else 0.0
            holdout = {"trades": n, "wins": w, "win_rate": wr}
            log(f"Walk-forward pick: confidence >= {best['threshold']:.2f} "
                f"({best['win_rate']:.2%} over {int(best['trades'])} trades)")
            log(f"Final exam on newest {n_hold} candles: {wr:.2%} over {n} trades")
            if n >= min_holdout_trades and wr > be:
                threshold = best["threshold"]
                log(f"TRADABLE: confirmed on unseen recent data.")
            else:
                why = "too few trades" if n < min_holdout_trades else "below break-even"
                log(f"NOT TRADABLE: failed the final exam ({why}). Likely luck or the "
                    "pattern already stopped working.")
        final = EnsembleModel().fit(X, y)
        bundle["horizons"][h] = {
            "model": final,
            "threshold": threshold,
            "oos_accuracy": acc,
            "oos_best": best if threshold else None,
            "holdout": holdout,
            "table": tab.to_dict("records"),
        }
    return bundle
