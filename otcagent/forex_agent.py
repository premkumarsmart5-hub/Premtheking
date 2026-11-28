"""The Forex AI agent: one model trained on many real forex pairs at once.

Pipeline
  1. features for every pair (forex_features.build_forex_features)
  2. pooled dataset sorted by time (12 pairs = 12x more examples than one pair)
  3. walk-forward test by TIME on the older 80% -> picks a confidence threshold
  4. final exam on the newest 20% (never used for any choice)
  5. breakdown of the exam by pair, session and regime
  6. final model on the most recent data, saved for live use
Live: analyse() returns CALL / PUT / NO TRADE plus a full analysis report.
"""
import numpy as np
import pandas as pd

from .backtest import breakeven, choose_threshold, evaluate_thresholds, trade_stats, wilson_lower
from .candlesticks import describe_row
from .features import make_labels
from .forex_features import (REVERSAL_VOTES, SETUPS, TREND_VOTES, build_forex_features, htf_for,
                             session_name)
from .model import EnsembleModel

WARMUP = 250


def pair_dataset(df, tf, horizons, pair_code, n_pairs):
    f = build_forex_features(df, tf)
    for i in range(n_pairs):
        f[f"pair_{i}"] = np.float32(i == pair_code)
    labels = {h: make_labels(df, h, tf).astype(np.float32) for h in horizons}
    f = f.iloc[WARMUP:]
    labels = {h: v.iloc[WARMUP:] for h, v in labels.items()}
    return f, labels


def pooled(frames: dict, tf: int, horizons, log=print):
    names = sorted(frames)
    X, Y, P = [], {h: [] for h in horizons}, []
    for i, name in enumerate(names):
        f, lab = pair_dataset(frames[name], tf, horizons, i, len(names))
        X.append(f)
        P.append(pd.Series(name, index=f.index))
        for h in horizons:
            Y[h].append(lab[h])
        log(f"  {name}: {len(f)} candles, {f.shape[1]} features")
    X = pd.concat(X)
    order = np.argsort(X.index.values, kind="stable")
    X = X.iloc[order]
    P = pd.concat(P).iloc[order]
    Y = {h: pd.concat(v).iloc[order] for h, v in Y.items()}
    return X, Y, P, names


def _fit(X, y, max_rows):
    if len(X) > max_rows:  # train on the most recent rows: markets change
        X, y = X.iloc[-max_rows:], y.iloc[-max_rows:]
    return EnsembleModel(fast=True).fit(X, y)


def walk_forward_time(X, y, horizon_min, n_folds, max_rows, start_frac=0.5):
    t = X.index.values.astype("datetime64[ns]")
    ok = y.notna().to_numpy()
    edges = np.quantile(t.astype("int64"), np.linspace(start_frac, 1, n_folds + 1)).astype("datetime64[ns]")
    probs = pd.Series(np.nan, index=np.arange(len(X)))
    gap = np.timedelta64(int(horizon_min), "m")
    for i in range(n_folds):
        a, b = edges[i], edges[i + 1]
        tr = (t < a - gap) & ok
        te = (t >= a) & ((t < b) if i < n_folds - 1 else (t <= b)) & ok
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        m = _fit(X[tr], y[tr], max_rows)
        probs[np.where(te)[0]] = m.predict_proba(X[te])
    return probs


def breakdown(probs, y, keys, thr):
    rows = []
    for k in pd.unique(keys):
        m = keys == k
        n, w = trade_stats(probs[m], y[m], thr)
        rows.append((k, n, w / n if n else 0.0))
    return rows


def train_forex(frames: dict, tf: int = 1, horizons=(1, 2, 3, 5), payout: float = 0.80,
                n_folds: int = 3, max_rows: int = 60000, holdout_frac: float = 0.2,
                min_trades: int = 200, log=print) -> dict:
    log(f"Building features for {len(frames)} pairs ({tf}-min candles)…")
    X, Y, P, names = pooled(frames, tf, horizons, log)
    be = breakeven(payout)
    t = X.index.values.astype("datetime64[ns]")
    cut = np.quantile(t.astype("int64"), 1 - holdout_frac).astype("datetime64[ns]")
    log(f"Pooled: {len(X)} rows. Final exam = everything after {pd.Timestamp(cut)}.")
    bundle = {"tf": tf, "payout": payout, "breakeven": be, "pairs": names,
              "columns": list(X.columns), "horizons": {}}
    session = np.array([session_name(pd.Timestamp(x)) for x in X.index])
    regime = np.where(X["rg_trending"].to_numpy() > 0, "trending (ADX>25)", "ranging (ADX<=25)")
    for h in horizons:
        hm = h * tf
        y = Y[h]
        log(f"\n=== Expiry {hm} min ===")
        gap = np.timedelta64(int(hm), "m")
        dev = t < cut - gap
        hold = (t >= cut) & y.notna().to_numpy()
        Xd, yd = X[dev], y[dev]
        probs = walk_forward_time(Xd, yd, hm, n_folds, max_rows)
        sel = probs.notna().to_numpy()
        pv, yv = probs.to_numpy()[sel], yd.to_numpy()[sel]
        acc = float(np.mean((pv > 0.5) == (yv == 1)))
        log(f"Walk-forward accuracy (every candle, all pairs): {acc:.2%} on {len(pv)} candles")
        tab = evaluate_thresholds(pd.Series(pv), pd.Series(yv), payout, min_trades)
        best = choose_threshold(tab)
        show = tab[tab["threshold"].isin([0.5, 0.55, 0.6, 0.65])]
        log(show[["threshold", "trades", "win_rate", "wilson_low"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        thr, exam = None, None
        # per-pair thresholds: one shared AI, but each pair must prove itself separately
        pdev = P[dev].to_numpy()[sel]
        pair_best = {}
        for name in names:
            mk = pdev == name
            if mk.sum() < 500:
                continue
            tb = evaluate_thresholds(pd.Series(pv[mk]), pd.Series(yv[mk]), payout, min_trades=100)
            bp = choose_threshold(tb)
            if bp:
                pair_best[name] = bp
        pair_thr = {}
        if not best and not pair_best:
            log(f"NOT TRADABLE: nothing beat break-even ({be:.2%}) with 95% confidence, "
                "for all pairs together or for any single pair.")
        else:
            m = _fit(Xd[yd.notna().to_numpy()], yd[yd.notna()], max_rows)
            ph = m.predict_proba(X[hold])
            yh = y[hold].to_numpy()
            ph_pair = P[hold].to_numpy()
            if best:
                n, w = trade_stats(ph, yh, best["threshold"])
                wr = w / n if n else 0.0
                exam = {"trades": n, "win_rate": wr, "wilson_low": wilson_lower(w, n)}
                log(f"All pairs: walk-forward pick confidence >= {best['threshold']:.2f} "
                    f"({best['win_rate']:.2%} over {int(best['trades'])} trades)")
                log(f"FINAL EXAM (newest 20%): {wr:.2%} over {n} trades")
                log("  by pair:    " + ", ".join(f"{k} {v:.0%}/{n_}" for k, n_, v in
                                                breakdown(ph, yh, ph_pair, best["threshold"]) if n_))
                log("  by session: " + ", ".join(f"{k} {v:.0%}/{n_}" for k, n_, v in
                                                breakdown(ph, yh, session[hold], best["threshold"]) if n_))
                log("  by regime:  " + ", ".join(f"{k} {v:.0%}/{n_}" for k, n_, v in
                                                breakdown(ph, yh, regime[hold], best["threshold"]) if n_))
                if n >= 50 and wr > be:
                    thr = best["threshold"]
                    log(f"TRADABLE (all pairs) at confidence >= {thr:.2f}")
                else:
                    log(f"All pairs: failed the final exam (break-even {be:.2%}).")
            else:
                log(f"All pairs together: no edge (break-even {be:.2%}).")
            for name, bp in pair_best.items():
                mk = ph_pair == name
                n, w = trade_stats(ph[mk], yh[mk], bp["threshold"])
                wr = w / n if n else 0.0
                ok_ = n >= 30 and wr > be
                log(f"  {name}: walk-forward {bp['win_rate']:.1%}/{int(bp['trades'])} at >= {bp['threshold']:.2f}"
                    f" -> final exam {wr:.1%}/{n}  {'PASS' if ok_ else 'fail'}")
                if ok_:
                    pair_thr[name] = {"threshold": bp["threshold"], "win_rate": wr, "trades": n}
            if pair_thr:
                log(f"TRADABLE pairs: {', '.join(pair_thr)}")
        ok = y.notna().to_numpy()
        bundle["horizons"][h] = {"model": _fit(X[ok], y[ok], max_rows), "threshold": thr,
                                 "pair_thresholds": pair_thr,
                                 "oos_accuracy": acc, "exam": exam,
                                 "oos_best": ({"win_rate": exam["win_rate"], "trades": exam["trades"]}
                                              if thr else None)}
    return bundle


# ------------------------------------------------------------------ live analysis

def analyse(df: pd.DataFrame, bundle: dict, pair: str) -> dict:
    """Full analysis of the last closed candle + CALL/PUT/NO TRADE per expiry."""
    tf = bundle["tf"]
    names = bundle["pairs"]
    f = build_forex_features(df, tf)
    for i, _ in enumerate(names):
        f[f"pair_{i}"] = np.float32(names[i] == pair)
    row = f.iloc[[-1]][bundle["columns"]]
    r = row.iloc[0]
    votes = {k: int(r[f"vote_{k}"]) for k in SETUPS}
    bull = [k for k, v in votes.items() if v > 0]
    bear = [k for k, v in votes.items() if v < 0]
    cdl = r[[c for c in row.columns if c.startswith("cdl_")]]
    cdl.index = [c[4:] for c in cdl.index]
    htfs = {f"{m}m": ("up" if r[f"htf{m}_trend"] > 0 else "down" if r[f"htf{m}_trend"] < 0 else "flat")
            for m in htf_for(tf)}
    ts = df.index[-1]
    out = {
        "asset": pair, "candle_time": str(ts), "price": float(df["close"].iloc[-1]),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "breakeven": bundle["breakeven"],
        "patterns": describe_row(cdl),
        "indicators": {"bullish": ", ".join(bull) or "-", "bearish": ", ".join(bear) or "-",
                       "trend votes": int(sum(votes[k] for k in TREND_VOTES)),
                       "reversal votes": int(sum(votes[k] for k in REVERSAL_VOTES)),
                       "higher TF": ", ".join(f"{k} {v}" for k, v in htfs.items()),
                       "regime": "trending" if r["rg_trending"] > 0 else "ranging",
                       "session": session_name(ts),
                       "RSI14": round(float(r["rsi14"]), 1), "ADX": round(float(r["adx"]), 1)},
        "signals": {},
    }
    for h, e in bundle["horizons"].items():
        parts = {k: float(v[0]) for k, v in e["model"].predict_proba_parts(row).items()}
        p = float(np.mean(list(parts.values())))
        lean = "CALL" if p >= 0.5 else "PUT"
        conf = max(p, 1 - p)
        agree = all((v >= 0.5) == (p >= 0.5) for v in parts.values())
        pt = (e.get("pair_thresholds") or {}).get(pair)
        thr = pt["threshold"] if pt else e["threshold"]
        if thr is None:
            action, reason = "NO TRADE", f"{pair} {h * tf}m failed the backtest"
        elif conf < thr:
            action, reason = "NO TRADE", f"confidence {conf:.1%} < {thr:.0%}"
        elif not agree:
            action, reason = "NO TRADE", "AI models disagree"
        else:
            action, reason = lean, "all checks passed"
        ex = pt or e.get("exam") or {}
        out["signals"][f"{h * tf}m"] = {
            "action": action, "lean": lean, "confidence": round(conf, 4), "required": thr,
            "reason": reason, "models": {k: round(v, 3) for k, v in parts.items()},
            "backtest_win_rate": round(ex["win_rate"], 4) if thr and ex else None,
            "backtest_trades": int(ex.get("trades", 0)) if thr else 0,
        }
    return out


def format_analysis(a: dict) -> str:
    i = a["indicators"]
    lines = [f"[{a['candle_time']}] {a['asset']} @ {a['price']:.5f}   {i['session']}, {i['regime']}",
             f"  higher TF: {i['higher TF']}   RSI {i['RSI14']}  ADX {i['ADX']}",
             f"  patterns : {', '.join(a['patterns']) or 'none'}",
             f"  bullish  : {i['bullish']}",
             f"  bearish  : {i['bearish']}"]
    for k, s in a["signals"].items():
        tag = {"CALL": "▲ CALL", "PUT": "▼ PUT"}.get(s["action"], "· NO TRADE")
        lines.append(f"  {k:>4}: {tag:<11} {s['lean']} {s['confidence']:.0%}  ({s['reason']})")
    return "\n".join(lines)
