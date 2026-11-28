"""Live signal engine: CALL / PUT / NO TRADE for 1, 2, 3 and 5 minute expiries."""
import time
from datetime import datetime, timezone

import pickle
import numpy as np

from .features import WARMUP, build_features
from .patterns import describe_latest, detect_patterns
from . import indicators as ind


def load_bundle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_bundle(bundle, path):
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)


def indicator_confluence(df) -> dict:
    """Plain-language view of the classic indicators on the last candle."""
    c = df["close"]
    r = float(ind.rsi(c, 14).iloc[-1])
    k, _ = ind.stochastic(df)
    pb, _ = ind.bollinger(c)
    _, _, hist = ind.macd(c)
    trend = "up" if ind.ema(c, 9).iloc[-1] > ind.ema(c, 21).iloc[-1] else "down"
    return {
        "rsi14": round(r, 1),
        "stoch_k": round(float(k.iloc[-1]), 1),
        "bb_pctb": round(float(pb.iloc[-1]), 2),
        "macd_hist": "rising" if hist.iloc[-1] > hist.iloc[-2] else "falling",
        "ema_trend": trend,
        "supertrend": "up" if ind.supertrend_dir(df).iloc[-1] > 0 else "down",
    }


def generate(df, bundle, asset="") -> dict:
    """Signals for the candle that just closed (last row of df)."""
    if len(df) < WARMUP + 5:
        raise ValueError(f"need at least {WARMUP + 5} candles")
    feats = build_features(df).iloc[[-1]]
    out = {
        "asset": asset,
        "candle_time": str(df.index[-1]),
        "price": float(df["close"].iloc[-1]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patterns": describe_latest(detect_patterns(df)),
        "indicators": indicator_confluence(df),
        "breakeven": bundle["breakeven"],
        "signals": {},
    }
    for h, entry in bundle["horizons"].items():
        model = entry["model"]
        parts = {k: float(v[0]) for k, v in model.predict_proba_parts(feats).items()}
        p = float(np.mean(list(parts.values())))
        direction = "CALL" if p >= 0.5 else "PUT"
        conf = p if p >= 0.5 else 1 - p
        agree = all((v >= 0.5) == (p >= 0.5) for v in parts.values())
        thr = entry["threshold"]
        if thr is None:
            action, reason = "NO TRADE", "expiry not validated (no edge in backtest)"
        elif conf < thr:
            action, reason = "NO TRADE", f"confidence {conf:.0%} < required {thr:.0%}"
        elif not agree:
            action, reason = "NO TRADE", "models disagree"
        else:
            action, reason = direction, "all checks passed"
        best = entry.get("oos_best") or {}
        out["signals"][f"{h}m"] = {
            "action": action,
            "lean": direction,
            "confidence": round(conf, 4),
            "required": thr,
            "reason": reason,
            "models": {k: round(v, 3) for k, v in parts.items()},
            "backtest_win_rate": round(best["win_rate"], 4) if best else None,
            "backtest_trades": int(best["trades"]) if best else 0,
        }
    return out


def format_signal(s: dict) -> str:
    lines = [f"[{s['candle_time']}] {s['asset']} @ {s['price']:.5f}",
             f"  patterns: {', '.join(s['patterns']) or 'none'}",
             "  " + " | ".join(f"{k}={v}" for k, v in s["indicators"].items())]
    for h, sig in s["signals"].items():
        tag = {"CALL": "▲ CALL", "PUT": "▼ PUT"}.get(sig["action"], "· NO TRADE")
        lines.append(f"  {h:>3}: {tag:<11} conf {sig['confidence']:.0%}  ({sig['reason']})")
    return "\n".join(lines)


def run_live(source, asset, bundle, on_signal=print, lookback=400):
    """Wait for each 1-minute candle to close, then emit signals."""
    while True:
        now = time.time()
        wait = 60 - (now % 60) + 1.0  # 1 s after the minute boundary
        time.sleep(wait)
        try:
            df = source.candles(asset, lookback)
            # drop the candle that is still forming (started this minute)
            if len(df) and df.index[-1].timestamp() > time.time() - 60:
                df = df.iloc[:-1]
            if not len(df) or df.index[-1].timestamp() < time.time() - 180:
                last = df.index[-1] if len(df) else "never"
                print(f"{asset}: market closed / no fresh candles (last {last}) - no signals",
                      flush=True)
                continue
            on_signal(generate(df, bundle, asset))
        except Exception as e:  # keep running through network hiccups
            print(f"live error: {e}")
