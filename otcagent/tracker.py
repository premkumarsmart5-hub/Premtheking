"""Scores live signals automatically, so you can compare with your demo account.

Entry = close of the signal candle. Expiry = close of the candle that ends
`expiry` minutes later. Higher close -> CALL wins, lower -> PUT wins, equal -> tie
(IQ Option refunds a tie). Your real demo entry is a few seconds later, so your
results can differ slightly - that difference is worth watching too.
"""
import json
from pathlib import Path

import pandas as pd


class Tracker:
    def __init__(self, tf: int, log_path="results_log.jsonl", payout: float = 0.80):
        self.tf = tf
        self.payout = payout
        self.log = Path(log_path)
        self.pending = []
        self.tally = {}
        self.recent = []  # last finished trades, newest first

    def add(self, tag, asset, candle_open, price, action, expiry_min):
        self.pending.append({"tag": tag, "asset": asset, "entry_candle": pd.Timestamp(candle_open),
                             "price": float(price), "action": action, "expiry": int(expiry_min)})

    def update(self, df: pd.DataFrame, asset: str) -> list:
        lines, keep = [], []
        for p in self.pending:
            if p["asset"] != asset:
                keep.append(p)
                continue
            target = p["entry_candle"] + pd.Timedelta(minutes=p["expiry"])
            if target not in df.index:
                if df.index[-1] > target:  # candle missing (market gap): drop as void
                    lines.append(f"  void: {p['tag']} {p['action']} from {p['entry_candle']} (no expiry candle)")
                else:
                    keep.append(p)
                continue
            close = float(df.loc[target, "close"])
            diff = close - p["price"]
            if diff == 0:
                res = "TIE"
            elif (diff > 0) == (p["action"] == "CALL"):
                res = "WIN"
            else:
                res = "LOSS"
            t = self.tally.setdefault(p["tag"], {"WIN": 0, "LOSS": 0, "TIE": 0})
            t[res] += 1
            mark = {"WIN": "✔ WIN", "LOSS": "✘ LOSS", "TIE": "= TIE"}[res]
            lines.append(f"  {mark}: {p['tag']} {p['action']} @ {p['price']:.5f} -> {close:.5f}"
                         f"  ({p['expiry']} min)   {self.score(p['tag'])}")
            self.recent.insert(0, {"tag": p["tag"], "action": p["action"], "entry": p["price"],
                                   "close": close, "result": res, "expiry": p["expiry"],
                                   "time": str(target + pd.Timedelta(minutes=self.tf))})
            del self.recent[20:]
            with self.log.open("a") as fh:
                fh.write(json.dumps({**p, "entry_candle": str(p["entry_candle"]), "expiry_close": close,
                                     "result": res}) + "\n")
        self.pending = keep
        return lines

    def score(self, tag) -> str:
        t = self.tally.get(tag, {"WIN": 0, "LOSS": 0, "TIE": 0})
        n = t["WIN"] + t["LOSS"]
        wr = t["WIN"] / n if n else 0.0
        profit = t["WIN"] * self.payout - t["LOSS"]
        return f"score {t['WIN']}W-{t['LOSS']}L-{t['TIE']}T = {wr:.0%}, {profit:+.2f} stakes"

    def board(self) -> dict:
        return {tag: self.score(tag) for tag in self.tally}

    def detail(self) -> dict:
        out = {}
        for tag, t in self.tally.items():
            n = t["WIN"] + t["LOSS"]
            out[tag] = {"wins": t["WIN"], "losses": t["LOSS"], "ties": t["TIE"],
                        "win_rate": t["WIN"] / n if n else 0.0,
                        "profit": round(t["WIN"] * self.payout - t["LOSS"], 2)}
        return out

    def snapshot(self) -> dict:
        return {"scoreboard": self.board(), "score": self.detail(), "results": self.recent[:10],
                "pending": len(self.pending)}
