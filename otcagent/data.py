"""Candle data sources: IQ Option (OTC included), CSV files, and synthetic data.

IQ Option has no official public API. The connector uses the community
`iqoptionapi` package. Your email/password are read from the .env file on
YOUR device only and are never written anywhere else.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["open", "high", "low", "close"]


def load_env(path: str = ".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class IQOptionSource:
    def __init__(self, email: str = None, password: str = None, account: str = "PRACTICE"):
        load_env()
        self.email = email or os.environ.get("IQ_EMAIL")
        self.password = password or os.environ.get("IQ_PASSWORD")
        self.account = account or os.environ.get("IQ_ACCOUNT", "PRACTICE")
        if not self.email or not self.password:
            raise SystemExit("Set IQ_EMAIL and IQ_PASSWORD in your .env file (see .env.example).")
        self.api = None

    def connect(self):
        try:
            from iqoptionapi.stable_api import IQ_Option
        except ImportError:
            raise SystemExit("Install the connector first: pip install -r requirements-iq.txt")
        self.api = IQ_Option(self.email, self.password)
        ok, reason = self.api.connect()
        if not ok:
            e = self.email
            masked = (e[:2] + "***" + e[e.find("@"):]) if "@" in e else "(not an email)"
            hint = ""
            if e == "you@example.com":
                hint = "\n  .env still has the EXAMPLE values - edit it and save (CTRL+O, Enter, CTRL+X)."
            raise SystemExit(
                f"IQ Option login failed: {reason}\n"
                f"  Email used: {masked}  (password length: {len(self.password)})" + hint +
                "\n  Check: same email/password you type on the IQ Option website; no spaces;"
                "\n  if you sign in with Google/Facebook you must first set a password in IQ Option settings;"
                "\n  turn off 2-step verification for the demo account if it is on.")
        self.api.change_balance(self.account)  # PRACTICE = demo account
        return self

    def _ensure(self):
        if self.api is None or not self.api.check_connect():
            self.connect()

    def _get_with_retry(self, asset, n, end_time, timeout=30, tries=5):
        """get_candles can hang forever when the internet drops: time it out,
        reconnect and try again."""
        import threading
        for attempt in range(1, tries + 1):
            box = {}

            def work():
                try:
                    box["v"] = self.api.get_candles(asset, 60, n, end_time)
                except Exception as e:  # noqa: BLE001
                    box["e"] = e

            t = threading.Thread(target=work, daemon=True)
            t.start()
            t.join(timeout)
            if "v" in box:
                return box["v"]
            why = box.get("e", "no answer in %ds" % timeout)
            print(f"  network problem ({why}); reconnecting, try {attempt}/{tries}…", flush=True)
            time.sleep(min(60, 5 * attempt))
            try:
                self.connect()
            except (Exception, SystemExit) as e:  # still offline - wait and retry
                print(f"  reconnect failed: {e}", flush=True)
        raise ConnectionError(f"gave up on {asset} after {tries} tries - check your internet")

    def candles(self, asset: str, count: int = 300, end_time: float = None) -> pd.DataFrame:
        """Fetch up to `count` 1-minute candles ending at end_time (default: now)."""
        self._ensure()
        end_time = end_time or time.time()
        out = []
        remaining = count
        while remaining > 0:
            n = min(1000, remaining)
            batch = self._get_with_retry(asset, n, end_time)
            if not batch:
                break
            out = batch + out
            remaining -= len(batch)
            end_time = batch[0]["from"] - 1
            if len(batch) < n:
                break
        return to_frame(out)

    def history(self, asset: str, days: float = 7, log=print) -> pd.DataFrame:
        total = int(days * 24 * 60)
        frames, end = [], time.time()
        got = 0
        while got < total:
            df = self.candles(asset, min(1000, total - got), end)
            if df.empty:
                break
            frames.append(df)
            got += len(df)
            end = df.index[0].timestamp() - 1
            log(f"  downloaded {got}/{total} candles (back to {df.index[0]})")
            time.sleep(0.3)
        if not frames:
            return pd.DataFrame(columns=COLS)
        all_ = pd.concat(frames).sort_index()
        return all_[~all_.index.duplicated()]


def to_frame(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(raw)
    df = df.rename(columns={"min": "low", "max": "high"})
    df.index = pd.to_datetime(df["from"], unit="s", utc=True)
    df = df[COLS].astype(float).sort_index()
    return df[~df.index.duplicated()]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if c.lower() in ("time", "timestamp", "date", "datetime", "from")), None)
    if tcol is None:
        raise ValueError("CSV needs a time column (time/timestamp/date/from)")
    t = df[tcol]
    if np.issubdtype(t.dtype, np.number) and len(t) and t.max() < 1_000_000_000:
        raise SystemExit(f"{path} has broken timestamps (saved by an older version). "
                         f"Delete it and run collect again:  rm {path}")
    df.index = pd.to_datetime(t, unit="s", utc=True) if np.issubdtype(t.dtype, np.number) \
        else pd.to_datetime(t, utc=True)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"min": "low", "max": "high"})
    df = df[COLS].astype(float).sort_index()
    return df[~df.index.duplicated(keep="last")]


def save_csv(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    out.insert(0, "time", ((out.index - epoch) // pd.Timedelta(seconds=1)).astype("int64"))
    out.to_csv(path, index=False)


def synthetic(n: int = 20000, edge: float = 0.0, seed: int = 1) -> pd.DataFrame:
    """Random-walk candles. With edge > 0, a hidden mean-reversion rule is planted
    (after 3 same-colour candles the next one tends to reverse) so tests can check
    the pipeline actually finds a real pattern - and finds nothing in pure noise."""
    rng = np.random.default_rng(seed)
    close = np.empty(n)
    close[0] = 1.1
    colors = []
    for i in range(1, n):
        step = rng.normal(0, 1e-4)
        if edge and len(colors) >= 3 and colors[-1] == colors[-2] == colors[-3]:
            if rng.random() < edge:
                step = -colors[-1] * abs(step)
        close[i] = close[i - 1] + step
        colors.append(1 if step > 0 else -1)
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 4e-5, (2, n)))
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
