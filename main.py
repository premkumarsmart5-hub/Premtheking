"""OTC Signal Agent - command line.

  python main.py collect  --asset EURUSD-OTC --days 7     # download candle history
  python main.py train    --asset EURUSD-OTC --payout 0.85 # backtest + train AI models
  python main.py live     --asset EURUSD-OTC --dashboard   # live CALL/PUT signals
  python main.py demo                                      # offline demo, no login needed
"""
import argparse
import json
from pathlib import Path

from otcagent import data
from otcagent.backtest import train_and_validate
from otcagent.signals import format_signal, generate, load_bundle, run_live, save_bundle


def paths(asset):
    safe = asset.replace("/", "_")
    return Path("data") / f"{safe}_1m.csv", Path("models") / f"{safe}.pkl"


def cmd_collect(a):
    csv, _ = paths(a.asset)
    src = data.IQOptionSource(account=a.account).connect()
    print(f"Downloading {a.days} days of 1-minute {a.asset} candles…")
    new = src.history(a.asset, a.days)
    if csv.exists():  # merge with what we already have, history keeps growing
        old = data.load_csv(csv)
        new = old.combine_first(new) if len(old) else new
    data.save_csv(new, csv)
    print(f"Saved {len(new)} candles to {csv}")


def cmd_train(a):
    csv, model_path = paths(a.asset)
    src_csv = Path(a.csv or csv)
    if not src_csv.exists():
        raise SystemExit(f"No data file {src_csv}. Run first:  python main.py collect --asset {a.asset}")
    df = data.load_csv(src_csv)
    print(f"Training on {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    bundle = train_and_validate(df, payout=a.payout, n_folds=a.folds, min_trades=a.min_trades)
    bundle["asset"] = a.asset
    model_path.parent.mkdir(exist_ok=True)
    save_bundle(bundle, model_path)
    summary = {f"{h}m": (f"TRADE at conf >= {e['threshold']:.2f}" if e["threshold"] else "NO TRADE")
               for h, e in bundle["horizons"].items()}
    print(f"\nSaved model to {model_path}\nSummary: {json.dumps(summary)}")


def cmd_live(a):
    _, model_path = paths(a.asset)
    bundle = load_bundle(model_path)
    src = data.IQOptionSource(account=a.account).connect()
    publish = None
    if a.dashboard:
        from otcagent import dashboard
        dashboard.serve(a.host, a.port)
        publish = dashboard.publish
    log = Path("signals_log.jsonl")

    def on_signal(sig):
        print(format_signal(sig), flush=True)
        with log.open("a") as fh:
            fh.write(json.dumps(sig) + "\n")
        if publish:
            publish(sig)

    print(f"Live on {a.asset}. Signals appear 1 s after each minute closes. Ctrl+C to stop.")
    run_live(src, a.asset, bundle, on_signal)


def cmd_demo(a):
    print("Demo on synthetic candles with a hidden pattern (no login needed).")
    df = data.synthetic(n=a.rows, edge=0.65, seed=3)
    bundle = train_and_validate(df, payout=a.payout, n_folds=4)
    print("\nSignal for the last candle:\n" + format_signal(generate(df, bundle, "SYNTH-OTC")))


def main():
    p = argparse.ArgumentParser(description="OTC CALL/PUT signal agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "train", "live"):
        s = sub.add_parser(name)
        s.add_argument("--asset", default="EURUSD-OTC")
        s.add_argument("--account", default="PRACTICE", choices=["PRACTICE", "REAL"])
    sub.choices["collect"].add_argument("--days", type=float, default=7)
    t = sub.choices["train"]
    t.add_argument("--csv", help="train from a CSV instead of data/<asset>_1m.csv")
    t.add_argument("--payout", type=float, default=0.85, help="0.85 = 85%% payout")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--min-trades", type=int, default=100)
    lv = sub.choices["live"]
    lv.add_argument("--dashboard", action="store_true")
    lv.add_argument("--host", default="127.0.0.1")
    lv.add_argument("--port", type=int, default=8080)
    d = sub.add_parser("demo")
    d.add_argument("--rows", type=int, default=15000)
    d.add_argument("--payout", type=float, default=0.85)
    a = p.parse_args()
    {"collect": cmd_collect, "train": cmd_train, "live": cmd_live, "demo": cmd_demo}[a.cmd](a)


if __name__ == "__main__":
    main()
