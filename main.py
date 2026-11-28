"""OTC Signal Agent - command line.

  python main.py collect  --asset EURUSD-OTC --days 7     # download candle history
  python main.py train    --asset EURUSD-OTC --payout 0.85 # backtest + train AI models
  python main.py live     --asset EURUSD-OTC --dashboard   # live CALL/PUT signals
  python main.py scan     --days 10                       # test 12 OTC pairs, list winners
  python main.py scan     --market real --days 14         # same for real forex pairs
  python main.py scan     --tf 5                          # 5-min candles: 5/10/15 min expiries
  python main.py strategy --name all                      # test classic rules on saved data
  python main.py demo                                      # offline demo, no login needed
"""
import argparse
import json
from pathlib import Path

from otcagent import data
from otcagent.backtest import train_and_validate
from otcagent.signals import format_signal, generate, load_bundle, run_live, save_bundle


def paths(asset, tf=1):
    safe = asset.replace("/", "_")
    return Path("data") / f"{safe}_{tf}m.csv", Path("models") / f"{safe}_{tf}m.pkl"


def expiries_to_candles(a):
    """--expiries is in minutes; turn it into a number of candles of size --tf."""
    exp = a.expiries or ("1,2,3,5" if a.tf == 1 else "5,10,15")
    out = []
    for m in (int(x) for x in exp.split(",")):
        if m % a.tf:
            raise SystemExit(f"expiry {m} min is not a multiple of the {a.tf}-min candle")
        out.append(m // a.tf)
    return tuple(out)


def default_days(a):
    return a.days if a.days else (10 if a.tf == 1 else 60)


REAL_SCAN = [  # real interbank forex: open Mon-Fri only
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY",
    "EURGBP", "GBPJPY", "AUDCAD", "NZDUSD", "USDCHF", "EURAUD",
]

DEFAULT_SCAN = [
    "EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "AUDCAD-OTC", "EURGBP-OTC", "EURJPY-OTC",
    "NZDUSD-OTC", "USDCHF-OTC", "GBPJPY-OTC", "AUDUSD-OTC", "USDCAD-OTC", "EURAUD-OTC",
]


def collect_asset(src, asset, days, tf=1):
    csv, _ = paths(asset, tf)
    print(f"Downloading {days} days of {tf}-minute {asset} candles…")
    new = src.history(asset, days, size=60 * tf)
    if csv.exists():  # merge with what we already have, history keeps growing
        old = data.load_csv(csv)
        new = old.combine_first(new) if len(old) else new
    data.save_csv(new, csv)
    print(f"Saved {len(new)} candles to {csv}")
    return new


def cmd_collect(a):
    src = data.IQOptionSource(account=a.account).connect()
    collect_asset(src, a.asset, default_days(a), a.tf)


def cmd_scan(a):
    assets = ([x.strip() for x in a.assets.split(",")] if a.assets
              else REAL_SCAN if a.market == "real" else DEFAULT_SCAN)
    src = None if a.no_download else data.IQOptionSource(account=a.account).connect()
    horizons = expiries_to_candles(a)
    days = default_days(a)
    tf = a.tf
    import pandas as pd
    out = Path(f"scan_results_{tf}m.csv")
    rows = []
    if a.resume and out.exists():
        rows = [r for r in pd.read_csv(out, dtype=str).fillna("-").to_dict("records")
                if not str(r["verdict"]).startswith("ERROR")]
        done = {r["asset"] for r in rows}
        skip = [x for x in assets if x in done]
        assets = [x for x in assets if x not in done]
        if skip:
            print(f"Resuming: already done {', '.join(skip)}")
    elif out.exists():
        out.unlink()
    for i, asset in enumerate(assets, 1):
        print(f"\n##### [{i}/{len(assets)}] {asset} #####")
        try:
            if src:
                df = collect_asset(src, asset, days, tf)
            else:
                df = data.load_csv(paths(asset, tf)[0])
            if len(df) < 2000:
                raise ValueError(f"only {len(df)} candles - asset closed or unknown?")
            bundle = train_and_validate(df, payout=a.payout, horizons=horizons,
                                        n_folds=a.folds, tf=tf, log=print if a.verbose else (lambda *_: None))
            bundle["asset"] = asset
            Path("models").mkdir(exist_ok=True)
            save_bundle(bundle, paths(asset, tf)[1])
            for h, e in bundle["horizons"].items():
                ho = e.get("holdout") or {}
                rows.append({"asset": asset, "expiry": f"{h * tf}m",
                             "accuracy": f"{e['oos_accuracy']:.1%}",
                             "exam_win": f"{ho['win_rate']:.1%}" if ho else "-",
                             "exam_trades": ho.get("trades", 0),
                             "verdict": f"TRADE >= {e['threshold']:.2f}" if e["threshold"] else "NO TRADE"})
                print(f"  {h * tf}m: {rows[-1]['verdict']} (accuracy {rows[-1]['accuracy']})", flush=True)
        except (Exception, SystemExit) as e:
            print(f"  skipped: {e}")
            rows.append({"asset": asset, "expiry": "-", "accuracy": "-", "exam_win": "-",
                         "exam_trades": 0, "verdict": f"ERROR: {str(e)[:60]}"})
        pd.DataFrame(rows).to_csv(out, index=False)  # save progress after every asset
    res = pd.DataFrame(rows)
    print("\n================ SCAN RESULTS ================")
    print(f"{'asset':<12}{'exp':>4}{'acc':>7}{'exam':>7}  verdict")
    for r in rows:
        print(f"{r['asset']:<12}{r['expiry']:>4}{r['accuracy']:>7}{r['exam_win']:>7}  {r['verdict']}")
    good = res[res["verdict"].str.startswith("TRADE")]
    print(f"\n{len(good)} asset/expiry combination(s) passed both the walk-forward test "
          f"and the final exam (break-even {1 / (1 + a.payout):.2%}).")
    if len(good):
        print(f"Next: forward-test one on PRACTICE:  python main.py live --tf {tf} --asset <ASSET> --dashboard")
    print(f"Saved to {out}")


def cmd_train(a):
    csv, model_path = paths(a.asset, a.tf)
    src_csv = Path(a.csv or csv)
    if not src_csv.exists():
        raise SystemExit(f"No data file {src_csv}. Run first:  python main.py collect --tf {a.tf} --asset {a.asset}")
    df = data.load_csv(src_csv)
    print(f"Training on {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    bundle = train_and_validate(df, payout=a.payout, horizons=expiries_to_candles(a),
                                n_folds=a.folds, min_trades=a.min_trades, tf=a.tf)
    bundle["asset"] = a.asset
    model_path.parent.mkdir(exist_ok=True)
    save_bundle(bundle, model_path)
    summary = {f"{h * a.tf}m": (f"TRADE at conf >= {e['threshold']:.2f}" if e["threshold"] else "NO TRADE")
               for h, e in bundle["horizons"].items()}
    print(f"\nSaved model to {model_path}\nSummary: {json.dumps(summary)}")


def cmd_live(a):
    _, model_path = paths(a.asset, a.tf)
    if not model_path.exists():
        raise SystemExit(f"No model {model_path}. Run first:  python main.py train --tf {a.tf} --asset {a.asset}")
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

    print(f"Live on {a.asset}. Signals appear 1 s after each {a.tf}-min candle closes. Ctrl+C to stop.")
    run_live(src, a.asset, bundle, on_signal)


def cmd_strategy(a):
    import pandas as pd
    from otcagent.strategies import STRATEGIES, test_strategy
    if a.list:
        for k, (_, desc) in STRATEGIES.items():
            print(f"{k:<22} {desc}")
        return
    names = list(STRATEGIES) if a.name == "all" else [x.strip() for x in a.name.split(",")]
    bad = [n for n in names if n not in STRATEGIES]
    if bad:
        raise SystemExit(f"unknown strategy {bad}. See: python main.py strategy --list")
    if a.assets:
        assets = [x.strip() for x in a.assets.split(",")]
    else:
        assets = sorted(p.name[: -len(f"_{a.tf}m.csv")] for p in Path("data").glob(f"*_{a.tf}m.csv"))
    if not assets:
        raise SystemExit(f"No {a.tf}-min data in data/. Run collect or scan with --tf {a.tf} first.")
    horizons = expiries_to_candles(a)
    frames = {}
    for asset in assets:
        try:
            frames[asset] = data.load_csv(paths(asset, a.tf)[0])
        except (Exception, SystemExit) as e:
            print(f"skip {asset}: {e}")
    print(f"Testing {len(names)} strateg{'y' if len(names) == 1 else 'ies'} on {len(frames)} assets, "
          f"{a.tf}-min candles, break-even {1 / (1 + a.payout):.2%}")
    rows = []
    for name in names:
        fn, desc = STRATEGIES[name]
        for asset, df in frames.items():
            for r_ in test_strategy(df, fn(df), horizons, a.tf, a.payout):
                rows.append({"strategy": name, "asset": asset, **r_})
    res = pd.DataFrame(rows)
    res.to_csv(f"strategy_results_{a.tf}m.csv", index=False)
    print(f"\n{'strategy':<22}{'exp':>4}{'trades':>8}{'win%':>7}{'pass':>5}")
    for (name, exp), g in res.groupby(["strategy", "expiry"], sort=False):
        n, w = g["trades"].sum(), g["wins"].sum()
        print(f"{name:<22}{exp:>4}{n:>8}{(w / n if n else 0):>7.1%}{int(g['pass'].sum()):>3}/{len(g)}")
    good = res[res["pass"]]
    print(f"\n(win% = all assets together; pass = assets where it beat break-even with "
          f"95% confidence AND on the newest 20%)")
    if len(good):
        print("\nPASSED:")
        for _, g in good.iterrows():
            print(f"  {g['strategy']} {g['asset']} {g['expiry']}: {g['win_rate']:.1%} over "
                  f"{g['trades']} trades, newest 20%: {g['exam_win_rate']:.1%}")
        print("Several strategies x assets were tested - re-check any pass on NEW data before trusting it.")
    else:
        print("No strategy/asset/expiry beat break-even.")
    print(f"Full details: strategy_results_{a.tf}m.csv")


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
        s.add_argument("--tf", type=int, choices=[1, 5], default=1,
                       help="candle size in minutes: 1 (expiries 1/2/3/5) or 5 (expiries 5/10/15)")
    sub.choices["collect"].add_argument("--days", type=float,
                                        help="default 10 (1-min) or 60 (5-min candles)")
    t = sub.choices["train"]
    t.add_argument("--csv", help="train from a CSV instead of data/<asset>_1m.csv")
    t.add_argument("--payout", type=float, default=0.85, help="0.85 = 85%% payout")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--min-trades", type=int, default=100)
    t.add_argument("--expiries", help="minutes, default 1,2,3,5 (--tf 1) or 5,10,15 (--tf 5)")
    lv = sub.choices["live"]
    lv.add_argument("--dashboard", action="store_true")
    lv.add_argument("--host", default="127.0.0.1")
    lv.add_argument("--port", type=int, default=8080)
    sc = sub.add_parser("scan", help="collect + test many OTC assets, list the ones with an edge")
    sc.add_argument("--market", choices=["otc", "real"], default="otc",
                    help="otc = broker OTC pairs (default), real = real forex pairs (Mon-Fri)")
    sc.add_argument("--assets", help="comma list, default: 12 pairs of the chosen --market")
    sc.add_argument("--tf", type=int, choices=[1, 5], default=1,
                    help="candle size in minutes: 1 (expiries 1/2/3/5) or 5 (expiries 5/10/15)")
    sc.add_argument("--days", type=float, help="default 10 (1-min) or 60 (5-min candles)")
    sc.add_argument("--payout", type=float, default=0.85)
    sc.add_argument("--folds", type=int, default=3)
    sc.add_argument("--expiries", help="minutes, default 1,2,3,5 (--tf 1) or 5,10,15 (--tf 5)")
    sc.add_argument("--account", default="PRACTICE", choices=["PRACTICE", "REAL"])
    sc.add_argument("--no-download", action="store_true", help="use CSVs already in data/")
    sc.add_argument("--resume", action="store_true", help="skip assets already in scan_results.csv")
    sc.add_argument("--verbose", action="store_true", help="print full reports")
    st = sub.add_parser("strategy", help="test rule-based strategies on downloaded candles")
    st.add_argument("--name", default="all", help="strategy name(s), comma list, or 'all'")
    st.add_argument("--list", action="store_true", help="show available strategies")
    st.add_argument("--assets", help="comma list, default: every asset with data for --tf")
    st.add_argument("--tf", type=int, choices=[1, 5], default=1)
    st.add_argument("--expiries", help="minutes, default 1,2,3,5 (--tf 1) or 5,10,15 (--tf 5)")
    st.add_argument("--payout", type=float, default=0.85)
    d = sub.add_parser("demo")
    d.add_argument("--rows", type=int, default=15000)
    d.add_argument("--payout", type=float, default=0.85)
    a = p.parse_args()
    {"collect": cmd_collect, "train": cmd_train, "live": cmd_live, "demo": cmd_demo,
     "scan": cmd_scan, "strategy": cmd_strategy}[a.cmd](a)


if __name__ == "__main__":
    main()
